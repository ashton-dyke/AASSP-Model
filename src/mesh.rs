//! The overlapping neural mesh — top-level structure and execution loop.
//!
//! Owns all neurons, circuits, and overlap zones. With LTC/CfC neurons,
//! the step is fully synchronous: compute all new states, then apply.

use crate::circuit::{Circuit, CircuitId};
use crate::config::MeshConfig;
use crate::neuron::{cfc_forward_inference, LtcNeuron};
use crate::overlap::OverlapZone;
use crate::topology::{TopologyConstraint, TopologyError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The central neural mesh structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlappingMesh {
    /// All neurons in the mesh.
    pub neurons: Vec<LtcNeuron>,

    /// All circuits (logical groupings).
    pub circuits: Vec<Circuit>,

    /// Overlap zones between circuit pairs.
    pub overlap_zones: Vec<OverlapZone>,

    /// Topology constraint enforcer.
    pub topology: TopologyConstraint,

    /// Configuration.
    pub config: MeshConfig,

    /// Current simulation tick.
    pub tick: u64,

    /// Input-clamped neurons: these neurons skip CfC forward and
    /// retain their externally set values during stepping.
    /// Set by `encode_input`, cleared by `reset_activations`.
    #[serde(skip)]
    pub clamped_neurons: Vec<bool>,
}

impl OverlappingMesh {
    /// Create a new mesh with the given config.
    pub fn new(config: MeshConfig) -> Self {
        let n = config.total_neurons;
        let neurons = (0..n).map(|_| LtcNeuron::new()).collect();

        Self {
            neurons,
            circuits: Vec::new(),
            overlap_zones: Vec::new(),
            topology: TopologyConstraint::new(),
            config,
            tick: 0,
            clamped_neurons: vec![false; n],
        }
    }

    /// Add a circuit to the mesh (validates topology).
    pub fn add_circuit(&mut self, circuit: Circuit) -> Result<(), TopologyError> {
        // Update membership counts.
        for &idx in &circuit.neuron_indices {
            if idx < self.neurons.len() {
                self.neurons[idx].circuit_membership_count += 1;
            }
        }

        self.circuits.push(circuit);

        // Validate topology.
        self.topology
            .validate(self.neurons.len(), &self.circuits)?;

        Ok(())
    }

    /// Run one synchronous CfC step.
    ///
    /// 1. Compute all new neuron states from the current state (CfC closed-form).
    ///    Clamped neurons (input neurons) skip CfC and retain their values.
    /// 2. Apply new states atomically.
    /// 3. Update circuit confidences.
    pub fn step(&mut self) {
        let dt = self.config.dt;

        // Phase 1: compute all new states from current state (parallel).
        // Clamped neurons retain their externally set values.
        let neurons = &self.neurons;
        let clamped = &self.clamped_neurons;
        let new_states: Vec<f32> = (0..neurons.len())
            .into_par_iter()
            .map(|i| {
                if i < clamped.len() && clamped[i] {
                    neurons[i].x // Keep clamped value.
                } else {
                    cfc_forward_inference(&neurons[i], neurons, dt)
                }
            })
            .collect();

        // Phase 2: apply atomically.
        for (i, x_new) in new_states.into_iter().enumerate() {
            self.neurons[i].x = x_new;
        }

        // Phase 3: update circuit confidences.
        for circuit in &mut self.circuits {
            circuit.update_confidence(&self.neurons);
        }

        self.tick += 1;
    }

    /// Get a circuit by ID.
    pub fn circuit_by_id(&self, id: CircuitId) -> Option<&Circuit> {
        self.circuits.iter().find(|c| c.id == id)
    }

    /// Get a mutable circuit by ID.
    pub fn circuit_by_id_mut(&mut self, id: CircuitId) -> Option<&mut Circuit> {
        self.circuits.iter_mut().find(|c| c.id == id)
    }

    /// Reset all neuron activations to zero.
    pub fn reset_activations(&mut self) {
        for neuron in &mut self.neurons {
            neuron.x = 0.0;
        }
        for c in &mut self.clamped_neurons {
            *c = false;
        }
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{CircuitCriticality, CircuitType};

    fn small_mesh() -> OverlappingMesh {
        let config = MeshConfig {
            total_neurons: 20,
            ..MeshConfig::default()
        };
        let mut mesh = OverlappingMesh::new(config);

        let c0 = Circuit::new(
            0, "det_a", (0..10).collect(), 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        );
        let c1 = Circuit::new(
            1, "det_b", (10..20).collect(), 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        );
        mesh.add_circuit(c0).unwrap();
        mesh.add_circuit(c1).unwrap();
        mesh
    }

    #[test]
    fn step_increments_tick() {
        let mut mesh = small_mesh();
        assert_eq!(mesh.tick, 0);
        mesh.step();
        assert_eq!(mesh.tick, 1);
    }

    #[test]
    fn step_runs_without_panic() {
        let mut mesh = small_mesh();
        for neuron in &mut mesh.neurons {
            neuron.bias = 0.3;
        }
        for _ in 0..100 {
            mesh.step();
        }
    }

    #[test]
    fn step_updates_activations() {
        let mut mesh = small_mesh();
        mesh.neurons[0].bias = 0.5;
        mesh.neurons[0].tau = 0.01;
        mesh.step();
        assert!(mesh.neurons[0].x != 0.0, "Biased neuron should have changed");
    }

    #[test]
    fn topology_rejects_triple_membership() {
        let config = MeshConfig {
            total_neurons: 10,
            ..MeshConfig::default()
        };
        let mut mesh = OverlappingMesh::new(config);

        mesh.add_circuit(Circuit::new(
            0, "a", vec![0, 1, 2], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        )).unwrap();
        mesh.add_circuit(Circuit::new(
            1, "b", vec![1, 2, 3], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        )).unwrap();
        let result = mesh.add_circuit(Circuit::new(
            2, "c", vec![2, 3, 4], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        ));
        assert!(result.is_err());
    }

    #[test]
    fn overlap_neurons_get_counted() {
        let config = MeshConfig {
            total_neurons: 20,
            ..MeshConfig::default()
        };
        let mut mesh = OverlappingMesh::new(config);
        mesh.add_circuit(Circuit::new(
            0, "a", vec![0, 1, 2, 3], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        )).unwrap();
        mesh.add_circuit(Circuit::new(
            1, "b", vec![2, 3, 4, 5], 0.01,
            CircuitType::Causation, CircuitCriticality::Safety,
        )).unwrap();

        assert_eq!(mesh.neurons[2].circuit_membership_count, 2);
        assert_eq!(mesh.neurons[0].circuit_membership_count, 1);
        assert_eq!(mesh.neurons[4].circuit_membership_count, 1);
    }

    #[test]
    fn cfc_step_preserves_finite_state() {
        let mut mesh = small_mesh();
        for n in &mut mesh.neurons {
            n.bias = 0.5;
            n.x = 0.3;
        }
        for _ in 0..1000 {
            mesh.step();
        }
        for n in &mesh.neurons {
            assert!(n.x.is_finite(), "Neuron state should remain finite");
            assert!(n.x.abs() < 2.0, "CfC should bound activations");
        }
    }

    #[test]
    fn reset_activations_zeros_state() {
        let mut mesh = small_mesh();
        mesh.neurons[0].x = 0.5;
        mesh.tick = 42;
        mesh.reset_activations();
        assert_eq!(mesh.neurons[0].x, 0.0);
        assert_eq!(mesh.tick, 0);
    }
}

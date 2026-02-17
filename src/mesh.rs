//! The overlapping neural mesh — top-level structure and execution loop.
//!
//! Owns all neurons, circuits, and overlap zones. Implements the two-phase
//! update cycle: parallel computation → sequential merge.

use crate::circuit::{Circuit, CircuitId, CircuitType};
use crate::config::MeshConfig;
use crate::neuron::LiquidNeuron;
use crate::overlap::OverlapZone;
use crate::topology::{TopologyConstraint, TopologyError};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// The complete overlapping neural mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlappingMesh {
    /// All neurons in the mesh (shared substrate).
    pub neurons: Vec<LiquidNeuron>,

    /// All circuits operating on subsets of neurons.
    pub circuits: Vec<Circuit>,

    /// Overlap zones between circuit pairs.
    pub overlap_zones: Vec<OverlapZone>,

    /// Topology validator.
    pub topology: TopologyConstraint,

    /// Global mesh configuration.
    pub config: MeshConfig,

    /// Current timestep (monotonically increasing).
    pub tick: u64,

    /// Seeded RNG for reproducible stochastic firing.
    #[serde(skip)]
    pub rng: Option<StdRng>,
}

impl OverlappingMesh {
    /// Create a new mesh with the given neuron count and configuration.
    pub fn new(config: MeshConfig) -> Self {
        let neurons = (0..config.total_neurons)
            .map(|_| LiquidNeuron::new())
            .collect();

        Self {
            neurons,
            circuits: Vec::new(),
            overlap_zones: Vec::new(),
            topology: TopologyConstraint::new(),
            config,
            tick: 0,
            rng: Some(StdRng::seed_from_u64(42)),
        }
    }

    /// Add a circuit to the mesh. Validates topology after addition.
    pub fn add_circuit(&mut self, circuit: Circuit) -> Result<(), TopologyError> {
        self.circuits.push(circuit);

        if let Err(e) = self.topology.validate(self.neurons.len(), &self.circuits) {
            self.circuits.pop();
            return Err(e);
        }

        // Update neuron membership counts.
        let last = self.circuits.last().unwrap();
        for &idx in &last.neuron_indices {
            self.neurons[idx].circuit_membership_count += 1;
        }

        Ok(())
    }

    /// Add an overlap zone. The two circuits must already exist.
    pub fn add_overlap_zone(&mut self, zone: OverlapZone) {
        self.overlap_zones.push(zone);
    }

    /// Run one mesh timestep: parallel computation → sequential merge.
    pub fn step(&mut self) {
        let dt = self.config.dt;

        // ═══════════════════════════════════════════
        // PHASE 1: COMPUTATION (collect updates from firing circuits)
        // ═══════════════════════════════════════════
        //
        // We iterate circuits, check firing, and compute updates.
        // The neuron slice is immutable during this phase.
        let mut all_updates: Vec<(CircuitId, Vec<(usize, f32)>)> = Vec::new();

        // We need mutable access to circuits (for should_fire accumulator)
        // and immutable access to neurons simultaneously. Split the borrow
        // by collecting firing decisions and updates in sequence.
        let rng = self.rng.get_or_insert_with(|| StdRng::seed_from_u64(42));

        for circuit in &mut self.circuits {
            if circuit.should_fire(dt, rng) {
                let updates = circuit.compute_updates(&self.neurons, dt);
                circuit.update_confidence(&self.neurons);
                all_updates.push((circuit.id, updates));
            }
        }

        // ═══════════════════════════════════════════
        // PHASE 2: SEQUENTIAL MERGE (handles overlaps)
        // ═══════════════════════════════════════════
        self.merge_updates(&all_updates);

        // ═══════════════════════════════════════════
        // PHASE 3: POST-STEP BOOKKEEPING
        // ═══════════════════════════════════════════
        self.tick += 1;
    }

    /// Merge all proposed updates into the neuron state.
    pub fn merge_updates(&mut self, updates: &[(CircuitId, Vec<(usize, f32)>)]) {
        // Build map: neuron_index → Vec<(circuit_id, delta)>
        let mut neuron_updates: Vec<Vec<(CircuitId, f32)>> = vec![Vec::new(); self.neurons.len()];

        for (circuit_id, circuit_updates) in updates {
            for &(neuron_idx, delta) in circuit_updates {
                neuron_updates[neuron_idx].push((*circuit_id, delta));
            }
        }

        // Apply updates
        for (neuron_idx, pending) in neuron_updates.iter().enumerate() {
            match pending.len() {
                0 => {} // No update this timestep.
                1 => {
                    // Single circuit — direct application.
                    self.neurons[neuron_idx].x += pending[0].1;
                }
                2 => {
                    // Overlap zone — use adaptive merge.
                    let delta = self.merge_overlap(neuron_idx, pending);
                    self.neurons[neuron_idx].x += delta;
                }
                _ => {
                    // Should never happen (topology enforces max 2).
                    log::error!(
                        "Neuron {} has {} pending updates (max 2 expected)",
                        neuron_idx,
                        pending.len()
                    );
                    // Fallback: use highest-confidence circuit.
                    let best = pending
                        .iter()
                        .max_by(|a, b| {
                            self.get_circuit_confidence(a.0)
                                .partial_cmp(&self.get_circuit_confidence(b.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap();
                    self.neurons[neuron_idx].x += best.1;
                }
            }
        }
    }

    /// Merge two competing updates for an overlap neuron.
    fn merge_overlap(&self, _neuron_idx: usize, pending: &[(CircuitId, f32)]) -> f32 {
        debug_assert!(pending.len() == 2);

        let (circuit_a_id, delta_a) = pending[0];
        let (circuit_b_id, delta_b) = pending[1];

        let circuit_a = self.get_circuit(circuit_a_id);
        let circuit_b = self.get_circuit(circuit_b_id);

        // Find the overlap zone for these two circuits.
        if let Some(zone) = self.find_overlap_zone(circuit_a_id, circuit_b_id) {
            // Safety override: detection circuit with high confidence wins.
            let context_threshold =
                zone.gate
                    .adjust_threshold(zone.safety_threshold, circuit_a, circuit_b);

            if circuit_a.circuit_type == CircuitType::Detection
                && circuit_a.confidence > context_threshold
            {
                return delta_a;
            }
            if circuit_b.circuit_type == CircuitType::Detection
                && circuit_b.confidence > context_threshold
            {
                return delta_b;
            }

            // Learned gate merge.
            let (priority_a, priority_b) =
                zone.gate
                    .compute_priorities(circuit_a, delta_a, circuit_b, delta_b);

            delta_a * priority_a + delta_b * priority_b
        } else {
            // No overlap zone registered — fall back to average.
            (delta_a + delta_b) * 0.5
        }
    }

    /// Look up a circuit by ID.
    fn get_circuit(&self, id: CircuitId) -> &Circuit {
        self.circuits
            .iter()
            .find(|c| c.id == id)
            .expect("Circuit ID not found in mesh")
    }

    /// Get a circuit's confidence by ID.
    fn get_circuit_confidence(&self, id: CircuitId) -> f32 {
        self.circuits
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.confidence)
            .unwrap_or(0.0)
    }

    /// Find the overlap zone connecting two circuits (order-independent).
    fn find_overlap_zone(&self, a: CircuitId, b: CircuitId) -> Option<&OverlapZone> {
        self.overlap_zones.iter().find(|z| z.connects(a, b))
    }

    /// Validate the current topology.
    pub fn validate_topology(&self) -> Result<(), TopologyError> {
        self.topology.validate(self.neurons.len(), &self.circuits)
    }

    /// Check if the mesh is healthy (all neurons finite, circuits present).
    pub fn is_healthy(&self) -> bool {
        self.neurons.iter().all(|n| n.x.is_finite())
            && !self.circuits.is_empty()
    }

    /// Get the total neuron count.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Get the total circuit count.
    pub fn circuit_count(&self) -> usize {
        self.circuits.len()
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
        OverlappingMesh::new(config)
    }

    fn detection_circuit(id: u32, indices: Vec<usize>) -> Circuit {
        Circuit::new(
            id,
            format!("det_{id}"),
            indices,
            0.01,
            CircuitType::Detection,
            CircuitCriticality::Safety,
        )
    }

    #[test]
    fn add_circuit_validates_topology() {
        let mut mesh = small_mesh();
        // Two circuits sharing neurons — should be OK.
        mesh.add_circuit(detection_circuit(0, vec![0, 1, 2, 3]))
            .unwrap();
        mesh.add_circuit(detection_circuit(1, vec![2, 3, 4, 5]))
            .unwrap();

        // Third circuit also claiming neuron 2 — should fail.
        let result = mesh.add_circuit(detection_circuit(2, vec![2, 6, 7]));
        assert!(result.is_err());
        // Mesh should still only have 2 circuits (rollback).
        assert_eq!(mesh.circuit_count(), 2);
    }

    #[test]
    fn step_does_not_panic_with_no_circuits() {
        let mut mesh = small_mesh();
        for _ in 0..100 {
            mesh.step();
        }
        assert_eq!(mesh.tick, 100);
    }

    #[test]
    fn step_updates_neurons() {
        let mut mesh = small_mesh();
        // Give neuron 0 a bias so it has something to compute.
        mesh.neurons[0].bias = 0.5;

        mesh.add_circuit(detection_circuit(0, vec![0, 1, 2]))
            .unwrap();

        let initial_x = mesh.neurons[0].x;

        // Run enough steps for the Safety circuit to fire (tau=0.01, dt=0.001 → fires every 10 steps).
        for _ in 0..10 {
            mesh.step();
        }

        // Neuron 0 should have changed due to its bias.
        assert_ne!(mesh.neurons[0].x, initial_x);
    }

    #[test]
    fn overlap_merge_works() {
        let mut mesh = small_mesh();

        // Set up neurons with connections for meaningful computation.
        mesh.neurons[5].bias = 0.5;
        mesh.neurons[6].bias = -0.3;

        // Two circuits sharing neurons 5 and 6.
        mesh.add_circuit(detection_circuit(0, vec![0, 1, 2, 3, 4, 5, 6]))
            .unwrap();

        let mut causation = Circuit::new(
            1,
            "causation",
            vec![5, 6, 7, 8, 9],
            0.01,
            CircuitType::Causation,
            CircuitCriticality::Safety,
        );
        causation.confidence = 0.3;
        mesh.add_circuit(causation).unwrap();

        mesh.add_overlap_zone(OverlapZone::new(0, 1, vec![5, 6]));

        // Run steps — the merge should handle overlap neurons without panicking.
        for _ in 0..100 {
            mesh.step();
        }

        assert!(mesh.is_healthy());
    }

    #[test]
    fn safety_override_in_merge() {
        let mut mesh = small_mesh();

        // Detection circuit with very high confidence.
        let mut det = detection_circuit(0, vec![0, 1, 2]);
        det.confidence = 0.95;
        mesh.add_circuit(det).unwrap();

        let mut pred = Circuit::new(
            1,
            "prediction",
            vec![1, 2, 3],
            0.01,
            CircuitType::Prediction,
            CircuitCriticality::Safety,
        );
        pred.confidence = 0.1;
        mesh.add_circuit(pred).unwrap();

        let zone = OverlapZone::new(0, 1, vec![1, 2]);
        mesh.add_overlap_zone(zone);

        // The detection circuit should win the merge due to safety override.
        let delta = mesh.merge_overlap(
            1,
            &[(0, 0.5), (1, -0.3)],
        );
        // Detection wins, so delta should be 0.5.
        assert!((delta - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ten_thousand_steps_no_panic() {
        let mut mesh = small_mesh();
        mesh.neurons[0].bias = 0.2;
        mesh.neurons[5].bias = -0.1;

        mesh.add_circuit(detection_circuit(0, vec![0, 1, 2, 3, 4]))
            .unwrap();
        mesh.add_circuit(Circuit::new(
            1,
            "mem",
            vec![5, 6, 7, 8, 9],
            0.2,
            CircuitType::Memory,
            CircuitCriticality::Performance,
        ))
        .unwrap();

        for _ in 0..10_000 {
            mesh.step();
        }

        assert_eq!(mesh.tick, 10_000);
        assert!(mesh.is_healthy());
    }

    #[test]
    fn safety_circuit_determinism() {
        // Two meshes with identical setup should produce bit-identical
        // safety circuit results.
        let setup = || {
            let mut mesh = small_mesh();
            mesh.neurons[0].bias = 0.3;
            mesh.neurons[1].bias = -0.2;
            mesh.add_circuit(detection_circuit(0, vec![0, 1, 2, 3, 4]))
                .unwrap();
            mesh
        };

        let mut mesh_a = setup();
        let mut mesh_b = setup();

        for _ in 0..500 {
            mesh_a.step();
            mesh_b.step();
        }

        for i in 0..5 {
            assert_eq!(
                mesh_a.neurons[i].x, mesh_b.neurons[i].x,
                "Neuron {i} diverged between identical meshes"
            );
        }
    }
}

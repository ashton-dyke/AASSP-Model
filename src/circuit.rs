//! Circuit system — logical groupings of neurons with firing control.

use crate::neuron::{compute_neuron_update, LiquidNeuron};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Unique identifier for a circuit.
pub type CircuitId = u32;

/// What function a circuit performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitType {
    /// Fast anomaly detection.
    Detection,
    /// Root cause analysis.
    Causation,
    /// Forward state projection.
    Prediction,
    /// Working memory (short/medium/long).
    Memory,
}

/// Determines the firing mode of a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitCriticality {
    /// Deterministic firing. Fires when accumulator exceeds tau.
    /// Used for detection and causation circuits.
    Safety,

    /// Stochastic firing. Fires with probability dt/tau per timestep.
    /// Used for memory and prediction circuits.
    Performance,
}

/// A logical grouping of neurons that performs a specific function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Unique identifier.
    pub id: CircuitId,

    /// Human-readable name (e.g. "kick_detection").
    pub name: String,

    /// Which neurons this circuit reads and writes.
    pub neuron_indices: Vec<usize>,

    /// Time constant in seconds. Small tau = fast updates.
    pub tau: f32,

    /// Circuit function type.
    pub circuit_type: CircuitType,

    /// Criticality level (determines firing mode).
    pub criticality: CircuitCriticality,

    /// Current confidence output (0.0 to 1.0).
    pub confidence: f32,

    /// Deterministic accumulator for Safety firing.
    pub fire_accumulator: f32,

    /// Whether this circuit's weights are frozen.
    pub frozen: bool,
}

impl Circuit {
    /// Create a new circuit.
    pub fn new(
        id: CircuitId,
        name: impl Into<String>,
        neuron_indices: Vec<usize>,
        tau: f32,
        circuit_type: CircuitType,
        criticality: CircuitCriticality,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            neuron_indices,
            tau,
            circuit_type,
            criticality,
            confidence: 0.0,
            fire_accumulator: 0.0,
            frozen: false,
        }
    }

    /// Determine whether this circuit should fire this timestep.
    pub fn should_fire(&mut self, dt: f32, rng: &mut impl Rng) -> bool {
        match self.criticality {
            CircuitCriticality::Safety => {
                self.fire_accumulator += dt;
                if self.fire_accumulator >= self.tau {
                    self.fire_accumulator -= self.tau;
                    true
                } else {
                    false
                }
            }
            CircuitCriticality::Performance => rng.gen::<f32>() < (dt / self.tau),
        }
    }

    /// Compute proposed updates for all neurons in this circuit.
    ///
    /// Returns Vec of (neuron_index, proposed_delta).
    /// Does NOT modify neuron state — the caller handles merge.
    pub fn compute_updates(&self, neurons: &[LiquidNeuron], dt: f32) -> Vec<(usize, f32)> {
        let mut updates = Vec::with_capacity(self.neuron_indices.len());

        for &neuron_idx in &self.neuron_indices {
            let delta = compute_neuron_update(&neurons[neuron_idx], neurons, self.tau, dt);
            updates.push((neuron_idx, delta));
        }

        updates
    }

    /// Update the circuit's confidence based on its output neurons.
    ///
    /// Confidence is the mean absolute activation of the circuit's neurons,
    /// clamped to [0, 1].
    pub fn update_confidence(&mut self, neurons: &[LiquidNeuron]) {
        if self.neuron_indices.is_empty() {
            self.confidence = 0.0;
            return;
        }

        let sum: f32 = self
            .neuron_indices
            .iter()
            .map(|&idx| neurons[idx].x.abs())
            .sum();

        self.confidence = (sum / self.neuron_indices.len() as f32).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::LiquidNeuron;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_test_circuit(
        criticality: CircuitCriticality,
        tau: f32,
        neuron_count: usize,
    ) -> Circuit {
        Circuit::new(
            0,
            "test",
            (0..neuron_count).collect(),
            tau,
            CircuitType::Detection,
            criticality,
        )
    }

    #[test]
    fn safety_fires_at_exact_tau_intervals() {
        let mut circuit = make_test_circuit(CircuitCriticality::Safety, 0.01, 0);
        let mut rng = StdRng::seed_from_u64(42);
        let dt = 0.001;
        let mut fire_count = 0;

        for _ in 0..1000 {
            if circuit.should_fire(dt, &mut rng) {
                fire_count += 1;
            }
        }

        // tau=0.01, dt=0.001 → fires every 10 steps → 100 fires in 1000 steps
        assert_eq!(fire_count, 100);
    }

    #[test]
    fn safety_firing_is_deterministic() {
        let dt = 0.001;

        let mut circuit1 = make_test_circuit(CircuitCriticality::Safety, 0.01, 0);
        let mut circuit2 = make_test_circuit(CircuitCriticality::Safety, 0.01, 0);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(99); // different seed — shouldn't matter

        for _ in 0..500 {
            let f1 = circuit1.should_fire(dt, &mut rng1);
            let f2 = circuit2.should_fire(dt, &mut rng2);
            assert_eq!(f1, f2, "Safety firing must be deterministic regardless of RNG");
        }
    }

    #[test]
    fn performance_fires_approximately_correct_rate() {
        let mut circuit = make_test_circuit(CircuitCriticality::Performance, 0.2, 0);
        let mut rng = StdRng::seed_from_u64(42);
        let dt = 0.001;
        let steps = 100_000;
        let mut fire_count = 0;

        for _ in 0..steps {
            if circuit.should_fire(dt, &mut rng) {
                fire_count += 1;
            }
        }

        // Expected: steps * (dt / tau) = 100000 * 0.005 = 500
        let expected = (steps as f32 * dt / circuit.tau) as i32;
        let tolerance = (expected as f32 * 0.15) as i32; // 15% tolerance
        assert!(
            (fire_count - expected).abs() < tolerance,
            "Expected ~{expected} fires, got {fire_count}"
        );
    }

    #[test]
    fn compute_updates_returns_correct_count() {
        let neurons: Vec<LiquidNeuron> = (0..10).map(|_| LiquidNeuron::new()).collect();
        let circuit = make_test_circuit(CircuitCriticality::Safety, 0.01, 5);
        let updates = circuit.compute_updates(&neurons, 0.001);
        assert_eq!(updates.len(), 5);
    }

    #[test]
    fn confidence_reflects_neuron_activations() {
        let mut neurons: Vec<LiquidNeuron> = (0..4).map(|_| LiquidNeuron::new()).collect();
        neurons[0].x = 0.5;
        neurons[1].x = -0.3;
        neurons[2].x = 0.7;
        neurons[3].x = 0.1;

        let mut circuit = make_test_circuit(CircuitCriticality::Safety, 0.01, 4);
        circuit.update_confidence(&neurons);

        let expected = (0.5 + 0.3 + 0.7 + 0.1) / 4.0;
        assert!((circuit.confidence - expected).abs() < 1e-5);
    }
}

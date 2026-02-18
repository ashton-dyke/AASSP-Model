//! Circuit system — logical groupings of neurons.
//!
//! Circuits are purely organizational: they define which neurons belong to
//! which functional group, for training, readout, and initialization.
//! With LTC/CfC neurons handling their own temporal dynamics, circuits
//! no longer control firing schedules.

use crate::neuron::LtcNeuron;
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

/// Safety criticality level — used for output-level safety override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitCriticality {
    /// Safety-critical circuit (detection, causation).
    Safety,
    /// Performance circuit (memory, prediction).
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

    /// Suggested initial time constant for neurons in this circuit.
    /// Used during initialization to set per-neuron tau values.
    pub tau: f32,

    /// Circuit function type.
    pub circuit_type: CircuitType,

    /// Criticality level (used for safety override at output level).
    pub criticality: CircuitCriticality,

    /// Current confidence output (0.0 to 1.0).
    pub confidence: f32,

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
            frozen: false,
        }
    }

    /// Update confidence based on readout neurons.
    ///
    /// Uses the mean absolute activation of the last 8 neurons.
    pub fn update_confidence(&mut self, neurons: &[LtcNeuron]) {
        let n = self.neuron_indices.len();
        if n == 0 {
            self.confidence = 0.0;
            return;
        }

        let readout_count = 8.min(n);
        let start = n - readout_count;
        let mean_abs: f32 = self.neuron_indices[start..]
            .iter()
            .map(|&idx| neurons[idx].x.abs())
            .sum::<f32>()
            / readout_count as f32;

        self.confidence = mean_abs.clamp(0.0, 1.0);
    }

    /// Compute proposed CfC updates for all neurons in this circuit.
    ///
    /// Returns Vec of (neuron_index, new_x_value).
    pub fn compute_cfc_updates(
        &self,
        neurons: &[LtcNeuron],
        dt: f32,
    ) -> Vec<(usize, f32)> {
        self.neuron_indices
            .iter()
            .map(|&idx| {
                let new_x = crate::neuron::cfc_forward_inference(&neurons[idx], neurons, dt);
                (idx, new_x)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_neurons(n: usize) -> Vec<LtcNeuron> {
        (0..n).map(|_| LtcNeuron::new()).collect()
    }

    #[test]
    fn circuit_creation() {
        let c = Circuit::new(
            0, "test", vec![0, 1, 2], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        );
        assert_eq!(c.id, 0);
        assert_eq!(c.neuron_indices.len(), 3);
        assert_eq!(c.confidence, 0.0);
        assert!(!c.frozen);
    }

    #[test]
    fn confidence_update() {
        let mut neurons = make_neurons(10);
        neurons[7].x = 0.8;
        neurons[8].x = 0.6;
        neurons[9].x = 0.4;

        let mut c = Circuit::new(
            0, "det", (0..10).collect(), 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        );
        c.update_confidence(&neurons);
        assert!(c.confidence > 0.0);
    }

    #[test]
    fn cfc_updates_produce_values() {
        let mut neurons = make_neurons(5);
        neurons[0].bias = 0.3;
        neurons[0].tau = 0.01;

        let c = Circuit::new(
            0, "det", vec![0, 1, 2], 0.01,
            CircuitType::Detection, CircuitCriticality::Safety,
        );
        let updates = c.compute_cfc_updates(&neurons, 0.001);
        assert_eq!(updates.len(), 3);
    }
}

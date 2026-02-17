//! Overlap zones and gating networks.
//!
//! Overlap zones define shared neurons between exactly two circuits.
//! Each zone has a learned gating network that determines merge priority.

use crate::circuit::{Circuit, CircuitId};
use serde::{Deserialize, Serialize};

/// Small MLP that determines merge priority between two circuits.
///
/// Input: [conf_a, tau_a, delta_a, conf_b, tau_b, delta_b] (6 features)
/// Hidden: 8 neurons, ReLU activation
/// Output: 2 logits → softmax → (priority_a, priority_b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapGate {
    /// Input → Hidden weights (8 x 6).
    pub weights_ih: [[f32; 6]; 8],
    /// Hidden biases.
    pub bias_h: [f32; 8],
    /// Hidden → Output weights (2 x 8).
    pub weights_ho: [[f32; 8]; 2],
    /// Output biases.
    pub bias_o: [f32; 2],
}

impl OverlapGate {
    /// Create a gate initialised to produce roughly equal priorities.
    pub fn new() -> Self {
        Self {
            weights_ih: [[0.0; 6]; 8],
            bias_h: [0.0; 8],
            weights_ho: [[0.0; 8]; 2],
            bias_o: [0.0; 2],
        }
    }

    /// Compute merge priorities for two circuits.
    /// Returns (priority_a, priority_b) where both sum to 1.0.
    pub fn compute_priorities(
        &self,
        circuit_a: &Circuit,
        delta_a: f32,
        circuit_b: &Circuit,
        delta_b: f32,
    ) -> (f32, f32) {
        let input = [
            circuit_a.confidence,
            circuit_a.tau,
            delta_a,
            circuit_b.confidence,
            circuit_b.tau,
            delta_b,
        ];

        // Hidden layer with ReLU
        let mut hidden = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = self.bias_h[i];
            for j in 0..6 {
                sum += self.weights_ih[i][j] * input[j];
            }
            hidden[i] = sum.max(0.0);
        }

        // Output layer
        let mut logits = [0.0f32; 2];
        for i in 0..2 {
            let mut sum = self.bias_o[i];
            for j in 0..8 {
                sum += self.weights_ho[i][j] * hidden[j];
            }
            logits[i] = sum;
        }

        // Softmax
        let max_logit = logits[0].max(logits[1]);
        let exp_a = (logits[0] - max_logit).exp();
        let exp_b = (logits[1] - max_logit).exp();
        let sum = exp_a + exp_b;

        (exp_a / sum, exp_b / sum)
    }

    /// Adjust the safety override threshold based on context.
    /// Returns a threshold in [0.5, 0.95].
    pub fn adjust_threshold(
        &self,
        base_threshold: f32,
        circuit_a: &Circuit,
        circuit_b: &Circuit,
    ) -> f32 {
        let disagreement = (circuit_a.confidence - circuit_b.confidence).abs();
        let adjusted = base_threshold - (disagreement * 0.1);
        adjusted.clamp(0.5, 0.95)
    }
}

impl Default for OverlapGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Defines the overlap between exactly two circuits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapZone {
    /// The two circuits that share neurons in this zone.
    pub circuit_a: CircuitId,
    pub circuit_b: CircuitId,

    /// Neuron indices that belong to BOTH circuits.
    pub shared_neuron_indices: Vec<usize>,

    /// Learned soft masks per shared neuron (0.0 to 1.0).
    /// During training, neurons with mask < PRUNE_THRESHOLD are removed.
    pub masks: Vec<f32>,

    /// Learned gating network for this overlap zone.
    pub gate: OverlapGate,

    /// Base priority threshold for safety override.
    pub safety_threshold: f32,
}

impl OverlapZone {
    /// Create a new overlap zone between two circuits.
    ///
    /// Masks are initialised to 0.5 as specified.
    pub fn new(
        circuit_a: CircuitId,
        circuit_b: CircuitId,
        shared_neuron_indices: Vec<usize>,
    ) -> Self {
        let mask_count = shared_neuron_indices.len();
        Self {
            circuit_a,
            circuit_b,
            shared_neuron_indices,
            masks: vec![0.5; mask_count],
            gate: OverlapGate::new(),
            safety_threshold: 0.75,
        }
    }

    /// Return indices of neurons whose mask exceeds the threshold.
    pub fn active_neuron_indices(&self, threshold: f32) -> Vec<usize> {
        self.shared_neuron_indices
            .iter()
            .zip(&self.masks)
            .filter(|(_, &mask)| mask > threshold)
            .map(|(&idx, _)| idx)
            .collect()
    }

    /// L1 regularisation loss to encourage sparse overlaps.
    pub fn sparsity_loss(&self) -> f32 {
        self.masks.iter().map(|m| m.abs()).sum()
    }

    /// Check if this zone connects the given pair of circuit IDs (order-independent).
    pub fn connects(&self, a: CircuitId, b: CircuitId) -> bool {
        (self.circuit_a == a && self.circuit_b == b)
            || (self.circuit_a == b && self.circuit_b == a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, CircuitCriticality, CircuitType};

    fn make_circuit(id: u32, tau: f32, confidence: f32, ct: CircuitType) -> Circuit {
        let mut c = Circuit::new(
            id,
            format!("circuit_{id}"),
            vec![],
            tau,
            ct,
            CircuitCriticality::Safety,
        );
        c.confidence = confidence;
        c
    }

    #[test]
    fn gate_priorities_sum_to_one() {
        let gate = OverlapGate::new();
        let ca = make_circuit(0, 0.01, 0.5, CircuitType::Detection);
        let cb = make_circuit(1, 0.05, 0.3, CircuitType::Causation);

        let (pa, pb) = gate.compute_priorities(&ca, 0.1, &cb, -0.05);
        let sum = pa + pb;
        assert!((sum - 1.0).abs() < 1e-6, "Priorities must sum to 1.0, got {sum}");
    }

    #[test]
    fn zero_gate_gives_equal_priorities() {
        let gate = OverlapGate::new(); // all zeros
        let ca = make_circuit(0, 0.01, 0.5, CircuitType::Detection);
        let cb = make_circuit(1, 0.01, 0.5, CircuitType::Detection);

        let (pa, pb) = gate.compute_priorities(&ca, 0.0, &cb, 0.0);
        assert!((pa - 0.5).abs() < 1e-6);
        assert!((pb - 0.5).abs() < 1e-6);
    }

    #[test]
    fn threshold_adjustment_lowers_on_disagreement() {
        let gate = OverlapGate::new();
        let ca = make_circuit(0, 0.01, 0.9, CircuitType::Detection);
        let cb = make_circuit(1, 0.01, 0.1, CircuitType::Detection);

        let base = 0.75;
        let adjusted = gate.adjust_threshold(base, &ca, &cb);
        // disagreement = 0.8, adjustment = 0.08
        // 0.75 - 0.08 = 0.67
        assert!((adjusted - 0.67).abs() < 1e-5);
    }

    #[test]
    fn threshold_clamped_to_range() {
        let gate = OverlapGate::new();
        let ca = make_circuit(0, 0.01, 1.0, CircuitType::Detection);
        let cb = make_circuit(1, 0.01, 0.0, CircuitType::Detection);

        // disagreement = 1.0, base - 0.1 = 0.4 → clamped to 0.5
        let adjusted = gate.adjust_threshold(0.5, &ca, &cb);
        assert!((adjusted - 0.5).abs() < 1e-5);
    }

    #[test]
    fn active_neurons_filtered_by_mask() {
        let mut zone = OverlapZone::new(0, 1, vec![10, 11, 12, 13, 14]);
        zone.masks = vec![0.1, 0.5, 0.8, 0.2, 0.9];

        let active = zone.active_neuron_indices(0.3);
        assert_eq!(active, vec![11, 12, 14]);
    }

    #[test]
    fn sparsity_loss_is_non_negative() {
        let zone = OverlapZone::new(0, 1, vec![0, 1, 2]);
        assert!(zone.sparsity_loss() >= 0.0);
    }

    #[test]
    fn connects_is_order_independent() {
        let zone = OverlapZone::new(3, 7, vec![]);
        assert!(zone.connects(3, 7));
        assert!(zone.connects(7, 3));
        assert!(!zone.connects(3, 8));
    }
}

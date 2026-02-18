//! Overlap zones between circuits.
//!
//! Overlap zones define shared neurons between exactly two circuits.
//! With LTC/CfC neurons, the gate dynamics are implicit in the neuron's
//! input-dependent time constant. The overlap zone tracks shared neurons
//! and learned masks for sparsity regularization during training.

use crate::circuit::CircuitId;
use serde::{Deserialize, Serialize};

/// Defines the overlap between exactly two circuits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapZone {
    /// The two circuits that share neurons in this zone.
    pub circuit_a: CircuitId,
    pub circuit_b: CircuitId,

    /// Neuron indices that belong to BOTH circuits.
    pub shared_neuron_indices: Vec<usize>,

    /// Learned soft masks per shared neuron (0.0 to 1.0).
    /// Used for sparsity regularization during training.
    pub masks: Vec<f32>,

    /// Base priority threshold for safety override at output level.
    pub safety_threshold: f32,
}

impl OverlapZone {
    /// Create a new overlap zone between two circuits.
    ///
    /// Masks are initialized to 0.5.
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

//! Topology constraint enforcement.
//!
//! Hard constraint: each neuron may belong to at most 2 circuits.

use crate::circuit::Circuit;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TopologyError {
    #[error(
        "Neuron {neuron_idx} belongs to {count} circuits (max {max})"
    )]
    ExcessMembership {
        neuron_idx: usize,
        count: u8,
        max: u8,
    },

    #[error("Neuron index {0} is out of bounds (total neurons: {1})")]
    NeuronOutOfBounds(usize, usize),
}

/// Enforces the max-circuits-per-neuron invariant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConstraint {
    /// Maximum circuits any single neuron may belong to. Always 2.
    pub max_circuits_per_neuron: u8,
}

impl TopologyConstraint {
    pub fn new() -> Self {
        Self {
            max_circuits_per_neuron: 2,
        }
    }

    /// Validate the entire mesh topology.
    /// Returns Err with details if any neuron exceeds the membership limit
    /// or any neuron index is out of bounds.
    pub fn validate(
        &self,
        neuron_count: usize,
        circuits: &[Circuit],
    ) -> Result<(), TopologyError> {
        let mut membership_count = vec![0u8; neuron_count];

        for circuit in circuits {
            for &neuron_idx in &circuit.neuron_indices {
                if neuron_idx >= neuron_count {
                    return Err(TopologyError::NeuronOutOfBounds(neuron_idx, neuron_count));
                }
                membership_count[neuron_idx] += 1;
                if membership_count[neuron_idx] > self.max_circuits_per_neuron {
                    return Err(TopologyError::ExcessMembership {
                        neuron_idx,
                        count: membership_count[neuron_idx],
                        max: self.max_circuits_per_neuron,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compute the membership count for every neuron. Useful for
    /// identifying overlap neurons (count == 2) vs exclusive neurons (count == 1).
    pub fn membership_counts(
        &self,
        neuron_count: usize,
        circuits: &[Circuit],
    ) -> Vec<u8> {
        let mut counts = vec![0u8; neuron_count];
        for circuit in circuits {
            for &idx in &circuit.neuron_indices {
                if idx < neuron_count {
                    counts[idx] += 1;
                }
            }
        }
        counts
    }
}

impl Default for TopologyConstraint {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{CircuitCriticality, CircuitType};

    fn make_circuit(id: u32, indices: Vec<usize>) -> Circuit {
        Circuit::new(
            id,
            format!("circuit_{id}"),
            indices,
            0.01,
            CircuitType::Detection,
            CircuitCriticality::Safety,
        )
    }

    #[test]
    fn accepts_valid_topology_no_overlap() {
        let tc = TopologyConstraint::new();
        let circuits = vec![
            make_circuit(0, vec![0, 1, 2]),
            make_circuit(1, vec![3, 4, 5]),
        ];
        assert!(tc.validate(6, &circuits).is_ok());
    }

    #[test]
    fn accepts_valid_topology_with_overlap() {
        let tc = TopologyConstraint::new();
        let circuits = vec![
            make_circuit(0, vec![0, 1, 2, 3]),
            make_circuit(1, vec![2, 3, 4, 5]), // neurons 2,3 shared
        ];
        assert!(tc.validate(6, &circuits).is_ok());
    }

    #[test]
    fn rejects_three_circuits_per_neuron() {
        let tc = TopologyConstraint::new();
        let circuits = vec![
            make_circuit(0, vec![0, 1, 2]),
            make_circuit(1, vec![1, 2, 3]),
            make_circuit(2, vec![2, 3, 4]), // neuron 2 now in 3 circuits
        ];
        let err = tc.validate(5, &circuits).unwrap_err();
        match err {
            TopologyError::ExcessMembership { neuron_idx, count, max } => {
                assert_eq!(neuron_idx, 2);
                assert_eq!(count, 3);
                assert_eq!(max, 2);
            }
            _ => panic!("Expected ExcessMembership error"),
        }
    }

    #[test]
    fn rejects_out_of_bounds_neuron() {
        let tc = TopologyConstraint::new();
        let circuits = vec![make_circuit(0, vec![0, 1, 99])];
        assert!(matches!(
            tc.validate(10, &circuits),
            Err(TopologyError::NeuronOutOfBounds(99, 10))
        ));
    }

    #[test]
    fn membership_counts_correct() {
        let tc = TopologyConstraint::new();
        let circuits = vec![
            make_circuit(0, vec![0, 1, 2]),
            make_circuit(1, vec![1, 2, 3]),
        ];
        let counts = tc.membership_counts(4, &circuits);
        assert_eq!(counts, vec![1, 2, 2, 1]);
    }
}

//! Training data utilities: batching, shuffling, label conversion.

use crate::training::synthetic::{AnomalyType, TrainingSequence};

/// Detection circuit index mapping.
/// Maps anomaly type to detection circuit index in the default layout.
pub fn anomaly_to_circuit_idx(anomaly: AnomalyType) -> Option<usize> {
    match anomaly {
        AnomalyType::Kick => Some(0),
        AnomalyType::Loss => Some(1),
        AnomalyType::PackOff => Some(2),
        AnomalyType::StickSlip => Some(3),
        AnomalyType::Founder => Some(4),
        AnomalyType::Normal => None,
    }
}

/// Convert a training sequence into detection labels for a specific sample index.
///
/// Returns Vec of (circuit_idx, target) where target is 1.0 for active anomaly.
pub fn detection_labels_for_sample(
    sequence: &TrainingSequence,
    sample_idx: usize,
) -> Vec<(usize, f32)> {
    // All 5 detection circuits get a label.
    let mut labels: Vec<(usize, f32)> = (0..5).map(|ci| (ci, 0.0)).collect();

    if sample_idx < sequence.labels.len() {
        let anomaly = sequence.labels[sample_idx];
        if let Some(ci) = anomaly_to_circuit_idx(anomaly) {
            labels[ci].1 = 1.0;
        }
    }

    labels
}

/// Convert an anomaly type to causation labels.
///
/// Returns Vec of (causation_circuit_idx, Vec<(readout_pos, target)>).
/// Readout positions map to: 0=torque, 1=pressure, 2=flow_balance, 3=wob, 4=spp, 5=rop, 6=gas, 7=rpm.
pub fn causation_labels_for_sample(
    sequence: &TrainingSequence,
    sample_idx: usize,
) -> Vec<(usize, Vec<(usize, f32)>)> {
    let anomaly = if sample_idx < sequence.labels.len() {
        sequence.labels[sample_idx]
    } else {
        AnomalyType::Normal
    };

    // Causation circuits: 5=torque, 6=pressure, 7=flow.
    let mut labels = Vec::new();

    match anomaly {
        AnomalyType::Kick => {
            // Flow causation (circuit 7): flow_balance is causal.
            labels.push((7, vec![(2, 1.0)]));
        }
        AnomalyType::Loss => {
            // Flow causation: flow_balance is causal.
            labels.push((7, vec![(2, 1.0)]));
            // Pressure causation: SPP changes.
            labels.push((6, vec![(4, 1.0)]));
        }
        AnomalyType::PackOff => {
            // Torque causation: torque is causal.
            labels.push((5, vec![(0, 1.0)]));
            // Pressure causation: SPP is causal.
            labels.push((6, vec![(4, 1.0)]));
        }
        AnomalyType::StickSlip => {
            // Torque causation: torque and RPM.
            labels.push((5, vec![(0, 1.0), (7, 1.0)]));
        }
        AnomalyType::Founder => {
            // Torque causation: WOB is causal.
            labels.push((5, vec![(3, 1.0)]));
        }
        AnomalyType::Normal => {}
    }

    labels
}

/// Shuffle indices using Fisher-Yates.
pub fn shuffle_indices(n: usize, rng: &mut impl rand::Rng) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::synthetic::SyntheticWellGenerator;

    #[test]
    fn detection_labels_normal() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.normal_sequence(10);
        let labels = detection_labels_for_sample(&seq, 0);
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().all(|(_, t)| *t == 0.0));
    }

    #[test]
    fn detection_labels_kick() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.kick_sequence(50, 10, 0.8);
        let labels = detection_labels_for_sample(&seq, 20);
        // Circuit 0 (kick_detection) should be 1.0.
        assert!((labels[0].1 - 1.0).abs() < 1e-6);
        // Others should be 0.0.
        assert!((labels[1].1).abs() < 1e-6);
    }

    #[test]
    fn causation_labels_packoff() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.packoff_sequence(50, 10, 0.8);
        let labels = causation_labels_for_sample(&seq, 15);
        // Should have torque and pressure causation entries.
        assert!(!labels.is_empty());
        let has_torque = labels.iter().any(|(ci, _)| *ci == 5);
        let has_pressure = labels.iter().any(|(ci, _)| *ci == 6);
        assert!(has_torque);
        assert!(has_pressure);
    }

    #[test]
    fn shuffle_produces_permutation() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(42);
        let indices = shuffle_indices(10, &mut rng);
        assert_eq!(indices.len(), 10);

        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}

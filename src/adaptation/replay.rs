//! Replay buffer for catastrophic forgetting prevention.

use crate::circuit::CircuitId;
use crate::io::wits::WitsSnapshot;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single training sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input WITS data snapshot.
    pub input: WitsSnapshot,

    /// Expected circuit outputs (labels).
    pub expected_outputs: HashMap<CircuitId, Vec<f32>>,

    /// Formation type (for formation-specific replay).
    pub formation: Option<String>,

    /// Sample importance weight.
    pub weight: f32,
}

/// Stores canonical drilling scenarios to prevent catastrophic forgetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Representative samples from pre-training data.
    pub samples: Vec<TrainingSample>,

    /// Maximum buffer size.
    pub max_size: usize,

    /// Fraction of training batch that comes from replay (vs live data).
    pub replay_fraction: f32,
}

impl ReplayBuffer {
    pub fn new(max_size: usize, replay_fraction: f32) -> Self {
        Self {
            samples: Vec::new(),
            max_size,
            replay_fraction,
        }
    }

    /// Add a sample to the buffer. If full, replace the oldest sample.
    pub fn add(&mut self, sample: TrainingSample) {
        if self.samples.len() >= self.max_size {
            self.samples.remove(0);
        }
        self.samples.push(sample);
    }

    /// Sample a mixed batch: (1 - replay_fraction) live + replay_fraction replay.
    ///
    /// Returns indices into live_samples and self.samples respectively.
    pub fn mixed_batch_indices(
        &self,
        live_count: usize,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> (Vec<usize>, Vec<usize>) {
        if self.samples.is_empty() {
            // No replay samples available — use all live.
            let live_indices: Vec<usize> = (0..live_count.min(batch_size)).collect();
            return (live_indices, Vec::new());
        }

        let n_replay = (batch_size as f32 * self.replay_fraction) as usize;
        let n_live = batch_size.saturating_sub(n_replay);

        let live_indices: Vec<usize> = (0..live_count.min(n_live)).collect();

        let replay_indices: Vec<usize> = (0..n_replay)
            .map(|_| rng.gen_range(0..self.samples.len()))
            .collect();

        (live_indices, replay_indices)
    }

    /// Current number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_sample(formation: Option<&str>) -> TrainingSample {
        TrainingSample {
            input: WitsSnapshot::zeros(),
            expected_outputs: HashMap::new(),
            formation: formation.map(String::from),
            weight: 1.0,
        }
    }

    #[test]
    fn buffer_respects_max_size() {
        let mut buffer = ReplayBuffer::new(3, 0.1);

        for i in 0..10 {
            buffer.add(make_sample(Some(&format!("formation_{i}"))));
        }

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn mixed_batch_has_correct_proportions() {
        let mut buffer = ReplayBuffer::new(100, 0.2); // 20% replay
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            buffer.add(make_sample(None));
        }

        let (live, replay) = buffer.mixed_batch_indices(100, 10, &mut rng);

        // 10 batch, 20% replay = 2 replay, 8 live.
        assert_eq!(live.len(), 8);
        assert_eq!(replay.len(), 2);
    }

    #[test]
    fn empty_buffer_returns_all_live() {
        let buffer = ReplayBuffer::new(100, 0.2);
        let mut rng = StdRng::seed_from_u64(42);

        let (live, replay) = buffer.mixed_batch_indices(5, 10, &mut rng);

        assert_eq!(live.len(), 5); // Only 5 live available.
        assert!(replay.is_empty());
    }
}

//! Episodic memory store.
//!
//! Lives OUTSIDE the neural mesh. Provides long-term memory across hours,
//! days, and wells using brute-force KNN (HNSW can be plugged in later).

use crate::neuron::LiquidNeuron;
use serde::{Deserialize, Serialize};

/// What the causation circuits concluded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosisType {
    KickDetected,
    LossDetected,
    PackOff,
    StickSlip,
    FounderCondition,
    MSEInefficiency,
    FormationChange,
    EquipmentAnomaly(String),
    Unknown,
}

/// What happened after the event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    Resolved {
        action_taken: String,
        time_to_resolve_s: f32,
    },
    FalseAlarm,
    Escalated,
    StillOngoing,
}

/// Key drilling parameters at time of event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillingContext {
    pub depth_m: f32,
    pub wob_klbs: f32,
    pub rop_ft_hr: f32,
    pub rpm: f32,
    pub torque_ft_lbs: f32,
    pub spp_psi: f32,
    pub flow_in_gpm: f32,
    pub flow_out_gpm: f32,
    pub mse_psi: f32,
    pub formation: Option<String>,
}

impl Default for DrillingContext {
    fn default() -> Self {
        Self {
            depth_m: 0.0,
            wob_klbs: 0.0,
            rop_ft_hr: 0.0,
            rpm: 0.0,
            torque_ft_lbs: 0.0,
            spp_psi: 0.0,
            flow_in_gpm: 0.0,
            flow_out_gpm: 0.0,
            mse_psi: 0.0,
            formation: None,
        }
    }
}

/// A single recorded episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Monotonic tick when recorded.
    pub tick: u64,

    /// Compressed mesh state snapshot.
    /// Only stores neurons with |x| > 0.01 (sparse representation).
    pub state_vector: Vec<(usize, f32)>,

    /// What the causation circuits concluded.
    pub diagnosis: DiagnosisType,

    /// Confidence at time of storage.
    pub confidence: f32,

    /// Key drilling parameters at time of event.
    pub drilling_context: DrillingContext,

    /// What happened after (outcome, filled in retrospectively).
    pub outcome: Option<Outcome>,
}

/// Episodic memory store with brute-force KNN.
///
/// Uses cosine similarity for nearest-neighbor search.
/// Can be upgraded to HNSW (via `instant-distance` or similar) for
/// large episode counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// All stored episodes.
    pub episodes: Vec<Episode>,

    /// Maximum number of stored episodes.
    pub max_episodes: usize,

    /// Minimum confidence to trigger storage.
    pub store_threshold: f32,

    /// Total neuron count (for dense vector expansion).
    pub neuron_count: usize,
}

impl EpisodicMemory {
    pub fn new(neuron_count: usize, max_episodes: usize, store_threshold: f32) -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes,
            store_threshold,
            neuron_count,
        }
    }

    /// Store a new episode if confidence exceeds threshold.
    pub fn maybe_store(
        &mut self,
        neurons: &[LiquidNeuron],
        tick: u64,
        diagnosis: DiagnosisType,
        confidence: f32,
        drilling_context: DrillingContext,
    ) {
        if confidence < self.store_threshold {
            return;
        }

        // Compress mesh state (only non-trivial neurons).
        let state_vector: Vec<(usize, f32)> = neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| n.x.abs() > 0.01)
            .map(|(i, n)| (i, n.x))
            .collect();

        let episode = Episode {
            tick,
            state_vector,
            diagnosis,
            confidence,
            drilling_context,
            outcome: None,
        };

        self.episodes.push(episode);

        // Prune if over capacity.
        if self.episodes.len() > self.max_episodes {
            self.prune_oldest_low_confidence();
        }
    }

    /// Find k most similar past episodes to the current neuron state.
    ///
    /// Uses cosine similarity on the dense state vectors.
    pub fn recall_similar(&self, neurons: &[LiquidNeuron], k: usize) -> Vec<&Episode> {
        if self.episodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let current: Vec<f32> = neurons.iter().map(|n| n.x).collect();
        let current_norm = vec_norm(&current);

        if current_norm < 1e-9 {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| {
                let similarity = sparse_cosine_similarity(
                    &ep.state_vector,
                    &current,
                    current_norm,
                );
                (i, similarity)
            })
            .collect();

        // Sort by similarity descending.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(k)
            .filter_map(|&(idx, _)| self.episodes.get(idx))
            .collect()
    }

    /// Remove the oldest low-confidence episode.
    fn prune_oldest_low_confidence(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        // Find the episode with the lowest confidence.
        let min_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        self.episodes.swap_remove(min_idx);
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Check health (store is functional).
    pub fn is_healthy(&self) -> bool {
        true // The in-memory store is always healthy.
    }
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity between a sparse vector and a pre-normed dense vector.
fn sparse_cosine_similarity(
    sparse: &[(usize, f32)],
    dense: &[f32],
    dense_norm: f32,
) -> f32 {
    if sparse.is_empty() || dense_norm < 1e-9 {
        return 0.0;
    }

    let dot: f32 = sparse
        .iter()
        .filter(|&&(idx, _)| idx < dense.len())
        .map(|&(idx, val)| val * dense[idx])
        .sum();

    let sparse_norm: f32 = sparse.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();

    if sparse_norm < 1e-9 {
        return 0.0;
    }

    dot / (sparse_norm * dense_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::LiquidNeuron;

    fn make_neurons_with_values(values: &[f32]) -> Vec<LiquidNeuron> {
        values
            .iter()
            .map(|&v| {
                let mut n = LiquidNeuron::new();
                n.x = v;
                n
            })
            .collect()
    }

    fn default_context() -> DrillingContext {
        DrillingContext::default()
    }

    #[test]
    fn store_respects_threshold() {
        let mut store = EpisodicMemory::new(10, 100, 0.85);
        let neurons = make_neurons_with_values(&[0.5, 0.3, 0.1]);

        // Below threshold — should not store.
        store.maybe_store(&neurons, 0, DiagnosisType::Unknown, 0.5, default_context());
        assert_eq!(store.len(), 0);

        // Above threshold — should store.
        store.maybe_store(&neurons, 1, DiagnosisType::KickDetected, 0.9, default_context());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn recall_returns_most_similar() {
        let mut store = EpisodicMemory::new(10, 100, 0.0);

        // Store episode with state [1, 0, 0, ...]
        let neurons_a = make_neurons_with_values(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        store.maybe_store(&neurons_a, 0, DiagnosisType::KickDetected, 0.9, default_context());

        // Store episode with state [0, 0, 1, ...]
        let neurons_b = make_neurons_with_values(&[0.0, 0.0, 1.0, 0.0, 0.0]);
        store.maybe_store(&neurons_b, 1, DiagnosisType::LossDetected, 0.9, default_context());

        // Query with state similar to first episode.
        let query = make_neurons_with_values(&[0.9, 0.1, 0.0, 0.0, 0.0]);
        let results = store.recall_similar(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tick, 0); // Should match first episode.
    }

    #[test]
    fn recall_empty_store() {
        let store = EpisodicMemory::new(10, 100, 0.0);
        let neurons = make_neurons_with_values(&[0.5]);
        let results = store.recall_similar(&neurons, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn pruning_removes_lowest_confidence() {
        let mut store = EpisodicMemory::new(10, 3, 0.0);
        let neurons = make_neurons_with_values(&[0.5]);

        store.maybe_store(&neurons, 0, DiagnosisType::Unknown, 0.5, default_context());
        store.maybe_store(&neurons, 1, DiagnosisType::Unknown, 0.9, default_context());
        store.maybe_store(&neurons, 2, DiagnosisType::Unknown, 0.7, default_context());
        assert_eq!(store.len(), 3);

        // This should trigger pruning (max_episodes = 3, now adding 4th).
        store.maybe_store(&neurons, 3, DiagnosisType::Unknown, 0.8, default_context());
        assert_eq!(store.len(), 3);

        // The episode with confidence 0.5 should have been removed.
        assert!(store.episodes.iter().all(|e| e.confidence >= 0.7));
    }

    #[test]
    fn state_vector_compression() {
        let mut store = EpisodicMemory::new(10, 100, 0.0);
        let neurons = make_neurons_with_values(&[0.5, 0.0, 0.005, 0.8, 0.0]);

        store.maybe_store(&neurons, 0, DiagnosisType::Unknown, 0.9, default_context());

        let ep = &store.episodes[0];
        // Only neurons with |x| > 0.01 should be stored.
        assert_eq!(ep.state_vector.len(), 2); // 0.5 and 0.8
        assert_eq!(ep.state_vector[0], (0, 0.5));
        assert_eq!(ep.state_vector[1], (3, 0.8));
    }

    #[test]
    fn knn_returns_correct_count() {
        let mut store = EpisodicMemory::new(10, 100, 0.0);
        let neurons = make_neurons_with_values(&[0.5, 0.3]);

        for i in 0..10 {
            store.maybe_store(
                &neurons,
                i,
                DiagnosisType::Unknown,
                0.9,
                default_context(),
            );
        }

        let results = store.recall_similar(&neurons, 5);
        assert_eq!(results.len(), 5);
    }
}

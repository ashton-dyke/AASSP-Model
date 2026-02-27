//! Mesh configuration and hyperparameters.

use serde::{Deserialize, Serialize};

/// Execution mode of the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationMode {
    /// All systems nominal.
    Full,
    /// Reduced accuracy: some circuits frozen or bypassed.
    Degraded,
    /// Minimum viable: only safety-critical detection.
    Minimal,
}

impl Default for DegradationMode {
    fn default() -> Self {
        DegradationMode::Full
    }
}

/// Central configuration for the neural mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Total number of neurons in the mesh.
    pub total_neurons: usize,

    /// Number of neurons reserved for circuits (rest are adapter neurons).
    pub reserved_neurons: usize,

    /// Integration timestep (seconds).
    pub dt: f32,

    /// Number of mesh steps per WITS data sample.
    pub mesh_steps_per_wits_sample: usize,

    /// Overlap pruning threshold — mask values below this are pruned.
    pub overlap_prune_threshold: f32,

    /// Safety override confidence threshold.
    pub safety_override_threshold: f32,

    /// Current degradation mode.
    pub mode: DegradationMode,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            total_neurons: 4_736,
            reserved_neurons: 3_456,
            dt: 0.005,
            mesh_steps_per_wits_sample: 100,
            overlap_prune_threshold: 0.1,
            safety_override_threshold: 0.75,
            mode: DegradationMode::Full,
        }
    }
}

//! Mesh configuration and hyperparameters.

use serde::{Deserialize, Serialize};

/// Execution mode of the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// No outputs. Silently learns baselines.
    Shadow,
    /// Generates advisories flagged as "unvalidated".
    Advisory,
    /// Full production.
    Production,
}

/// Degradation mode when components fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationMode {
    /// Full mesh operational.
    Full,
    /// Mesh failed, using physics-only calculations.
    PhysicsOnly,
    /// Mesh partially failed, running detection circuits only.
    DetectionOnly,
    /// Memory/episodic store unavailable.
    NoMemory,
}

/// Global mesh configuration with all hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    // --- Neuron parameters ---
    pub total_neurons: usize,
    pub reserved_neurons: usize,

    // --- Timestep ---
    pub dt: f32,
    pub wits_sample_rate_hz: f32,
    pub mesh_steps_per_wits_sample: u32,

    // --- Overlap ---
    pub initial_overlap_fraction: f32,
    pub mask_prune_threshold: f32,
    pub sparsity_lambda: f32,
    pub overlap_stability_lambda: f32,

    // --- Safety ---
    pub safety_override_threshold: f32,
    pub max_circuits_per_neuron: u8,

    // --- Memory ---
    pub short_term_tau: f32,
    pub short_term_decay: f32,
    pub medium_term_tau: f32,
    pub medium_term_decay: f32,
    pub long_term_tau: f32,
    pub long_term_decay: f32,

    // --- Episodic store ---
    pub episodic_store_threshold: f32,
    pub episodic_max_episodes: usize,
    pub episodic_knn_k: usize,

    // --- Replay buffer ---
    pub replay_buffer_size: usize,
    pub replay_fraction: f32,

    // --- Adapter ---
    pub formation_adapter_size: usize,
    pub well_adapter_size: usize,

    // --- Training ---
    pub stage1_steps: u32,
    pub stage1_lr: f32,
    pub stage2_rounds: u32,
    pub stage2_steps_per_round: u32,
    pub stage2_lr: f32,
    pub stage3_steps: u32,
    pub stage3_lr: f32,
    pub online_lr: f32,

    // --- Baseline learning ---
    pub baseline_min_samples: u64,
    pub baseline_period_s: f32,

    // --- Execution ---
    pub initial_mode: ExecutionMode,
    pub enable_episodic_store: bool,
    pub enable_online_learning: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            total_neurons: 4_736,
            reserved_neurons: 3_456,

            dt: 0.001,
            wits_sample_rate_hz: 1.0,
            mesh_steps_per_wits_sample: 100,

            initial_overlap_fraction: 0.10,
            mask_prune_threshold: 0.3,
            sparsity_lambda: 0.01,
            overlap_stability_lambda: 0.1,

            safety_override_threshold: 0.75,
            max_circuits_per_neuron: 2,

            short_term_tau: 0.2,
            short_term_decay: 0.7,
            medium_term_tau: 2.0,
            medium_term_decay: 0.07,
            long_term_tau: 30.0,
            long_term_decay: 0.004,

            episodic_store_threshold: 0.85,
            episodic_max_episodes: 10_000,
            episodic_knn_k: 5,

            replay_buffer_size: 500,
            replay_fraction: 0.1,

            formation_adapter_size: 64,
            well_adapter_size: 64,

            stage1_steps: 5_000,
            stage1_lr: 1e-3,
            stage2_rounds: 5,
            stage2_steps_per_round: 500,
            stage2_lr: 1e-3,
            stage3_steps: 2_000,
            stage3_lr: 1e-4,
            online_lr: 1e-4,

            baseline_min_samples: 100,
            baseline_period_s: 604_800.0,

            initial_mode: ExecutionMode::Shadow,
            enable_episodic_store: true,
            enable_online_learning: true,
        }
    }
}

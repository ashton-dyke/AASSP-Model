//! Production mesh: three-tier architecture (universal, adapters, adaptive).

use crate::adaptation::adapter::AdapterLayer;
use crate::adaptation::replay::ReplayBuffer;
use crate::circuit::Circuit;
use crate::config::{DegradationMode, ExecutionMode, MeshConfig};
use crate::io::encoding::Baselines;
use crate::io::output::MeshOutput;
use crate::io::wits::WitsSnapshot;
use crate::memory::episodic::EpisodicMemory;
use crate::mesh::OverlappingMesh;

use serde::{Deserialize, Serialize};

/// The full production mesh with three-tier architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionMesh {
    // ═══ TIER 1: UNIVERSAL (Frozen after pre-training) ═══
    pub universal_mesh: OverlappingMesh,

    // ═══ TIER 2: ADAPTERS (Plastic, thin layers) ═══
    pub formation_adapter: AdapterLayer,
    pub well_adapter: AdapterLayer,

    // ═══ TIER 3: ADAPTIVE (Plastic, online learning) ═══
    pub adaptive_circuits: Vec<Circuit>,

    // ═══ SUPPORT STRUCTURES ═══
    pub replay_buffer: ReplayBuffer,
    pub episodic_store: EpisodicMemory,
    pub baselines: Baselines,

    // ═══ EXECUTION STATE ═══
    pub execution_mode: ExecutionMode,
}

impl ProductionMesh {
    /// Create a new production mesh with default configuration.
    pub fn new(config: MeshConfig) -> Self {
        let neuron_count = config.total_neurons;
        let replay_size = config.replay_buffer_size;
        let replay_frac = config.replay_fraction;
        let episodic_max = config.episodic_max_episodes;
        let episodic_thresh = config.episodic_store_threshold;
        let mode = config.initial_mode;

        Self {
            universal_mesh: OverlappingMesh::new(config),
            formation_adapter: AdapterLayer::new(64, Vec::new(), Vec::new()),
            well_adapter: AdapterLayer::new(64, Vec::new(), Vec::new()),
            adaptive_circuits: Vec::new(),
            replay_buffer: ReplayBuffer::new(replay_size, replay_frac),
            episodic_store: EpisodicMemory::new(neuron_count, episodic_max, episodic_thresh),
            baselines: Baselines::new(),
            execution_mode: mode,
        }
    }

    /// Process a single WITS sample through the full pipeline.
    pub fn process(&mut self, wits: &WitsSnapshot) -> MeshOutput {
        // Update baselines.
        self.baselines.update(wits);

        // Encode input into detection circuit neurons.
        crate::io::encoding::encode_input(&mut self.universal_mesh, wits, &self.baselines);

        // Run mesh steps.
        let steps = self.universal_mesh.config.mesh_steps_per_wits_sample;
        for _ in 0..steps {
            self.universal_mesh.step();
        }

        // Produce output.
        let mode = self.health_check();
        MeshOutput::empty(mode)
    }

    /// Check system health and determine degradation mode.
    pub fn health_check(&self) -> DegradationMode {
        if !self.universal_mesh.is_healthy() {
            return DegradationMode::PhysicsOnly;
        }

        if !self.episodic_store.is_healthy() {
            return DegradationMode::NoMemory;
        }

        if !self.all_circuit_types_active() {
            return DegradationMode::DetectionOnly;
        }

        DegradationMode::Full
    }

    /// Check if all circuit types have at least one representative.
    fn all_circuit_types_active(&self) -> bool {
        use crate::circuit::CircuitType;

        let types = [
            CircuitType::Detection,
            CircuitType::Causation,
            CircuitType::Prediction,
            CircuitType::Memory,
        ];

        types.iter().all(|ct| {
            self.universal_mesh
                .circuits
                .iter()
                .any(|c| c.circuit_type == *ct)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_mesh_creation() {
        let config = MeshConfig::default();
        let mesh = ProductionMesh::new(config);

        assert_eq!(mesh.universal_mesh.neuron_count(), 4_736);
        assert_eq!(mesh.formation_adapter.neurons.len(), 64);
        assert_eq!(mesh.well_adapter.neurons.len(), 64);
        assert_eq!(mesh.execution_mode, ExecutionMode::Shadow);
    }

    #[test]
    fn health_check_physics_only_when_unhealthy() {
        let config = MeshConfig::default();
        let mut mesh = ProductionMesh::new(config);

        // Corrupt a neuron.
        mesh.universal_mesh.neurons[0].x = f32::NAN;
        assert_eq!(mesh.health_check(), DegradationMode::PhysicsOnly);
    }

    #[test]
    fn process_wits_does_not_panic() {
        let config = MeshConfig {
            total_neurons: 20,
            mesh_steps_per_wits_sample: 10,
            ..MeshConfig::default()
        };
        let mut mesh = ProductionMesh::new(config);
        let wits = WitsSnapshot::zeros();

        let output = mesh.process(&wits);
        // With no circuits, is_healthy() returns false → PhysicsOnly.
        assert_eq!(output.mode, DegradationMode::PhysicsOnly);
    }
}

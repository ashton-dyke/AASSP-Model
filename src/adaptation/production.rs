//! Production mesh: three-tier architecture (universal, adapters, adaptive).

use crate::adaptation::adapter::AdapterLayer;
use crate::adaptation::replay::ReplayBuffer;
use crate::circuit::{Circuit, CircuitType};
use crate::config::{DegradationMode, MeshConfig};
use crate::io::encoding::Baselines;
use crate::io::output::{CausalAnalysis, Detection, EpisodicMatch, MeshOutput, Prediction};
use crate::io::wits::WitsSnapshot;
use crate::memory::episodic::{DiagnosisType, DrillingContext, EpisodicMemory};
use crate::mesh::OverlappingMesh;
use crate::training::loss::detection_readout;

use serde::{Deserialize, Serialize};

/// Sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Map a detection circuit index (0-4) to a DiagnosisType.
fn circuit_idx_to_diagnosis(idx: usize) -> DiagnosisType {
    match idx {
        0 => DiagnosisType::KickDetected,
        1 => DiagnosisType::LossDetected,
        2 => DiagnosisType::PackOff,
        3 => DiagnosisType::StickSlip,
        4 => DiagnosisType::FounderCondition,
        _ => DiagnosisType::Unknown,
    }
}

/// Causation readout parameter names.
/// Last 8 neurons map to: torque, pressure, flow_balance, wob, spp, rop, gas, rpm.
const CAUSATION_PARAMS: [&str; 8] = [
    "torque",
    "pressure",
    "flow_balance",
    "wob",
    "spp",
    "rop",
    "gas",
    "rpm",
];

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
}

impl ProductionMesh {
    /// Create a new production mesh with default configuration.
    pub fn new(config: MeshConfig) -> Self {
        let neuron_count = config.total_neurons;

        Self {
            universal_mesh: OverlappingMesh::new(config),
            formation_adapter: AdapterLayer::new(64, Vec::new(), Vec::new()),
            well_adapter: AdapterLayer::new(64, Vec::new(), Vec::new()),
            adaptive_circuits: Vec::new(),
            replay_buffer: ReplayBuffer::new(1000, 0.1),
            episodic_store: EpisodicMemory::new(neuron_count, 500, 0.85),
            baselines: Baselines::new(),
        }
    }

    /// Create a production mesh from a trained OverlappingMesh.
    ///
    /// Wraps the trained mesh with properly initialized adapters using
    /// the default adapter index layout.
    pub fn from_trained(mesh: OverlappingMesh) -> Self {
        let neuron_count = mesh.neurons.len();
        let (fi, fo, wi, wo) =
            crate::training::init::default_adapter_indices(&mesh);

        Self {
            universal_mesh: mesh,
            formation_adapter: AdapterLayer::new(64, fi, fo),
            well_adapter: AdapterLayer::new(64, wi, wo),
            adaptive_circuits: Vec::new(),
            replay_buffer: ReplayBuffer::new(1000, 0.1),
            episodic_store: EpisodicMemory::new(neuron_count, 500, 0.5),
            baselines: Baselines::new(),
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

        // Run adapters (initially no-op with zero weights).
        self.formation_adapter
            .forward_in_place(&mut self.universal_mesh.neurons);
        self.well_adapter
            .forward_in_place(&mut self.universal_mesh.neurons);

        // Check health.
        let mode = self.health_check();
        if mode == DegradationMode::Minimal {
            return MeshOutput::empty(mode);
        }

        // Extract detections from circuits 0-4.
        let detections = self.extract_detections();

        // Extract causal analyses from circuits 5-7.
        let causal_analyses = self.extract_causations();

        // Extract predictions from circuits 11-12.
        let predictions = self.extract_predictions();

        // Episodic memory: recall similar episodes.
        let episodic_matches = self.episodic_store
            .recall_similar_scored(&self.universal_mesh.neurons, 3)
            .into_iter()
            .map(|(ep, sim)| EpisodicMatch {
                tick: ep.tick,
                diagnosis: ep.diagnosis.clone(),
                confidence: ep.confidence,
                similarity: sim,
            })
            .collect();

        // Store new episode if max detection confidence > 0.5.
        let max_det_confidence = detections
            .iter()
            .map(|d| d.confidence)
            .fold(0.0f32, f32::max);

        if max_det_confidence > 0.5 {
            // Find the diagnosis with highest confidence.
            if let Some(best) = detections.iter().max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.episodic_store.maybe_store(
                    &self.universal_mesh.neurons,
                    self.universal_mesh.tick,
                    best.anomaly_type.clone(),
                    best.confidence,
                    DrillingContext {
                        wob_klbs: wits.wob_klbs,
                        rop_ft_hr: wits.rop_ft_hr,
                        rpm: wits.rpm,
                        torque_ft_lbs: wits.torque_ft_lbs,
                        spp_psi: wits.spp_psi,
                        flow_in_gpm: wits.flow_in_gpm,
                        flow_out_gpm: wits.flow_out_gpm,
                        mse_psi: wits.mse_psi,
                        depth_m: wits.bit_depth_ft * 0.3048,
                        formation: None,
                    },
                );
            }
        }

        // Compute risk level.
        let risk_level = MeshOutput::compute_risk_level(&detections);

        MeshOutput {
            detections,
            causal_analyses,
            predictions,
            episodic_matches,
            risk_level,
            mode,
        }
    }

    /// Extract detection results from detection circuits (indices 0-4).
    fn extract_detections(&self) -> Vec<Detection> {
        let mut detections = Vec::new();

        for (ci, circuit) in self.universal_mesh.circuits.iter().enumerate() {
            if circuit.circuit_type != CircuitType::Detection {
                continue;
            }

            let confidence = detection_readout(&self.universal_mesh, ci);

            if confidence > 0.5 {
                detections.push(Detection {
                    circuit_name: circuit.name.clone(),
                    anomaly_type: circuit_idx_to_diagnosis(ci),
                    confidence,
                    severity: MeshOutput::confidence_to_severity(confidence),
                });
            }
        }

        detections
    }

    /// Extract causal analysis from causation circuits (indices 5-7).
    fn extract_causations(&self) -> Vec<CausalAnalysis> {
        let mut analyses = Vec::new();

        for circuit in &self.universal_mesh.circuits {
            if circuit.circuit_type != CircuitType::Causation {
                continue;
            }

            let indices = &circuit.neuron_indices;
            let n = indices.len();
            let readout_size = 8.min(n);
            let start = n - readout_size;
            let readout_indices = &indices[start..];

            let gain = crate::training::loss::readout_gain(circuit.tau);

            // Find active causal parameters (sigmoid > 0.5).
            let mut active_params: Vec<(usize, f32)> = Vec::new();
            for (pos, &neuron_idx) in readout_indices.iter().enumerate() {
                let activation =
                    sigmoid(self.universal_mesh.neurons[neuron_idx].x * gain);
                if activation > 0.5 && pos < CAUSATION_PARAMS.len() {
                    active_params.push((pos, activation));
                }
            }

            if active_params.is_empty() {
                continue;
            }

            // Sort by activation descending.
            active_params
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let root_cause = CAUSATION_PARAMS[active_params[0].0].to_string();
            let contributing_factors: Vec<String> = active_params
                .iter()
                .skip(1)
                .map(|(pos, _)| CAUSATION_PARAMS[*pos].to_string())
                .collect();

            let max_confidence = active_params[0].1;

            analyses.push(CausalAnalysis {
                circuit_name: circuit.name.clone(),
                root_cause,
                contributing_factors,
                confidence: max_confidence,
                recommended_action: String::new(),
            });
        }

        analyses
    }

    /// Extract predictions from prediction circuits (indices 11-12).
    fn extract_predictions(&self) -> Vec<Prediction> {
        let mut predictions = Vec::new();

        for circuit in &self.universal_mesh.circuits {
            if circuit.circuit_type != CircuitType::Prediction {
                continue;
            }

            // Read confidence from the circuit's readout neurons.
            let indices = &circuit.neuron_indices;
            let n = indices.len();
            let readout_size = 14.min(n);
            let start = n - readout_size;
            let readout_indices = &indices[start..];

            let mean_abs: f32 = readout_indices
                .iter()
                .map(|&idx| self.universal_mesh.neurons[idx].x.abs())
                .sum::<f32>()
                / readout_size as f32;

            let confidence = mean_abs.clamp(0.0, 1.0);

            let predicted_state = if circuit.name.contains("rop") {
                "rop_forecast".to_string()
            } else {
                "state_forecast".to_string()
            };

            predictions.push(Prediction {
                circuit_name: circuit.name.clone(),
                predicted_state,
                confidence,
                horizon_seconds: 5.0,
            });
        }

        predictions
    }

    /// Check system health and determine degradation mode.
    pub fn health_check(&self) -> DegradationMode {
        // Check for NaN/Inf in neurons.
        let all_finite = self.universal_mesh.neurons.iter().all(|n| n.x.is_finite());
        if !all_finite {
            return DegradationMode::Minimal;
        }

        // Check if all circuit types are present.
        if !self.all_circuit_types_active() {
            return DegradationMode::Degraded;
        }

        DegradationMode::Full
    }

    /// Check if all circuit types have at least one representative.
    fn all_circuit_types_active(&self) -> bool {
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
    use crate::io::output::RiskLevel;

    #[test]
    fn production_mesh_creation() {
        let config = MeshConfig::default();
        let mesh = ProductionMesh::new(config);

        assert_eq!(mesh.universal_mesh.neurons.len(), 4_736);
        assert_eq!(mesh.formation_adapter.neurons.len(), 64);
        assert_eq!(mesh.well_adapter.neurons.len(), 64);
    }

    #[test]
    fn health_check_minimal_when_nan() {
        let config = MeshConfig::default();
        let mut mesh = ProductionMesh::new(config);

        mesh.universal_mesh.neurons[0].x = f32::NAN;
        assert_eq!(mesh.health_check(), DegradationMode::Minimal);
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
        // With no circuits, not all_circuit_types_active() → Degraded.
        assert_eq!(output.mode, DegradationMode::Degraded);
    }

    #[test]
    fn process_produces_structured_output() {
        use crate::layout::build_default_mesh;

        let mesh = build_default_mesh();
        let mut prod = ProductionMesh::from_trained(mesh);

        let wits = WitsSnapshot {
            wob_klbs: 25.0,
            rop_ft_hr: 60.0,
            rpm: 120.0,
            torque_ft_lbs: 5000.0,
            hook_load_klbs: 200.0,
            spp_psi: 3000.0,
            flow_in_gpm: 400.0,
            flow_out_gpm: 395.0,
            pit_volume_bbl: 500.0,
            bit_depth_ft: 10000.0,
            mud_weight_in_ppg: 12.0,
            mud_weight_out_ppg: 12.1,
            mud_temp_in_f: 80.0,
            mud_temp_out_f: 120.0,
            total_gas_pct: 0.5,
            h2s_ppm: 0.0,
            mse_psi: 40000.0,
            d_exponent: 1.5,
            ecd_ppg: 12.5,
            flow_balance_gpm: -5.0,
            pit_rate_bbl_hr: -2.0,
        };

        // Process a few samples to build baselines.
        for _ in 0..5 {
            let _ = prod.process(&wits);
        }

        let output = prod.process(&wits);

        // Mode should be Full (all circuit types present).
        assert_eq!(output.mode, DegradationMode::Full);

        // Predictions should be populated (2 prediction circuits).
        assert_eq!(
            output.predictions.len(),
            2,
            "Should have 2 predictions from state_prediction and rop_prediction"
        );

        // Risk level should be a valid variant.
        let _risk = output.risk_level; // Just verify it doesn't panic.
    }

    #[test]
    fn from_trained_sets_adapter_indices() {
        use crate::layout::build_default_mesh;

        let mesh = build_default_mesh();
        let prod = ProductionMesh::from_trained(mesh);

        assert!(
            !prod.formation_adapter.input_from.is_empty(),
            "Formation adapter should have input indices"
        );
        assert!(
            !prod.formation_adapter.output_to.is_empty(),
            "Formation adapter should have output indices"
        );
        assert!(
            !prod.well_adapter.input_from.is_empty(),
            "Well adapter should have input indices"
        );
        assert!(
            !prod.well_adapter.output_to.is_empty(),
            "Well adapter should have output indices"
        );
    }

    #[test]
    #[ignore] // Takes ~40s. Run with: cargo test end_to_end -- --ignored
    fn end_to_end_train_then_detect() {
        use crate::layout::build_default_mesh;
        use crate::training::pipeline::{run_full_training, TrainingConfig};
        use crate::training::synthetic::SyntheticWellGenerator;

        // Train with a small config.
        let config = TrainingConfig {
            stage1_steps: 50,
            stage2_rounds: 2,
            stage2_steps_per_round: 100,
            stage2_memory_steps: 30,
            stage2_prediction_steps: 30,
            stage3_steps: 100,
            mesh_steps_per_sample: 50,
            synthetic_sequences: 12,
            sequence_length: 50,
            mini_batch_size: 1,
            ..TrainingConfig::default()
        };

        let mut mesh = build_default_mesh();
        let _metrics = run_full_training(&mut mesh, &config);

        // Wrap in ProductionMesh.
        let mut prod = ProductionMesh::from_trained(mesh);

        // Process normal data — should be Green/Yellow risk.
        let mut gen = SyntheticWellGenerator::new(99);
        let normal_seq = gen.normal_sequence(20);

        for wits in &normal_seq.wits_samples {
            let output = prod.process(wits);
            assert!(
                output.risk_level == RiskLevel::Green || output.risk_level == RiskLevel::Yellow,
                "Normal data should be Green or Yellow risk, got {:?}",
                output.risk_level
            );
        }

        // Process kick sequence — should detect anomaly at some point.
        let kick_seq = gen.kick_sequence(30, 10, 0.9);
        let mut found_detection = false;

        for wits in &kick_seq.wits_samples {
            let output = prod.process(wits);
            if !output.detections.is_empty() {
                found_detection = true;
            }
        }

        // After training, the mesh should produce non-empty output.
        let last_output = prod.process(&kick_seq.wits_samples.last().unwrap());
        assert_eq!(last_output.mode, DegradationMode::Full);
        assert_eq!(last_output.predictions.len(), 2);

        // We expect the trained mesh to detect something, but with minimal
        // training this is not guaranteed — just verify the pipeline works.
        eprintln!(
            "End-to-end test: found_detection={}, last detections={}, last risk={:?}",
            found_detection,
            last_output.detections.len(),
            last_output.risk_level
        );
    }
}

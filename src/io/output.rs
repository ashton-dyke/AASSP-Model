//! Mesh output types: detections, causal analyses, predictions, risk level.

use crate::config::DegradationMode;
use crate::memory::episodic::DiagnosisType;
use serde::{Deserialize, Serialize};

/// Severity of a detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Overall risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// >85% efficiency, no anomalies.
    Green,
    /// Minor anomaly, log it.
    Yellow,
    /// Sustained issue, needs attention.
    Amber,
    /// Hard limit breach, immediate alert.
    Red,
}

/// A single detection result from one circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub circuit_name: String,
    pub anomaly_type: DiagnosisType,
    pub confidence: f32,
    pub severity: Severity,
}

/// A causal analysis result from one circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAnalysis {
    pub circuit_name: String,
    pub root_cause: String,
    pub contributing_factors: Vec<String>,
    pub confidence: f32,
    pub recommended_action: String,
}

/// A prediction from a prediction circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub circuit_name: String,
    pub predicted_state: String,
    pub confidence: f32,
    pub horizon_seconds: f32,
}

/// An episodic memory match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMatch {
    pub tick: u64,
    pub diagnosis: DiagnosisType,
    pub confidence: f32,
    pub similarity: f32,
}

/// The mesh's complete output after processing one WITS snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshOutput {
    pub detections: Vec<Detection>,
    pub causal_analyses: Vec<CausalAnalysis>,
    pub predictions: Vec<Prediction>,
    pub episodic_matches: Vec<EpisodicMatch>,
    pub risk_level: RiskLevel,
    pub mode: DegradationMode,
}

impl MeshOutput {
    /// Create an empty output (used when mesh has nothing to report).
    pub fn empty(mode: DegradationMode) -> Self {
        Self {
            detections: Vec::new(),
            causal_analyses: Vec::new(),
            predictions: Vec::new(),
            episodic_matches: Vec::new(),
            risk_level: RiskLevel::Green,
            mode,
        }
    }

    /// Determine risk level from detection results.
    pub fn compute_risk_level(detections: &[Detection]) -> RiskLevel {
        if detections.is_empty() {
            return RiskLevel::Green;
        }

        let max_severity = detections
            .iter()
            .map(|d| &d.severity)
            .max_by_key(|s| match s {
                Severity::Low => 0,
                Severity::Medium => 1,
                Severity::High => 2,
                Severity::Critical => 3,
            });

        match max_severity {
            Some(Severity::Critical) => RiskLevel::Red,
            Some(Severity::High) => RiskLevel::Amber,
            Some(Severity::Medium) => RiskLevel::Yellow,
            _ => RiskLevel::Green,
        }
    }

    /// Map confidence to severity.
    pub fn confidence_to_severity(confidence: f32) -> Severity {
        if confidence > 0.9 {
            Severity::Critical
        } else if confidence > 0.7 {
            Severity::High
        } else if confidence > 0.5 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }
}

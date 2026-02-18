//! SAIREN-OS LTC Neural Mesh
//!
//! A synchronous overlapping neural mesh built on Liquid Time-Constant (LTC)
//! neurons with Closed-form Continuous-depth (CfC) evaluation. Multiple neural
//! circuits share neurons through overlap zones and communicate through shared
//! state. Each neuron has input-dependent gating that modulates its effective
//! time constant, enabling adaptive temporal dynamics without async scheduling.

pub mod adaptation;
pub mod circuit;
pub mod config;
pub mod io;
pub mod layout;
pub mod memory;
pub mod mesh;
pub mod neuron;
pub mod overlap;
pub mod physics;
pub mod topology;
pub mod training;

// Re-export primary types for convenience.
pub use adaptation::{AdapterLayer, ProductionMesh, ReplayBuffer};
pub use circuit::{Circuit, CircuitCriticality, CircuitId, CircuitType};
pub use config::{DegradationMode, MeshConfig};
pub use io::{BaselineStats, CausalAnalysis, Detection, MeshOutput, RiskLevel, Severity, WitsSnapshot};
pub use memory::{DiagnosisType, EpisodicMemory, HierarchicalMemory};
pub use mesh::OverlappingMesh;
pub use neuron::{LtcNeuron, CfcIntermediate, cfc_forward, cfc_forward_inference, sigmoid};
pub use overlap::OverlapZone;
pub use topology::{TopologyConstraint, TopologyError};

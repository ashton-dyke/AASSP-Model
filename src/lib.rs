//! SAIREN-OS Asynchronous Overlapping Neural Mesh
//!
//! An overlapping liquid neural network mesh implementing Asymmetric
//! Asynchronous Shared-State Parallelism (AASSP). Multiple neural circuits
//! share neurons through overlap zones, process at different timescales,
//! and communicate through shared state rather than message passing.

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

// Re-export primary types for convenience.
pub use adaptation::{AdapterLayer, ProductionMesh, ReplayBuffer};
pub use circuit::{Circuit, CircuitCriticality, CircuitId, CircuitType};
pub use config::{DegradationMode, ExecutionMode, MeshConfig};
pub use io::{BaselineStats, CausalAnalysis, Detection, MeshOutput, RiskLevel, Severity, WitsSnapshot};
pub use memory::{DiagnosisType, EpisodicMemory, HierarchicalMemory};
pub use mesh::OverlappingMesh;
pub use neuron::LiquidNeuron;
pub use overlap::{OverlapGate, OverlapZone};
pub use topology::{TopologyConstraint, TopologyError};

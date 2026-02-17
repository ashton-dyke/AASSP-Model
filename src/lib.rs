//! SAIREN-OS Asynchronous Overlapping Neural Mesh
//!
//! An overlapping liquid neural network mesh implementing Asymmetric
//! Asynchronous Shared-State Parallelism (AASSP). Multiple neural circuits
//! share neurons through overlap zones, process at different timescales,
//! and communicate through shared state rather than message passing.

pub mod circuit;
pub mod config;
pub mod mesh;
pub mod neuron;
pub mod overlap;
pub mod topology;

// Re-export primary types for convenience.
pub use circuit::{Circuit, CircuitCriticality, CircuitId, CircuitType};
pub use config::{DegradationMode, ExecutionMode, MeshConfig};
pub use mesh::OverlappingMesh;
pub use neuron::LiquidNeuron;
pub use overlap::{OverlapGate, OverlapZone};
pub use topology::{TopologyConstraint, TopologyError};

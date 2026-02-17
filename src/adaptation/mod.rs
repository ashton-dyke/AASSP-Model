//! Modular architecture and adaptation: adapters, replay, production mesh.

pub mod adapter;
pub mod production;
pub mod replay;

pub use adapter::{AdapterLayer, AdapterNeuron};
pub use production::ProductionMesh;
pub use replay::{ReplayBuffer, TrainingSample};

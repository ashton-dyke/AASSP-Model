//! Training pipeline for the AASSP neural mesh.
//!
//! Implements manual BPTT (backpropagation through time) for liquid neuron
//! dynamics, three-stage offline training, and online adapter learning.

pub mod backward;
pub mod data;
pub mod init;
pub mod loss;
pub mod online;
pub mod optimizer;
pub mod pipeline;
pub mod state;
pub mod synthetic;

pub use optimizer::AdamOptimizer;
pub use pipeline::{evaluate_detection_accuracy, run_full_training, TrainingConfig, TrainingMetrics};
pub use state::{ForwardRecord, GradientAccumulator};
pub use synthetic::SyntheticWellGenerator;

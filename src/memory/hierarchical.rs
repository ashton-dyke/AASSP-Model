//! Hierarchical memory: short-term, medium-term, and long-term memory circuits.
//!
//! Each memory circuit has exponential decay and a learned write gate
//! that controls whether new information is stored.

use crate::circuit::{Circuit, CircuitCriticality, CircuitType};
use crate::neuron::LtcNeuron;
use serde::{Deserialize, Serialize};

/// Learned write gate: determines whether causation output should be
/// written into the memory circuit.
///
/// Simple linear layer → sigmoid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteGate {
    /// Weights (length = number of causation output features).
    pub weights: Vec<f32>,
    /// Bias.
    pub bias: f32,
}

impl WriteGate {
    /// Create a write gate for the given input dimension.
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim],
            bias: 0.0,
        }
    }

    /// Forward pass: dot(weights, input) + bias → sigmoid.
    pub fn forward(&self, causation_output: &[f32]) -> f32 {
        let sum: f32 = self
            .weights
            .iter()
            .zip(causation_output)
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias;

        // Sigmoid
        1.0 / (1.0 + (-sum).exp())
    }
}

/// A memory circuit with exponential decay and write-gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCircuit {
    /// The underlying circuit.
    pub circuit: Circuit,

    /// Exponential decay rate (lambda).
    /// Applied every timestep: x *= exp(-lambda * dt).
    pub decay_rate: f32,

    /// Learned write gate.
    pub write_gate: WriteGate,
}

impl MemoryCircuit {
    /// Create a new memory circuit.
    pub fn new(circuit: Circuit, decay_rate: f32, causation_dim: usize) -> Self {
        Self {
            circuit,
            decay_rate,
            write_gate: WriteGate::new(causation_dim),
        }
    }

    /// Update the memory circuit: decay existing activations, then
    /// optionally write new information gated by causation output.
    pub fn update(
        &mut self,
        neurons: &mut [LtcNeuron],
        causation_output: &[f32],
        dt: f32,
    ) {
        // Step 1: Exponential decay of all memory neurons.
        let decay_factor = (-self.decay_rate * dt).exp();
        for &neuron_idx in &self.circuit.neuron_indices {
            neurons[neuron_idx].x *= decay_factor;
        }

        // Step 2: Compute write gate.
        let gate_value = self.write_gate.forward(causation_output);

        // Step 3: Gated write — compute CfC update and scale by gate.
        if gate_value > 0.05 {
            let updates = self.circuit.compute_cfc_updates(neurons, dt);
            for (neuron_idx, new_x) in updates {
                let delta = new_x - neurons[neuron_idx].x;
                neurons[neuron_idx].x += gate_value * delta;
            }
        }
    }
}

/// Three-level hierarchical memory system (inside the mesh).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalMemory {
    /// Recent events, decays in ~1 second.
    pub short_term: MemoryCircuit,
    /// Current operation context, decays in ~10 seconds.
    pub medium_term: MemoryCircuit,
    /// Well behaviour patterns, decays in ~5 minutes.
    pub long_term: MemoryCircuit,
}

impl HierarchicalMemory {
    /// Create the three-level memory system with default parameters.
    pub fn new(
        short_range: std::ops::Range<usize>,
        medium_range: std::ops::Range<usize>,
        long_range: std::ops::Range<usize>,
        causation_dim: usize,
    ) -> Self {
        let short_circuit = Circuit::new(
            100,
            "short_term_memory",
            short_range.collect(),
            0.2,
            CircuitType::Memory,
            CircuitCriticality::Performance,
        );

        let medium_circuit = Circuit::new(
            101,
            "medium_term_memory",
            medium_range.collect(),
            2.0,
            CircuitType::Memory,
            CircuitCriticality::Performance,
        );

        let long_circuit = Circuit::new(
            102,
            "long_term_memory",
            long_range.collect(),
            30.0,
            CircuitType::Memory,
            CircuitCriticality::Performance,
        );

        Self {
            short_term: MemoryCircuit::new(short_circuit, 0.7, causation_dim),
            medium_term: MemoryCircuit::new(medium_circuit, 0.07, causation_dim),
            long_term: MemoryCircuit::new(long_circuit, 0.004, causation_dim),
        }
    }

    /// Update all three memory levels.
    pub fn update(
        &mut self,
        neurons: &mut [LtcNeuron],
        causation_output: &[f32],
        dt: f32,
    ) {
        self.short_term.update(neurons, causation_output, dt);
        self.medium_term.update(neurons, causation_output, dt);
        self.long_term.update(neurons, causation_output, dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::LtcNeuron;
    use approx::assert_abs_diff_eq;

    fn make_neurons(count: usize) -> Vec<LtcNeuron> {
        (0..count).map(|_| LtcNeuron::new()).collect()
    }

    #[test]
    fn write_gate_sigmoid_bounds() {
        let gate = WriteGate::new(3);
        let output = gate.forward(&[1.0, 2.0, 3.0]);
        assert_abs_diff_eq!(output, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn write_gate_high_input_approaches_one() {
        let mut gate = WriteGate::new(1);
        gate.weights = vec![10.0];
        gate.bias = 0.0;
        let output = gate.forward(&[1.0]);
        assert!(output > 0.99);
    }

    #[test]
    fn write_gate_low_input_approaches_zero() {
        let mut gate = WriteGate::new(1);
        gate.weights = vec![10.0];
        gate.bias = 0.0;
        let output = gate.forward(&[-1.0]);
        assert!(output < 0.01);
    }

    #[test]
    fn decay_reduces_activation() {
        let mut neurons = make_neurons(10);
        neurons[0].x = 1.0;
        neurons[1].x = 0.5;

        let circuit = Circuit::new(
            0, "test_mem", vec![0, 1, 2],
            0.2, CircuitType::Memory, CircuitCriticality::Performance,
        );
        let mut mem = MemoryCircuit::new(circuit, 0.7, 3);

        let dt = 0.1;
        let causation_output = vec![0.0, 0.0, 0.0];
        mem.update(&mut neurons, &causation_output, dt);

        assert!(neurons[0].x < 1.0, "Neuron should have decayed");
    }

    #[test]
    fn write_gate_blocks_low_confidence() {
        let mut neurons = make_neurons(5);
        neurons[0].x = 0.8;

        let circuit = Circuit::new(
            0, "test_mem", vec![0, 1, 2],
            0.2, CircuitType::Memory, CircuitCriticality::Performance,
        );
        let mut mem = MemoryCircuit::new(circuit, 0.7, 3);

        mem.write_gate.bias = -10.0;

        let before = neurons[0].x;
        let dt = 0.1;
        mem.update(&mut neurons, &[0.0, 0.0, 0.0], dt);

        let expected = before * (-0.7_f32 * 0.1).exp();
        assert_abs_diff_eq!(neurons[0].x, expected, epsilon = 1e-5);
    }

    #[test]
    fn hierarchical_memory_all_levels_update() {
        let mut neurons = make_neurons(30);
        neurons[0].x = 0.9;
        neurons[10].x = 0.7;
        neurons[20].x = 0.5;

        let mut memory = HierarchicalMemory::new(0..10, 10..20, 20..30, 3);
        let causation = vec![0.0, 0.0, 0.0];

        memory.update(&mut neurons, &causation, 0.001);

        assert!(neurons[0].x < 0.9);
        assert!(neurons[10].x < 0.7);
        assert!(neurons[20].x < 0.5);
    }

    #[test]
    fn short_term_decays_faster_than_long_term() {
        let mut neurons = make_neurons(30);
        neurons[0].x = 1.0;
        neurons[20].x = 1.0;

        let mut memory = HierarchicalMemory::new(0..10, 10..20, 20..30, 3);
        memory.short_term.write_gate.bias = -20.0;
        memory.medium_term.write_gate.bias = -20.0;
        memory.long_term.write_gate.bias = -20.0;

        for _ in 0..1000 {
            memory.update(&mut neurons, &[0.0; 3], 0.001);
        }

        assert!(
            neurons[0].x < neurons[20].x,
            "Short-term ({}) should have decayed more than long-term ({})",
            neurons[0].x,
            neurons[20].x
        );

        assert!(neurons[0].x < 0.6, "Short-term should be well-decayed: {}", neurons[0].x);
        assert!(neurons[20].x > 0.99, "Long-term should barely have decayed: {}", neurons[20].x);
    }
}

//! Adapter layers: thin transformation between frozen universal circuits
//! and plastic adaptive circuits.

use crate::neuron::LtcNeuron;
use serde::{Deserialize, Serialize};

/// A single adapter neuron (not part of the main mesh neuron array).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterNeuron {
    pub x: f32,
    pub weights_in: Vec<f32>,
    pub weights_out: Vec<f32>,
    pub bias: f32,
}

impl AdapterNeuron {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            x: 0.0,
            weights_in: vec![0.0; input_dim],
            weights_out: vec![0.0; output_dim],
            bias: 0.0,
        }
    }
}

/// Thin transformation layer between frozen universal circuits
/// and plastic adaptive circuits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterLayer {
    /// Adapter neurons.
    pub neurons: Vec<AdapterNeuron>,

    /// Which universal circuit output neuron indices feed into this adapter.
    pub input_from: Vec<usize>,

    /// Which adaptive circuit input neuron indices this adapter feeds.
    pub output_to: Vec<usize>,

    /// Whether this adapter is currently learning.
    pub trainable: bool,
}

impl AdapterLayer {
    /// Create an adapter layer with the given size.
    pub fn new(
        size: usize,
        input_from: Vec<usize>,
        output_to: Vec<usize>,
    ) -> Self {
        let input_dim = input_from.len();
        let output_dim = output_to.len();

        Self {
            neurons: (0..size)
                .map(|_| AdapterNeuron::new(input_dim, output_dim))
                .collect(),
            input_from,
            output_to,
            trainable: true,
        }
    }

    /// Forward pass: read from universal mesh neurons, compute adapter
    /// activations, write to adaptive circuit input neurons.
    pub fn forward(
        &mut self,
        universal_neurons: &[LtcNeuron],
        adaptive_neurons: &mut [LtcNeuron],
    ) {
        // Gather inputs from universal mesh.
        let inputs: Vec<f32> = self
            .input_from
            .iter()
            .map(|&idx| {
                if idx < universal_neurons.len() {
                    universal_neurons[idx].x
                } else {
                    0.0
                }
            })
            .collect();

        // Compute adapter neuron activations.
        for adapter_neuron in &mut self.neurons {
            let sum: f32 = adapter_neuron
                .weights_in
                .iter()
                .zip(&inputs)
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + adapter_neuron.bias;

            adapter_neuron.x = sum.tanh();
        }

        // Write adapter outputs to adaptive circuit neurons.
        for (out_idx_pos, &out_idx) in self.output_to.iter().enumerate() {
            if out_idx < adaptive_neurons.len() {
                let sum: f32 = self
                    .neurons
                    .iter()
                    .map(|an| {
                        if out_idx_pos < an.weights_out.len() {
                            an.x * an.weights_out[out_idx_pos]
                        } else {
                            0.0
                        }
                    })
                    .sum();

                adaptive_neurons[out_idx].x += sum;
            }
        }
    }

    /// Placeholder for gradient-free training step.
    pub fn train_step(
        &mut self,
        _universal_output: &[f32],
        _expected: &std::collections::HashMap<u32, Vec<f32>>,
        _lr: f32,
    ) {
        if !self.trainable {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::LtcNeuron;

    fn make_neurons(count: usize) -> Vec<LtcNeuron> {
        (0..count).map(|_| LtcNeuron::new()).collect()
    }

    #[test]
    fn adapter_forward_produces_output() {
        let mut universal = make_neurons(10);
        universal[0].x = 0.5;
        universal[1].x = 0.3;

        let mut adaptive = make_neurons(5);

        let mut adapter = AdapterLayer::new(4, vec![0, 1], vec![3, 4]);

        for neuron in &mut adapter.neurons {
            neuron.weights_in = vec![0.5, 0.5];
            neuron.weights_out = vec![0.3, 0.3];
            neuron.bias = 0.0;
        }

        adapter.forward(&universal, &mut adaptive);

        assert!(adapter.neurons.iter().any(|n| n.x.abs() > 0.0));
    }

    #[test]
    fn frozen_adapter_does_not_train() {
        let mut adapter = AdapterLayer::new(4, vec![0, 1], vec![3, 4]);
        adapter.trainable = false;
        adapter.train_step(&[], &std::collections::HashMap::new(), 1e-3);
    }

    #[test]
    fn adapter_neuron_tanh_bounded() {
        let mut universal = make_neurons(2);
        universal[0].x = 100.0;
        universal[1].x = -100.0;

        let mut adaptive = make_neurons(2);

        let mut adapter = AdapterLayer::new(2, vec![0, 1], vec![0, 1]);
        for neuron in &mut adapter.neurons {
            neuron.weights_in = vec![1.0, 1.0];
            neuron.weights_out = vec![1.0, 1.0];
        }

        adapter.forward(&universal, &mut adaptive);

        for n in &adapter.neurons {
            assert!(n.x.abs() <= 1.0, "Adapter neuron exceeded tanh bounds: {}", n.x);
        }
    }
}

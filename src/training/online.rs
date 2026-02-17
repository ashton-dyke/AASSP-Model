//! Online adapter learning during live drilling.
//!
//! Only the formation and well adapter layers train online.
//! The universal mesh core remains frozen.

use crate::adaptation::adapter::AdapterLayer;
use crate::neuron::LiquidNeuron;

/// Compute the gradient for a single adapter layer forward pass.
///
/// Returns (weight_in_grads, weight_out_grads, bias_grads) per adapter neuron.
pub fn adapter_backward(
    adapter: &AdapterLayer,
    universal_neurons: &[LiquidNeuron],
    dl_d_adaptive: &[f32], // ∂L/∂(adaptive_neuron.x) for each output_to index
) -> Vec<(Vec<f32>, Vec<f32>, f32)> {
    if !adapter.trainable {
        return adapter
            .neurons
            .iter()
            .map(|n| (vec![0.0; n.weights_in.len()], vec![0.0; n.weights_out.len()], 0.0))
            .collect();
    }

    // Gather inputs (same as forward pass).
    let inputs: Vec<f32> = adapter
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

    let mut neuron_grads: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::new();

    for an in &adapter.neurons {
        // Forward: x = tanh(dot(weights_in, inputs) + bias)
        let pre_act: f32 = an
            .weights_in
            .iter()
            .zip(&inputs)
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + an.bias;
        let tanh_val = pre_act.tanh();
        let tanh_deriv = 1.0 - tanh_val * tanh_val;

        // Output: adaptive[out_idx] += Σ(an.x * an.weights_out[pos])
        // ∂L/∂(an.x) = Σ_pos(∂L/∂adaptive[out_idx] * an.weights_out[pos])
        let mut dl_d_an_x = 0.0f32;
        let mut d_weights_out = vec![0.0f32; an.weights_out.len()];

        for (pos, &_out_idx) in adapter.output_to.iter().enumerate() {
            if pos < an.weights_out.len() {
                // Find the gradient for this output.
                let dl_out = if pos < dl_d_adaptive.len() {
                    dl_d_adaptive[pos]
                } else {
                    0.0
                };

                // ∂L/∂weights_out[pos] = ∂L/∂adaptive * an.x
                d_weights_out[pos] = dl_out * an.x;

                // ∂L/∂an.x
                dl_d_an_x += dl_out * an.weights_out[pos];
            }
        }

        // ∂L/∂pre_act = ∂L/∂an.x * tanh'(pre_act)
        let dl_d_pre = dl_d_an_x * tanh_deriv;

        // ∂L/∂weights_in[k] = ∂L/∂pre_act * inputs[k]
        let d_weights_in: Vec<f32> = inputs.iter().map(|&x| dl_d_pre * x).collect();

        // ∂L/∂bias = ∂L/∂pre_act
        let d_bias = dl_d_pre;

        neuron_grads.push((d_weights_in, d_weights_out, d_bias));
    }

    neuron_grads
}

/// Apply gradient updates to an adapter layer.
pub fn adapter_sgd_step(
    adapter: &mut AdapterLayer,
    grads: &[(Vec<f32>, Vec<f32>, f32)],
    lr: f32,
) {
    if !adapter.trainable {
        return;
    }

    for (neuron, (d_win, d_wout, d_bias)) in adapter.neurons.iter_mut().zip(grads) {
        for (w, g) in neuron.weights_in.iter_mut().zip(d_win) {
            *w -= lr * g;
        }
        for (w, g) in neuron.weights_out.iter_mut().zip(d_wout) {
            *w -= lr * g;
        }
        neuron.bias -= lr * d_bias;
    }
}

/// Run one online training step for an adapter.
///
/// Uses the detection readout error as the loss signal.
pub fn online_train_step(
    adapter: &mut AdapterLayer,
    universal_neurons: &[LiquidNeuron],
    _adaptive_neurons: &[LiquidNeuron],
    error_signal: &[f32], // ∂L/∂(adaptive output) for each output_to position
    lr: f32,
) {
    let grads = adapter_backward(adapter, universal_neurons, error_signal);
    adapter_sgd_step(adapter, &grads, lr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptation::adapter::AdapterLayer;
    use crate::neuron::LiquidNeuron;

    fn make_neurons(n: usize) -> Vec<LiquidNeuron> {
        (0..n).map(|_| LiquidNeuron::new()).collect()
    }

    #[test]
    fn adapter_backward_produces_gradients() {
        let mut universal = make_neurons(4);
        universal[0].x = 0.5;
        universal[1].x = 0.3;

        let mut adapter = AdapterLayer::new(2, vec![0, 1], vec![2, 3]);
        for n in &mut adapter.neurons {
            n.weights_in = vec![0.5, 0.5];
            n.weights_out = vec![0.3, 0.3];
        }
        adapter.forward(&universal, &mut make_neurons(4));

        let dl_d_adaptive = vec![1.0, 0.5];
        let grads = adapter_backward(&adapter, &universal, &dl_d_adaptive);

        assert_eq!(grads.len(), 2);
        // Gradients should be non-zero.
        let (d_win, d_wout, _d_bias) = &grads[0];
        assert!(d_win.iter().any(|g| g.abs() > 1e-9));
        assert!(d_wout.iter().any(|g| g.abs() > 1e-9));
    }

    #[test]
    fn adapter_sgd_step_modifies_weights() {
        let _universal = make_neurons(2);
        let mut adapter = AdapterLayer::new(2, vec![0, 1], vec![0, 1]);
        for n in &mut adapter.neurons {
            n.weights_in = vec![0.5, 0.5];
            n.weights_out = vec![0.3, 0.3];
            n.bias = 0.1;
        }

        let grads = vec![
            (vec![0.1, 0.2], vec![0.3, 0.4], 0.05),
            (vec![0.1, 0.2], vec![0.3, 0.4], 0.05),
        ];

        let old_w = adapter.neurons[0].weights_in[0];
        adapter_sgd_step(&mut adapter, &grads, 0.01);
        let new_w = adapter.neurons[0].weights_in[0];

        assert!((new_w - (old_w - 0.01 * 0.1)).abs() < 1e-6);
    }

    #[test]
    fn frozen_adapter_not_updated() {
        let _universal = make_neurons(2);
        let mut adapter = AdapterLayer::new(2, vec![0, 1], vec![0, 1]);
        adapter.trainable = false;

        let grads = vec![
            (vec![1.0, 1.0], vec![1.0, 1.0], 1.0),
            (vec![1.0, 1.0], vec![1.0, 1.0], 1.0),
        ];

        let old_bias = adapter.neurons[0].bias;
        adapter_sgd_step(&mut adapter, &grads, 0.01);
        assert!((adapter.neurons[0].bias - old_bias).abs() < 1e-9);
    }
}

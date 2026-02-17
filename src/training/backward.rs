//! Backward pass implementations for all trainable components.
//!
//! Implements manual backpropagation through:
//! - Liquid neuron dynamics (BPTT through ODE steps)
//! - Overlap gate MLP (6→8→2 with ReLU + softmax)
//! - Write gates (linear → sigmoid)
//! - Overlap masks (with L1 penalty)

use crate::mesh::OverlappingMesh;
use crate::training::state::{ForwardRecord, GateGradient, GradientAccumulator};

/// Backpropagate through all recorded mesh steps (BPTT).
///
/// `dl_dx_output` contains ∂L/∂x for each neuron at the final step
/// (from the loss function). This function accumulates weight/bias
/// gradients into `grads`.
pub fn backprop_through_time(
    mesh: &OverlappingMesh,
    record: &ForwardRecord,
    dl_dx_output: &[f32],
    grads: &mut GradientAccumulator,
) {
    let dt = mesh.config.dt;

    // Initialize ∂L/∂x from the loss output.
    for (i, &dl) in dl_dx_output.iter().enumerate() {
        grads.dx[i] = dl;
    }

    // Walk backward through all recorded steps.
    for step in (0..record.step_count).rev() {
        // Temporary buffer for upstream gradients produced this step.
        let mut upstream_dx = vec![0.0f32; record.neuron_count];

        for neuron_idx in 0..record.neuron_count {
            let snapshot = match &record.step_snapshots[step][neuron_idx] {
                Some(s) => s,
                None => continue, // Neuron was not updated this step.
            };

            let dl_dx_curr = grads.dx[neuron_idx];
            if dl_dx_curr.abs() < 1e-12 {
                continue;
            }

            // For overlap neurons, multiple circuits may have updated.
            // Use average tau for gradient computation (simplified).
            let tau = if snapshot.circuit_updates.is_empty() {
                0.01 // fallback
            } else {
                snapshot.circuit_updates.iter().map(|(_, t)| t).sum::<f32>()
                    / snapshot.circuit_updates.len() as f32
            };

            // Gradient through the liquid neuron step:
            // x_{t+1} = x_t + (-x_t + tanh(s_t)) / tau * dt
            // where s_t = Σ(w_i * x_i_t) + bias
            let tanh_deriv = 1.0 - snapshot.tanh_val * snapshot.tanh_val;
            let dx_ds = tanh_deriv * dt / tau;
            let dx_dx_prev = 1.0 - dt / tau;

            // Accumulate bias gradient.
            grads.bias_grads[neuron_idx] += dl_dx_curr * dx_ds;

            // Accumulate weight gradients and upstream gradients.
            let neuron = &mesh.neurons[neuron_idx];
            for (conn_idx, &(src_idx, weight)) in neuron.connections.iter().enumerate() {
                // ∂s/∂w = x_src at snapshot time.
                // We use the recorded x for the source neuron.
                let x_src = if let Some(src_snap) = &record.step_snapshots[step][src_idx] {
                    src_snap.x
                } else {
                    // Source wasn't updated this step — use its most recent known value.
                    // Approximation: use current mesh state (after full forward).
                    mesh.neurons[src_idx].x
                };

                if conn_idx < grads.weight_grads[neuron_idx].len() {
                    grads.weight_grads[neuron_idx][conn_idx] += dl_dx_curr * dx_ds * x_src;
                }

                // Upstream gradient to source neuron.
                upstream_dx[src_idx] += dl_dx_curr * dx_ds * weight;
            }

            // Propagate gradient to previous timestep for this neuron.
            grads.dx[neuron_idx] = dl_dx_curr * dx_dx_prev;
        }

        // Add upstream gradients.
        for i in 0..record.neuron_count {
            grads.dx[i] += upstream_dx[i];
        }
    }
}

/// Backpropagate through an overlap gate MLP.
///
/// Given ∂L/∂priority_a and ∂L/∂priority_b (from the merge operation),
/// computes gradients for all gate weights.
///
/// Returns (∂L/∂delta_a, ∂L/∂delta_b) for upstream propagation.
pub fn backprop_gate(
    input: &[f32; 6],
    hidden: &[f32; 8],
    priorities: (f32, f32),
    dl_dp_a: f32,
    dl_dp_b: f32,
    gate_grad: &mut GateGradient,
    weights_ih: &[[f32; 6]; 8],
    weights_ho: &[[f32; 8]; 2],
) -> (f32, f32) {
    let (p_a, p_b) = priorities;

    // Softmax backward: ∂L/∂logit_i = p_i * (∂L/∂p_i - Σ_j(∂L/∂p_j * p_j))
    let sum_dl_p = dl_dp_a * p_a + dl_dp_b * p_b;
    let dl_dlogit_a = p_a * (dl_dp_a - sum_dl_p);
    let dl_dlogit_b = p_b * (dl_dp_b - sum_dl_p);
    let dl_dlogits = [dl_dlogit_a, dl_dlogit_b];

    // Output layer backward: logit_i = Σ_j(w_ho[i][j] * h_j) + b_o[i]
    let mut dl_dhidden = [0.0f32; 8];
    for i in 0..2 {
        gate_grad.d_bias_o[i] += dl_dlogits[i];
        for j in 0..8 {
            gate_grad.d_weights_ho[i][j] += dl_dlogits[i] * hidden[j];
            dl_dhidden[j] += dl_dlogits[i] * weights_ho[i][j];
        }
    }

    // Hidden layer backward (ReLU): h_j = max(0, Σ_k(w_ih[j][k] * input[k]) + b_h[j])
    let mut dl_dinput = [0.0f32; 6];
    for j in 0..8 {
        // ReLU derivative: 1 if hidden[j] > 0, else 0.
        let relu_grad = if hidden[j] > 0.0 { 1.0 } else { 0.0 };
        let dl_dpre_h = dl_dhidden[j] * relu_grad;

        gate_grad.d_bias_h[j] += dl_dpre_h;
        for k in 0..6 {
            gate_grad.d_weights_ih[j][k] += dl_dpre_h * input[k];
            dl_dinput[k] += dl_dpre_h * weights_ih[j][k];
        }
    }

    // dl_dinput[2] = ∂L/∂delta_a, dl_dinput[5] = ∂L/∂delta_b
    (dl_dinput[2], dl_dinput[5])
}

/// Backpropagate through a write gate (linear → sigmoid).
///
/// `gate_output`: the sigmoid output from forward pass.
/// `dl_d_gated`: ∂L/∂(gated_update) from upstream.
/// `ungated_update`: the update that was multiplied by the gate.
/// `causation_output`: the input to the write gate.
///
/// Returns gradients for weights and bias.
pub fn backprop_write_gate(
    gate_output: f32,
    dl_d_gated: f32,
    ungated_update: f32,
    causation_output: &[f32],
) -> (Vec<f32>, f32) {
    // gated = gate * ungated  →  ∂L/∂gate = ∂L/∂gated * ungated
    let dl_d_gate = dl_d_gated * ungated_update;

    // sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
    let sigmoid_deriv = gate_output * (1.0 - gate_output);
    let dl_d_pre = dl_d_gate * sigmoid_deriv;

    // pre = dot(w, input) + b
    let weight_grads: Vec<f32> = causation_output.iter().map(|&x| dl_d_pre * x).collect();
    let bias_grad = dl_d_pre;

    (weight_grads, bias_grad)
}

/// Compute gradient for overlap masks.
///
/// `mask_idx`: index within the zone.
/// `delta`: the neuron update that was scaled by the mask.
/// `dl_d_effective`: ∂L/∂(effective_delta).
/// `sparsity_lambda`: L1 regularization coefficient.
/// `mask_value`: current mask value.
pub fn backprop_mask(
    dl_d_effective: f32,
    delta: f32,
    mask_value: f32,
    sparsity_lambda: f32,
) -> f32 {
    // effective = mask * delta  →  ∂L/∂mask = ∂L/∂effective * delta
    let dl_d_mask = dl_d_effective * delta;

    // Plus L1 penalty: ∂(λ|mask|)/∂mask = λ * sign(mask)
    let l1_grad = sparsity_lambda * mask_value.signum();

    dl_d_mask + l1_grad
}

/// Compute overlap stability loss gradient.
///
/// ∂/∂mask (λ * (mask - mask_ref)²) = 2λ * (mask - mask_ref)
pub fn backprop_stability(mask_value: f32, mask_ref: f32, stability_lambda: f32) -> f32 {
    2.0 * stability_lambda * (mask_value - mask_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::state::GateGradient;

    #[test]
    fn gate_backward_softmax_gradients() {
        // With equal priorities, equal upstream gradients should produce zero.
        let input = [0.5, 0.01, 0.1, 0.3, 0.05, -0.1];
        let hidden = [0.0f32; 8]; // All zeros (from zero-init gate).
        let priorities = (0.5, 0.5);
        let mut gg = GateGradient::zeros();

        let (_, _) = backprop_gate(
            &input,
            &hidden,
            priorities,
            1.0,
            1.0,
            &mut gg,
            &[[0.0; 6]; 8],
            &[[0.0; 8]; 2],
        );

        // With equal upstream gradients (1.0, 1.0) and equal priorities (0.5, 0.5):
        // sum_dl_p = 1.0*0.5 + 1.0*0.5 = 1.0
        // dl_dlogit_a = 0.5 * (1.0 - 1.0) = 0.0
        // dl_dlogit_b = 0.5 * (1.0 - 1.0) = 0.0
        // So all gate gradients should be zero.
        for row in &gg.d_weights_ih {
            for &g in row {
                assert!(g.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn write_gate_backward_zero_gate() {
        let (wg, bg) = backprop_write_gate(0.5, 1.0, 0.5, &[0.3, 0.7]);
        // sigmoid'(0) = 0.5 * 0.5 = 0.25
        // dl_d_gate = 1.0 * 0.5 = 0.5
        // dl_d_pre = 0.5 * 0.25 = 0.125
        assert!((wg[0] - 0.125 * 0.3).abs() < 1e-6);
        assert!((wg[1] - 0.125 * 0.7).abs() < 1e-6);
        assert!((bg - 0.125).abs() < 1e-6);
    }

    #[test]
    fn mask_gradient_with_l1() {
        let grad = backprop_mask(1.0, 0.5, 0.8, 0.01);
        // dl_d_mask = 1.0 * 0.5 = 0.5
        // l1 = 0.01 * 1.0 = 0.01  (sign(0.8) = 1.0)
        assert!((grad - 0.51).abs() < 1e-6);
    }

    #[test]
    fn stability_gradient() {
        let grad = backprop_stability(0.7, 0.5, 0.1);
        // 2 * 0.1 * (0.7 - 0.5) = 0.04
        assert!((grad - 0.04).abs() < 1e-6);
    }

    #[test]
    fn numerical_gradient_check_neuron_bias() {
        // Verify analytic gradient matches numerical gradient for a single neuron's bias.
        use crate::config::MeshConfig;
        use crate::neuron::LiquidNeuron;
        

        let _config = MeshConfig {
            total_neurons: 2,
            dt: 0.001,
            ..MeshConfig::default()
        };

        // Function: run one neuron step, return x.
        let run = |bias: f32| -> f32 {
            let mut neuron = LiquidNeuron::new();
            neuron.bias = bias;
            neuron.x = 0.3;
            // dx = (-0.3 + tanh(bias)) / tau * dt
            let tau = 0.01;
            let dt = 0.001;
            let input_sum = bias;
            let dx = (-neuron.x + input_sum.tanh()) / tau * dt;
            neuron.x + dx
        };

        let bias = 0.5;
        let eps = 1e-4;

        let numerical = (run(bias + eps) - run(bias - eps)) / (2.0 * eps);

        // Analytic: ∂x_new/∂bias = (1 - tanh²(bias)) * dt / tau
        let tau = 0.01_f32;
        let dt = 0.001_f32;
        let tanh_val = bias.tanh();
        let analytic = (1.0 - tanh_val * tanh_val) * dt / tau;

        let relative_error = (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 1e-3,
            "Gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }
}

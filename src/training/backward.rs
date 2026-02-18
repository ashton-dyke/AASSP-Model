//! Backward pass implementations for all trainable components.
//!
//! Implements manual backpropagation through:
//! - LTC/CfC neuron dynamics (BPTT with analytical CfC gradients)
//! - Write gates (linear -> sigmoid)
//! - Overlap masks (with L1 penalty)

use crate::mesh::OverlappingMesh;

use crate::training::state::{ForwardRecord, GradientAccumulator};

/// Backpropagate through all recorded mesh steps (BPTT) for CfC neurons.
///
/// `dl_dx_output` contains dL/dx for each neuron at the final step
/// (from the loss function). This function accumulates weight/bias/gate/tau
/// gradients into `grads`.
///
/// CfC forward:
///   f = sigmoid(gate_sum)
///   A = tanh(drive_sum)
///   alpha = 1/tau + f
///   E = exp(-alpha * dt)
///   x_new = x_prev * E + (f * A / alpha) * (1 - E)
///
/// Analytical gradients:
///   dx_new/dx_prev = E
///   dx_new/df = dt * E * (f*A/alpha - x_prev) + A/(tau*alpha^2) * (1 - E)
///   dx_new/dA = (f/alpha) * (1 - E)
///   dx_new/dtau = dt*E/tau^2 * (x_prev - f*A/alpha) + f*A/(alpha^2*tau^2) * (1 - E)
///
/// Chain rules:
///   df/dgate_sum = f * (1 - f)
///   dA/ddrive_sum = 1 - A^2
///   ddrive_sum/dw_i = x_src_i,  ddrive_sum/dbias = 1
///   dgate_sum/dgate_w_i = x_src_i,  dgate_sum/dgate_bias = 1
pub fn backprop_through_time(
    mesh: &OverlappingMesh,
    record: &ForwardRecord,
    dl_dx_output: &[f32],
    grads: &mut GradientAccumulator,
) {
    let dt = mesh.config.dt;

    // Initialize dL/dx from the loss output.
    for (i, &dl) in dl_dx_output.iter().enumerate() {
        grads.dx[i] = dl;
    }

    // Walk backward through all recorded steps.
    for step in (0..record.step_count).rev() {
        // Temporary buffer for upstream gradients produced this step.
        let mut upstream_dx = vec![0.0f32; record.neuron_count];

        for neuron_idx in 0..record.neuron_count {
            let inter = &record.step_intermediates[step][neuron_idx];

            let dl_dx_curr = grads.dx[neuron_idx];
            if dl_dx_curr.abs() < 1e-12 {
                continue;
            }

            let f = inter.f;
            let a = inter.a;
            let alpha = inter.alpha;
            let exp_term = inter.exp_term;
            let x_prev = inter.x_prev;
            let tau = inter.tau;

            // Precompute common terms.
            let f_a_over_alpha = f * a / alpha;
            let one_minus_e = 1.0 - exp_term;

            // dx_new/df = dt * E * (f*A/alpha - x_prev) + A/(tau*alpha^2) * (1 - E)
            let dx_df = dt * exp_term * (f_a_over_alpha - x_prev)
                + a / (tau * alpha * alpha) * one_minus_e;

            // dx_new/dA = (f/alpha) * (1 - E)
            let dx_da = (f / alpha) * one_minus_e;

            // dx_new/dtau = dt*E/tau^2*(x_prev - f*A/alpha) + f*A/(alpha^2*tau^2)*(1-E)
            let tau_sq = tau * tau;
            let dx_dtau = dt * exp_term / tau_sq * (x_prev - f_a_over_alpha)
                + f * a / (alpha * alpha * tau_sq) * one_minus_e;

            // dx_new/dx_prev = E
            let dx_dx_prev = exp_term;

            // Chain through gate: df/dgate_sum = f*(1-f)
            let df_dgate_sum = f * (1.0 - f);

            // Chain through drive: dA/ddrive_sum = 1 - A^2
            let da_ddrive_sum = 1.0 - a * a;

            // Full chain: dL/dgate_sum = dL/dx_new * dx_new/df * df/dgate_sum
            let dl_dgate_sum = dl_dx_curr * dx_df * df_dgate_sum;

            // Full chain: dL/ddrive_sum = dL/dx_new * dx_new/dA * dA/ddrive_sum
            let dl_ddrive_sum = dl_dx_curr * dx_da * da_ddrive_sum;

            // Accumulate bias gradient: ddrive_sum/dbias = 1
            grads.bias_grads[neuron_idx] += dl_ddrive_sum;

            // Accumulate gate_bias gradient: dgate_sum/dgate_bias = 1
            grads.gate_bias_grads[neuron_idx] += dl_dgate_sum;

            // Accumulate tau gradient: dL/dtau = dL/dx_new * dx_new/dtau
            grads.tau_grads[neuron_idx] += dl_dx_curr * dx_dtau;

            // Accumulate weight gradients and upstream gradients to source neurons.
            let neuron = &mesh.neurons[neuron_idx];
            for (conn_idx, &(src_idx, _drive_w)) in neuron.connections.iter().enumerate() {
                // Get source neuron x at the time of this step.
                // Use the x_prev from source neuron's intermediate at this step,
                // which is the state before update (i.e., the value used in the sums).
                let x_src = record.step_intermediates[step][src_idx].x_prev;

                // Drive weight gradient: dL/dw_i = dL/ddrive_sum * x_src
                if conn_idx < grads.weight_grads[neuron_idx].len() {
                    grads.weight_grads[neuron_idx][conn_idx] += dl_ddrive_sum * x_src;
                }

                // Gate weight gradient: dL/dgate_w_i = dL/dgate_sum * x_src
                if conn_idx < grads.gate_weight_grads[neuron_idx].len() {
                    grads.gate_weight_grads[neuron_idx][conn_idx] += dl_dgate_sum * x_src;
                }

                // Upstream gradient to source neuron through drive path:
                //   dL/dx_src += dL/ddrive_sum * drive_w
                let drive_w = neuron.connections[conn_idx].1;
                upstream_dx[src_idx] += dl_ddrive_sum * drive_w;

                // Upstream gradient to source neuron through gate path:
                //   dL/dx_src += dL/dgate_sum * gate_w
                if conn_idx < neuron.gate_weights.len() {
                    upstream_dx[src_idx] += dl_dgate_sum * neuron.gate_weights[conn_idx];
                }
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

/// Backpropagate through a write gate (linear -> sigmoid).
///
/// `gate_output`: the sigmoid output from forward pass.
/// `dl_d_gated`: dL/d(gated_update) from upstream.
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
    // gated = gate * ungated  ->  dL/dgate = dL/dgated * ungated
    let dl_d_gate = dl_d_gated * ungated_update;

    // sigmoid derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
    let sigmoid_deriv = gate_output * (1.0 - gate_output);
    let dl_d_pre = dl_d_gate * sigmoid_deriv;

    // pre = dot(w, input) + b
    let weight_grads: Vec<f32> = causation_output.iter().map(|&x| dl_d_pre * x).collect();
    let bias_grad = dl_d_pre;

    (weight_grads, bias_grad)
}

/// Compute gradient for overlap masks.
///
/// `delta`: the neuron update that was scaled by the mask.
/// `dl_d_effective`: dL/d(effective_delta).
/// `sparsity_lambda`: L1 regularization coefficient.
/// `mask_value`: current mask value.
pub fn backprop_mask(
    dl_d_effective: f32,
    delta: f32,
    mask_value: f32,
    sparsity_lambda: f32,
) -> f32 {
    // effective = mask * delta  ->  dL/dmask = dL/deffective * delta
    let dl_d_mask = dl_d_effective * delta;

    // Plus L1 penalty: d(lambda|mask|)/dmask = lambda * sign(mask)
    let l1_grad = sparsity_lambda * mask_value.signum();

    dl_d_mask + l1_grad
}

/// Compute overlap stability loss gradient.
///
/// d/dmask (lambda * (mask - mask_ref)^2) = 2*lambda * (mask - mask_ref)
pub fn backprop_stability(mask_value: f32, mask_ref: f32, stability_lambda: f32) -> f32 {
    2.0 * stability_lambda * (mask_value - mask_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{cfc_forward, LtcNeuron};

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

    /// Helper: compute analytic dx/dbias, dx/dgate_bias, dx/dtau for a CfC neuron.
    /// This uses the full CfC gradient formulas from the intermediates.
    fn analytic_grads_from_inter(
        inter: &crate::neuron::CfcIntermediate,
        dt: f32,
    ) -> (f32, f32, f32) {
        let f = inter.f;
        let a = inter.a;
        let alpha = inter.alpha;
        let exp_term = inter.exp_term;
        let x_prev = inter.x_prev;
        let tau = inter.tau;
        let one_minus_e = 1.0 - exp_term;
        let f_a_over_alpha = f * a / alpha;

        // dx/dA = (f/alpha)*(1-E)
        let dx_da = (f / alpha) * one_minus_e;
        // dA/ddrive_sum = 1 - A^2
        let da_ddrive = 1.0 - a * a;
        // dx/dbias = dx_da * da_ddrive (bias only affects drive, not gate)
        let dx_dbias = dx_da * da_ddrive;

        // dx/df = dt * E * (f*A/alpha - x_prev) + A/(tau*alpha^2)*(1-E)
        let dx_df = dt * exp_term * (f_a_over_alpha - x_prev)
            + a / (tau * alpha * alpha) * one_minus_e;
        // df/dgate_bias = f*(1-f)
        let dx_dgate_bias = dx_df * f * (1.0 - f);

        // dx/dtau = dt*E/tau^2*(x_prev - f*A/alpha) + f*A/(alpha^2*tau^2)*(1-E)
        let tau_sq = tau * tau;
        let dx_dtau = dt * exp_term / tau_sq * (x_prev - f_a_over_alpha)
            + f * a / (alpha * alpha * tau_sq) * one_minus_e;

        (dx_dbias, dx_dgate_bias, dx_dtau)
    }

    #[test]
    fn numerical_gradient_check_cfc_bias() {
        // Use larger tau (slower decay => bigger 1-E term) for numerically stable gradients.
        let dt = 0.1_f32;
        let eps = 1e-4_f32;
        let base_bias = 0.5_f32;

        let run = |bias: f32| -> f32 {
            let mut neuron = LtcNeuron::new();
            neuron.bias = bias;
            neuron.x = 0.3;
            neuron.tau = 1.0; // Slow tau => alpha ~ 1 + f ~ 1.5
            let neurons = [neuron.clone()];
            let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
            x_new
        };

        let numerical = (run(base_bias + eps) - run(base_bias - eps)) / (2.0 * eps);

        let mut neuron = LtcNeuron::new();
        neuron.bias = base_bias;
        neuron.x = 0.3;
        neuron.tau = 1.0;
        let neurons = [neuron.clone()];
        let (_, inter) = cfc_forward(&neuron, &neurons, dt);

        let (analytic, _, _) = analytic_grads_from_inter(&inter, dt);

        let relative_error =
            (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 0.01,
            "Bias gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }

    #[test]
    fn numerical_gradient_check_cfc_gate_bias() {
        // Use larger dt and larger eps for cleaner numerics.
        // The gate_bias gradient goes through f which has a small effect on x,
        // so we need large enough perturbation to get above float32 noise.
        let dt = 0.5_f32;
        let eps = 1e-3_f32;
        let base_gate_bias = 0.0_f32;

        let run = |gate_bias: f32| -> f32 {
            let mut neuron = LtcNeuron::new();
            neuron.bias = 0.5;
            neuron.gate_bias = gate_bias;
            neuron.x = 0.3;
            neuron.tau = 1.0;
            let neurons = [neuron.clone()];
            let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
            x_new
        };

        let numerical = (run(base_gate_bias + eps) - run(base_gate_bias - eps)) / (2.0 * eps);

        let mut neuron = LtcNeuron::new();
        neuron.bias = 0.5;
        neuron.gate_bias = base_gate_bias;
        neuron.x = 0.3;
        neuron.tau = 1.0;
        let neurons = [neuron.clone()];
        let (_, inter) = cfc_forward(&neuron, &neurons, dt);

        let (_, analytic, _) = analytic_grads_from_inter(&inter, dt);

        let relative_error =
            (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 0.02,
            "Gate bias gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }

    #[test]
    fn numerical_gradient_check_cfc_tau() {
        let dt = 0.1_f32;
        let eps = 1e-5_f32;
        let base_tau = 1.0_f32;

        let run = |tau: f32| -> f32 {
            let mut neuron = LtcNeuron::new();
            neuron.bias = 0.5;
            neuron.x = 0.3;
            neuron.tau = tau;
            let neurons = [neuron.clone()];
            let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
            x_new
        };

        let numerical = (run(base_tau + eps) - run(base_tau - eps)) / (2.0 * eps);

        let mut neuron = LtcNeuron::new();
        neuron.bias = 0.5;
        neuron.x = 0.3;
        neuron.tau = base_tau;
        let neurons = [neuron.clone()];
        let (_, inter) = cfc_forward(&neuron, &neurons, dt);

        let (_, _, analytic) = analytic_grads_from_inter(&inter, dt);

        let relative_error =
            (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 0.01,
            "Tau gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }

    #[test]
    fn numerical_gradient_check_cfc_weight() {
        // Use larger tau and dt for stable numerical gradients.
        let dt = 0.1_f32;
        let eps = 1e-4_f32;
        let base_w = 0.3_f32;

        let run = |w: f32| -> f32 {
            let n0 = LtcNeuron {
                x: 0.5,
                tau: 1.0,
                bias: 0.0,
                gate_bias: 0.0,
                connections: vec![],
                gate_weights: vec![],
                circuit_membership_count: 0,
            };
            let n1 = LtcNeuron {
                x: 0.1,
                tau: 1.0,
                bias: 0.2,
                gate_bias: 0.0,
                connections: vec![(0, w)],
                gate_weights: vec![0.0],
                circuit_membership_count: 0,
            };
            let neurons = [n0, n1.clone()];
            let (x_new, _) = cfc_forward(&n1, &neurons, dt);
            x_new
        };

        let numerical = (run(base_w + eps) - run(base_w - eps)) / (2.0 * eps);

        // Analytic.
        let n0 = LtcNeuron {
            x: 0.5,
            tau: 1.0,
            bias: 0.0,
            gate_bias: 0.0,
            connections: vec![],
            gate_weights: vec![],
            circuit_membership_count: 0,
        };
        let n1 = LtcNeuron {
            x: 0.1,
            tau: 1.0,
            bias: 0.2,
            gate_bias: 0.0,
            connections: vec![(0, base_w)],
            gate_weights: vec![0.0],
            circuit_membership_count: 0,
        };
        let neurons = [n0, n1.clone()];
        let (_, inter) = cfc_forward(&n1, &neurons, dt);

        let f = inter.f;
        let a = inter.a;
        let alpha = inter.alpha;
        let one_minus_e = 1.0 - inter.exp_term;

        let dx_da = (f / alpha) * one_minus_e;
        let da_ddrive = 1.0 - a * a;
        let x_src = 0.5_f32;
        let analytic = dx_da * da_ddrive * x_src;

        let relative_error =
            (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 0.01,
            "Weight gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }

    #[test]
    fn numerical_gradient_check_cfc_gate_weight() {
        let dt = 0.1_f32;
        let eps = 1e-4_f32;
        let base_gw = 0.2_f32;

        let run = |gw: f32| -> f32 {
            let n0 = LtcNeuron {
                x: 0.5,
                tau: 1.0,
                bias: 0.0,
                gate_bias: 0.0,
                connections: vec![],
                gate_weights: vec![],
                circuit_membership_count: 0,
            };
            let n1 = LtcNeuron {
                x: 0.1,
                tau: 1.0,
                bias: 0.2,
                gate_bias: 0.0,
                connections: vec![(0, 0.3)],
                gate_weights: vec![gw],
                circuit_membership_count: 0,
            };
            let neurons = [n0, n1.clone()];
            let (x_new, _) = cfc_forward(&n1, &neurons, dt);
            x_new
        };

        let numerical = (run(base_gw + eps) - run(base_gw - eps)) / (2.0 * eps);

        // Analytic.
        let n0 = LtcNeuron {
            x: 0.5,
            tau: 1.0,
            bias: 0.0,
            gate_bias: 0.0,
            connections: vec![],
            gate_weights: vec![],
            circuit_membership_count: 0,
        };
        let n1 = LtcNeuron {
            x: 0.1,
            tau: 1.0,
            bias: 0.2,
            gate_bias: 0.0,
            connections: vec![(0, 0.3)],
            gate_weights: vec![base_gw],
            circuit_membership_count: 0,
        };
        let neurons = [n0, n1.clone()];
        let (_, inter) = cfc_forward(&n1, &neurons, dt);

        let f = inter.f;
        let a = inter.a;
        let alpha = inter.alpha;
        let exp_term = inter.exp_term;
        let x_prev = inter.x_prev;
        let tau = inter.tau;
        let one_minus_e = 1.0 - exp_term;
        let f_a_over_alpha = f * a / alpha;

        let dx_df = dt * exp_term * (f_a_over_alpha - x_prev)
            + a / (tau * alpha * alpha) * one_minus_e;
        let df_dgate = f * (1.0 - f);
        let x_src = 0.5_f32;
        let analytic = dx_df * df_dgate * x_src;

        let relative_error =
            (analytic - numerical).abs() / (analytic.abs() + numerical.abs() + 1e-8);
        assert!(
            relative_error < 0.01,
            "Gate weight gradient check failed: analytic={analytic}, numerical={numerical}, rel_err={relative_error}"
        );
    }
}

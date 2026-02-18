//! Liquid Time-Constant (LTC) neuron with Closed-form Continuous-depth (CfC) evaluation.
//!
//! Each neuron has input-dependent gating that modulates its effective time constant:
//!
//!   f = sigmoid(Σ(gate_wᵢ * xᵢ) + gate_bias)       — input-dependent gate
//!   A = tanh(Σ(wᵢ * xᵢ) + bias)                     — drive signal
//!   α = 1/τ + f                                       — effective decay rate
//!   x(t+dt) = x(t)·exp(-α·dt) + (f·A/α)·(1 - exp(-α·dt))  — CfC closed-form

use serde::{Deserialize, Serialize};

/// Sigmoid activation.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A single LTC neuron in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcNeuron {
    /// Current activation state.
    pub x: f32,

    /// Per-neuron base time constant (learned).
    pub tau: f32,

    /// Drive pathway bias.
    pub bias: f32,

    /// Gate pathway bias.
    pub gate_bias: f32,

    /// Incoming connections for the drive pathway: (source_neuron_index, weight).
    /// Sparse representation — only non-zero connections stored.
    pub connections: Vec<(usize, f32)>,

    /// Gate weights, parallel to connections (one gate weight per connection).
    pub gate_weights: Vec<f32>,

    /// Number of circuits this neuron currently belongs to (enforced max: 2).
    pub circuit_membership_count: u8,
}

impl LtcNeuron {
    /// Create a new neuron with zero state and no connections.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            tau: 0.01,
            bias: 0.0,
            gate_bias: 0.0,
            connections: Vec::new(),
            gate_weights: Vec::new(),
            circuit_membership_count: 0,
        }
    }

    /// Create a neuron with given bias and connections (gate weights zeroed).
    pub fn with_connections(bias: f32, connections: Vec<(usize, f32)>) -> Self {
        let gate_weights = vec![0.0; connections.len()];
        Self {
            x: 0.0,
            tau: 0.01,
            bias,
            gate_bias: 0.0,
            connections,
            gate_weights,
            circuit_membership_count: 0,
        }
    }
}

impl Default for LtcNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediate values from a CfC forward step, needed for backprop.
#[derive(Debug, Clone, Copy)]
pub struct CfcIntermediate {
    /// Previous activation (before update).
    pub x_prev: f32,
    /// Gate output: f = sigmoid(gate_sum).
    pub f: f32,
    /// Drive signal: A = tanh(drive_sum).
    pub a: f32,
    /// Effective decay rate: α = 1/τ + f.
    pub alpha: f32,
    /// Exponential term: exp(-α·dt).
    pub exp_term: f32,
    /// Pre-activation sum for drive: Σ(wᵢ·xᵢ) + bias.
    pub drive_sum: f32,
    /// Pre-activation sum for gate: Σ(gate_wᵢ·xᵢ) + gate_bias.
    pub gate_sum: f32,
    /// Per-neuron tau.
    pub tau: f32,
}

/// Compute the CfC closed-form forward step for a single neuron.
///
/// Returns (new_x, intermediate) where intermediate holds values for backprop.
pub fn cfc_forward(
    neuron: &LtcNeuron,
    all_neurons: &[LtcNeuron],
    dt: f32,
) -> (f32, CfcIntermediate) {
    let mut drive_sum = neuron.bias;
    let mut gate_sum = neuron.gate_bias;

    for (i, &(src_idx, drive_w)) in neuron.connections.iter().enumerate() {
        let src_x = all_neurons[src_idx].x;
        drive_sum += drive_w * src_x;
        if i < neuron.gate_weights.len() {
            gate_sum += neuron.gate_weights[i] * src_x;
        }
    }

    let f = sigmoid(gate_sum);
    let a = drive_sum.tanh();
    let alpha = (1.0 / neuron.tau) + f;
    let exp_term = (-alpha * dt).exp();

    // CfC closed-form: x_new = x·exp(-α·dt) + (f·A/α)·(1 - exp(-α·dt))
    let x_new = neuron.x * exp_term + (f * a / alpha) * (1.0 - exp_term);

    let intermediate = CfcIntermediate {
        x_prev: neuron.x,
        f,
        a,
        alpha,
        exp_term,
        drive_sum,
        gate_sum,
        tau: neuron.tau,
    };

    (x_new, intermediate)
}

/// Compute only the new x value (no intermediates). For inference-only use.
pub fn cfc_forward_inference(
    neuron: &LtcNeuron,
    all_neurons: &[LtcNeuron],
    dt: f32,
) -> f32 {
    let mut drive_sum = neuron.bias;
    let mut gate_sum = neuron.gate_bias;

    for (i, &(src_idx, drive_w)) in neuron.connections.iter().enumerate() {
        let src_x = all_neurons[src_idx].x;
        drive_sum += drive_w * src_x;
        if i < neuron.gate_weights.len() {
            gate_sum += neuron.gate_weights[i] * src_x;
        }
    }

    let f = sigmoid(gate_sum);
    let a = drive_sum.tanh();
    let alpha = (1.0 / neuron.tau) + f;
    let exp_term = (-alpha * dt).exp();

    neuron.x * exp_term + (f * a / alpha) * (1.0 - exp_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zero_input_decays_toward_zero() {
        let mut neuron = LtcNeuron::new();
        neuron.x = 0.8;
        neuron.tau = 0.01;
        let neurons = [neuron.clone()];
        let dt = 0.001;

        let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
        // With f=sigmoid(0)=0.5, A=tanh(0)=0, alpha=100+0.5=100.5
        // x_new = 0.8 * exp(-100.5 * 0.001) + 0
        let expected = 0.8 * (-100.5_f32 * 0.001).exp();
        assert_abs_diff_eq!(x_new, expected, epsilon = 1e-5);
        assert!(x_new < 0.8, "Should decay");
    }

    #[test]
    fn cfc_bounded_output() {
        let mut neuron = LtcNeuron::new();
        neuron.bias = 1000.0;
        neuron.tau = 0.01;
        let neurons = [neuron.clone()];
        let dt = 0.001;

        let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
        assert!(x_new.is_finite(), "Output should be finite");
        assert!(x_new.abs() < 2.0, "Output should be bounded");
    }

    #[test]
    fn gate_modulates_dynamics() {
        let mut fast_neuron = LtcNeuron::new();
        fast_neuron.x = 0.5;
        fast_neuron.tau = 0.01;
        fast_neuron.gate_bias = 5.0;

        let mut slow_neuron = LtcNeuron::new();
        slow_neuron.x = 0.5;
        slow_neuron.tau = 0.01;
        slow_neuron.gate_bias = -5.0;

        let neurons_fast = [fast_neuron.clone()];
        let neurons_slow = [slow_neuron.clone()];
        let dt = 0.001;

        let (x_fast, _) = cfc_forward(&fast_neuron, &neurons_fast, dt);
        let (x_slow, _) = cfc_forward(&slow_neuron, &neurons_slow, dt);

        assert!(x_fast < x_slow, "Higher gate should cause faster decay: fast={x_fast}, slow={x_slow}");
    }

    #[test]
    fn connections_contribute_to_input() {
        let n0 = LtcNeuron {
            x: 0.5, tau: 0.01, bias: 0.0, gate_bias: 0.0,
            connections: vec![], gate_weights: vec![],
            circuit_membership_count: 0,
        };
        let n1 = LtcNeuron {
            x: 0.0, tau: 0.01, bias: 0.0, gate_bias: 0.0,
            connections: vec![(0, 1.0)], gate_weights: vec![0.0],
            circuit_membership_count: 0,
        };
        let neurons = [n0, n1.clone()];
        let dt = 0.001;

        let (x_new, inter) = cfc_forward(&n1, &neurons, dt);
        assert!(x_new > 0.0, "Should be driven positive by input");
        assert_abs_diff_eq!(inter.a, 0.5_f32.tanh(), epsilon = 1e-5);
    }

    #[test]
    fn neuron_converges_to_steady_state() {
        let mut neuron = LtcNeuron::new();
        neuron.bias = 0.3;
        neuron.tau = 0.01;
        let dt = 0.001;

        for _ in 0..2000 {
            let neurons = [neuron.clone()];
            let (x_new, _) = cfc_forward(&neuron, &neurons, dt);
            neuron.x = x_new;
        }

        let f = sigmoid(0.0);
        let a = 0.3_f32.tanh();
        let alpha = 100.0 + f;
        let expected = f * a / alpha;
        assert_abs_diff_eq!(neuron.x, expected, epsilon = 1e-3);
    }

    #[test]
    fn inference_matches_full_forward() {
        let mut neuron = LtcNeuron::new();
        neuron.x = 0.3;
        neuron.bias = 0.5;
        neuron.tau = 0.02;
        neuron.gate_bias = 0.1;
        neuron.connections = vec![(0, 0.3)];
        neuron.gate_weights = vec![0.2];

        let neurons = [neuron.clone()];
        let dt = 0.001;

        let (x_full, _) = cfc_forward(&neuron, &neurons, dt);
        let x_inf = cfc_forward_inference(&neuron, &neurons, dt);
        assert_abs_diff_eq!(x_full, x_inf, epsilon = 1e-9);
    }

    #[test]
    fn gate_weights_parallel_to_connections() {
        let neuron = LtcNeuron::with_connections(0.1, vec![(0, 0.5), (1, -0.3)]);
        assert_eq!(neuron.gate_weights.len(), neuron.connections.len());
    }
}

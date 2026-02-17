//! Liquid neuron implementation.
//!
//! Each neuron follows continuous-time dynamics:
//!   dx/dt = (-x + tanh(Σ(w_i * x_i) + bias)) / tau

use serde::{Deserialize, Serialize};

/// A single neuron in the liquid neural mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidNeuron {
    /// Current activation state (approximately -1.0 to 1.0).
    pub x: f32,

    /// Bias term.
    pub bias: f32,

    /// Incoming connections: (source_neuron_index, weight).
    /// Sparse representation — only non-zero connections stored.
    pub connections: Vec<(usize, f32)>,

    /// Number of circuits this neuron currently belongs to (enforced max: 2).
    pub circuit_membership_count: u8,
}

impl LiquidNeuron {
    /// Create a new neuron with zero state and no connections.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            bias: 0.0,
            connections: Vec::new(),
            circuit_membership_count: 0,
        }
    }

    /// Create a neuron with a given bias and connections.
    pub fn with_connections(bias: f32, connections: Vec<(usize, f32)>) -> Self {
        Self {
            x: 0.0,
            bias,
            connections,
            circuit_membership_count: 0,
        }
    }
}

impl Default for LiquidNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the proposed state delta for a single neuron.
///
/// Implements: dx = (-x + tanh(Σ(w_i * x_i) + bias)) / tau * dt
///
/// Returns the delta (not the new state). The caller is responsible for
/// applying it, potentially after merging with other circuit proposals.
pub fn compute_neuron_update(
    neuron: &LiquidNeuron,
    all_neurons: &[LiquidNeuron],
    tau: f32,
    dt: f32,
) -> f32 {
    let input_sum: f32 = neuron
        .connections
        .iter()
        .map(|&(src_idx, weight)| all_neurons[src_idx].x * weight)
        .sum::<f32>()
        + neuron.bias;

    let dx = (-neuron.x + input_sum.tanh()) / tau;
    dx * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn zero_input_decays_to_zero() {
        // A neuron with no connections and no bias should decay towards 0.
        let mut neuron = LiquidNeuron::new();
        neuron.x = 0.8;
        let neurons = [neuron.clone()];
        let tau = 0.01;
        let dt = 0.001;

        let delta = compute_neuron_update(&neuron, &neurons, tau, dt);
        // dx/dt = (-0.8 + tanh(0)) / 0.01 = (-0.8 + 0) / 0.01 = -80
        // delta = -80 * 0.001 = -0.08
        assert_abs_diff_eq!(delta, -0.08, epsilon = 1e-5);
        // Negative delta means decaying toward zero — correct.
    }

    #[test]
    fn tanh_bounds_output() {
        // Even with very large input, tanh saturates at ±1,
        // so the update should be bounded.
        let mut neuron = LiquidNeuron::new();
        neuron.bias = 1000.0; // huge bias
        let neurons = [neuron.clone()];
        let tau = 0.01;
        let dt = 0.001;

        let delta = compute_neuron_update(&neuron, &neurons, tau, dt);
        // dx/dt = (-0 + tanh(1000)) / 0.01 = 1.0 / 0.01 = 100
        // delta = 100 * 0.001 = 0.1
        assert_abs_diff_eq!(delta, 0.1, epsilon = 1e-5);
    }

    #[test]
    fn connections_contribute_to_input() {
        let n0 = LiquidNeuron {
            x: 0.5,
            bias: 0.0,
            connections: vec![],
            circuit_membership_count: 0,
        };
        let n1 = LiquidNeuron {
            x: 0.0,
            bias: 0.0,
            connections: vec![(0, 1.0)], // connected to n0 with weight 1.0
            circuit_membership_count: 0,
        };
        let neurons = [n0, n1.clone()];
        let tau = 0.01;
        let dt = 0.001;

        let delta = compute_neuron_update(&n1, &neurons, tau, dt);
        // input_sum = 0.5 * 1.0 + 0.0 = 0.5
        // dx/dt = (-0 + tanh(0.5)) / 0.01 = tanh(0.5) / 0.01
        let expected = 0.5_f32.tanh() / 0.01 * 0.001;
        assert_abs_diff_eq!(delta, expected, epsilon = 1e-5);
    }

    #[test]
    fn neuron_converges_to_steady_state() {
        // Run many steps; neuron should converge to tanh(bias) for
        // a neuron with no connections.
        let mut neuron = LiquidNeuron::new();
        neuron.bias = 0.3;
        let tau = 0.01;
        let dt = 0.001;

        for _ in 0..1000 {
            let neurons = [neuron.clone()];
            let delta = compute_neuron_update(&neuron, &neurons, tau, dt);
            neuron.x += delta;
        }

        // Steady state: x = tanh(bias) = tanh(0.3)
        assert_abs_diff_eq!(neuron.x, 0.3_f32.tanh(), epsilon = 1e-3);
    }
}

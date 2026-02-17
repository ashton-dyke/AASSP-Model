//! Adam optimizer with per-parameter momentum and RMS tracking.

use crate::mesh::OverlappingMesh;
use crate::training::state::GradientAccumulator;
use serde::{Deserialize, Serialize};

/// Per-parameter Adam state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamState {
    /// First moment (mean of gradients).
    pub m: Vec<f32>,
    /// Second moment (mean of squared gradients).
    pub v: Vec<f32>,
}

impl AdamState {
    pub fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }
}

/// Compute a single Adam parameter update delta (free function).
fn adam_delta(
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: &mut f32,
    v: &mut f32,
    grad: f32,
    bc1: f32,
    bc2: f32,
) -> f32 {
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;

    let m_hat = *m / bc1;
    let v_hat = *v / bc2;

    lr * m_hat / (v_hat.sqrt() + epsilon)
}

/// Adam optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: u64,

    /// State for neuron biases.
    pub bias_state: AdamState,
    /// State for neuron connection weights (flattened).
    pub weight_state: AdamState,
    /// State for gate parameters (flattened across all zones).
    pub gate_state: AdamState,
    /// State for overlap masks (flattened across all zones).
    pub mask_state: AdamState,
}

impl AdamOptimizer {
    pub fn new(lr: f32, neuron_count: usize, total_weights: usize, gate_params: usize, mask_params: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            bias_state: AdamState::new(neuron_count),
            weight_state: AdamState::new(total_weights),
            gate_state: AdamState::new(gate_params),
            mask_state: AdamState::new(mask_params),
        }
    }

    /// Build an optimizer sized for a mesh.
    pub fn from_mesh(mesh: &OverlappingMesh, lr: f32) -> Self {
        let neuron_count = mesh.neurons.len();
        let total_weights: usize = mesh.neurons.iter().map(|n| n.connections.len()).sum();
        let gate_params = mesh.overlap_zones.len() * 74;
        let mask_params: usize = mesh.overlap_zones.iter().map(|z| z.masks.len()).sum();

        Self::new(lr, neuron_count, total_weights, gate_params, mask_params)
    }

    /// Apply one Adam update step to the mesh using accumulated gradients.
    ///
    /// `frozen_circuits`: set of circuit IDs whose neurons should not be updated.
    /// `freeze_overlaps`: if true, don't update gate/mask parameters.
    pub fn step(
        &mut self,
        mesh: &mut OverlappingMesh,
        grads: &GradientAccumulator,
        frozen_circuits: &[u32],
        freeze_overlaps: bool,
    ) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;

        // Build set of frozen neuron indices.
        let mut frozen_neurons = vec![false; mesh.neurons.len()];
        for circuit in &mesh.circuits {
            if frozen_circuits.contains(&circuit.id) || circuit.frozen {
                for &idx in &circuit.neuron_indices {
                    frozen_neurons[idx] = true;
                }
            }
        }

        // Update neuron biases.
        for (i, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[i] {
                continue;
            }
            let g = grads.bias_grads[i];
            neuron.bias -= adam_delta(
                lr, beta1, beta2, epsilon,
                &mut self.bias_state.m[i],
                &mut self.bias_state.v[i],
                g, bc1, bc2,
            );
        }

        // Update neuron connection weights.
        let mut weight_offset = 0;
        for (neuron_idx, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[neuron_idx] {
                weight_offset += neuron.connections.len();
                continue;
            }
            for (conn_idx, (_, weight)) in neuron.connections.iter_mut().enumerate() {
                let flat_idx = weight_offset + conn_idx;
                if conn_idx < grads.weight_grads[neuron_idx].len()
                    && flat_idx < self.weight_state.m.len()
                {
                    let g = grads.weight_grads[neuron_idx][conn_idx];
                    *weight -= adam_delta(
                        lr, beta1, beta2, epsilon,
                        &mut self.weight_state.m[flat_idx],
                        &mut self.weight_state.v[flat_idx],
                        g, bc1, bc2,
                    );
                }
            }
            weight_offset += neuron.connections.len();
        }

        if freeze_overlaps {
            return;
        }

        // Update overlap gate parameters.
        let mut gate_offset = 0;
        for (zone_idx, zone) in mesh.overlap_zones.iter_mut().enumerate() {
            if zone_idx >= grads.gate_grads.len() {
                break;
            }
            let gg = &grads.gate_grads[zone_idx];

            for i in 0..8 {
                for j in 0..6 {
                    if gate_offset < self.gate_state.m.len() {
                        zone.gate.weights_ih[i][j] -= adam_delta(
                            lr, beta1, beta2, epsilon,
                            &mut self.gate_state.m[gate_offset],
                            &mut self.gate_state.v[gate_offset],
                            gg.d_weights_ih[i][j], bc1, bc2,
                        );
                    }
                    gate_offset += 1;
                }
            }
            for i in 0..8 {
                if gate_offset < self.gate_state.m.len() {
                    zone.gate.bias_h[i] -= adam_delta(
                        lr, beta1, beta2, epsilon,
                        &mut self.gate_state.m[gate_offset],
                        &mut self.gate_state.v[gate_offset],
                        gg.d_bias_h[i], bc1, bc2,
                    );
                }
                gate_offset += 1;
            }
            for i in 0..2 {
                for j in 0..8 {
                    if gate_offset < self.gate_state.m.len() {
                        zone.gate.weights_ho[i][j] -= adam_delta(
                            lr, beta1, beta2, epsilon,
                            &mut self.gate_state.m[gate_offset],
                            &mut self.gate_state.v[gate_offset],
                            gg.d_weights_ho[i][j], bc1, bc2,
                        );
                    }
                    gate_offset += 1;
                }
            }
            for i in 0..2 {
                if gate_offset < self.gate_state.m.len() {
                    zone.gate.bias_o[i] -= adam_delta(
                        lr, beta1, beta2, epsilon,
                        &mut self.gate_state.m[gate_offset],
                        &mut self.gate_state.v[gate_offset],
                        gg.d_bias_o[i], bc1, bc2,
                    );
                }
                gate_offset += 1;
            }
        }

        // Update overlap masks.
        let mut mask_offset = 0;
        for (zone_idx, zone) in mesh.overlap_zones.iter_mut().enumerate() {
            if zone_idx >= grads.mask_grads.len() {
                break;
            }
            for (i, mask) in zone.masks.iter_mut().enumerate() {
                if i < grads.mask_grads[zone_idx].len() {
                    let flat_idx = mask_offset + i;
                    if flat_idx < self.mask_state.m.len() {
                        *mask -= adam_delta(
                            lr, beta1, beta2, epsilon,
                            &mut self.mask_state.m[flat_idx],
                            &mut self.mask_state.v[flat_idx],
                            grads.mask_grads[zone_idx][i], bc1, bc2,
                        );
                        *mask = mask.clamp(0.0, 1.0);
                    }
                }
            }
            mask_offset += zone.masks.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_delta_basic() {
        let mut m = 0.0;
        let mut v = 0.0;
        let bc1 = 1.0 - 0.9;
        let bc2 = 1.0 - 0.999;

        let delta = adam_delta(0.001, 0.9, 0.999, 1e-8, &mut m, &mut v, 1.0, bc1, bc2);
        assert!((delta - 0.001).abs() < 1e-6);
    }

    #[test]
    fn from_mesh_sizes() {
        use crate::config::MeshConfig;
        use crate::mesh::OverlappingMesh;

        let config = MeshConfig {
            total_neurons: 10,
            ..MeshConfig::default()
        };
        let mesh = OverlappingMesh::new(config);
        let opt = AdamOptimizer::from_mesh(&mesh, 1e-3);

        assert_eq!(opt.bias_state.m.len(), 10);
    }
}

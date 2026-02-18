//! Adam optimizer with per-parameter momentum and RMS tracking.
//!
//! Handles all CfC neuron parameters: drive weights, gate weights,
//! drive bias, gate bias, tau, and overlap masks.

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

/// Compute learning rate with linear warmup and cosine decay.
///
/// During the first `warmup_steps`, LR ramps linearly from 0 to `base_lr`.
/// After warmup, LR decays via cosine schedule to 10% of `base_lr`.
pub fn cosine_lr(base_lr: f32, step: u32, total_steps: u32, warmup_steps: u32) -> f32 {
    if total_steps == 0 {
        return base_lr;
    }
    if step < warmup_steps {
        base_lr * (step + 1) as f32 / warmup_steps.max(1) as f32
    } else {
        let min_lr = base_lr * 0.1;
        let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
        let progress = (step - warmup_steps) as f32 / decay_steps as f32;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// Maintains separate Adam states for: drive biases, drive weights (flattened),
/// gate weights (flattened), gate biases, tau values, and overlap masks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub t: u64,

    /// State for neuron drive biases.
    pub bias_state: AdamState,
    /// State for neuron drive connection weights (flattened).
    pub weight_state: AdamState,
    /// State for neuron gate weights (flattened, parallel to drive weights).
    pub gate_weight_state: AdamState,
    /// State for neuron gate biases.
    pub gate_bias_state: AdamState,
    /// State for neuron tau values.
    pub tau_state: AdamState,
    /// State for overlap masks (flattened across all zones).
    pub mask_state: AdamState,
}

impl AdamOptimizer {
    pub fn new(
        lr: f32,
        neuron_count: usize,
        total_weights: usize,
        mask_params: usize,
    ) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            t: 0,
            bias_state: AdamState::new(neuron_count),
            weight_state: AdamState::new(total_weights),
            gate_weight_state: AdamState::new(total_weights),
            gate_bias_state: AdamState::new(neuron_count),
            tau_state: AdamState::new(neuron_count),
            mask_state: AdamState::new(mask_params),
        }
    }

    /// Update the learning rate (for LR scheduling).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Build an optimizer sized for a mesh.
    pub fn from_mesh(mesh: &OverlappingMesh, lr: f32) -> Self {
        let neuron_count = mesh.neurons.len();
        let total_weights: usize = mesh.neurons.iter().map(|n| n.connections.len()).sum();
        let mask_params: usize = mesh.overlap_zones.iter().map(|z| z.masks.len()).sum();

        Self::new(lr, neuron_count, total_weights, mask_params)
    }

    /// Apply one Adam update step to the mesh using accumulated gradients.
    ///
    /// `frozen_circuits`: set of circuit IDs whose neurons should not be updated.
    /// `freeze_overlaps`: if true, don't update mask parameters.
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

        // Update neuron drive biases.
        for (i, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[i] {
                continue;
            }
            let g = grads.bias_grads[i];
            neuron.bias -= adam_delta(
                lr,
                beta1,
                beta2,
                epsilon,
                &mut self.bias_state.m[i],
                &mut self.bias_state.v[i],
                g,
                bc1,
                bc2,
            );
        }

        // Update neuron gate biases.
        for (i, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[i] {
                continue;
            }
            let g = grads.gate_bias_grads[i];
            neuron.gate_bias -= adam_delta(
                lr,
                beta1,
                beta2,
                epsilon,
                &mut self.gate_bias_state.m[i],
                &mut self.gate_bias_state.v[i],
                g,
                bc1,
                bc2,
            );
        }

        // Update neuron tau values.
        for (i, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[i] {
                continue;
            }
            let g = grads.tau_grads[i];
            neuron.tau -= adam_delta(
                lr,
                beta1,
                beta2,
                epsilon,
                &mut self.tau_state.m[i],
                &mut self.tau_state.v[i],
                g,
                bc1,
                bc2,
            );
            // Clamp tau to positive range.
            neuron.tau = neuron.tau.max(1e-4);
        }

        // Update neuron drive connection weights.
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
                    // AdamW: decoupled weight decay applied before adam step.
                    if self.weight_decay > 0.0 {
                        *weight *= 1.0 - lr * self.weight_decay;
                    }
                    *weight -= adam_delta(
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        &mut self.weight_state.m[flat_idx],
                        &mut self.weight_state.v[flat_idx],
                        g,
                        bc1,
                        bc2,
                    );
                }
            }
            weight_offset += neuron.connections.len();
        }

        // Update neuron gate weights.
        let mut gw_offset = 0;
        for (neuron_idx, neuron) in mesh.neurons.iter_mut().enumerate() {
            if frozen_neurons[neuron_idx] {
                gw_offset += neuron.gate_weights.len();
                continue;
            }
            for (conn_idx, gw) in neuron.gate_weights.iter_mut().enumerate() {
                let flat_idx = gw_offset + conn_idx;
                if conn_idx < grads.gate_weight_grads[neuron_idx].len()
                    && flat_idx < self.gate_weight_state.m.len()
                {
                    let g = grads.gate_weight_grads[neuron_idx][conn_idx];
                    // AdamW: decoupled weight decay for gate weights too.
                    if self.weight_decay > 0.0 {
                        *gw *= 1.0 - lr * self.weight_decay;
                    }
                    *gw -= adam_delta(
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        &mut self.gate_weight_state.m[flat_idx],
                        &mut self.gate_weight_state.v[flat_idx],
                        g,
                        bc1,
                        bc2,
                    );
                }
            }
            gw_offset += neuron.gate_weights.len();
        }

        if freeze_overlaps {
            return;
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
                            lr,
                            beta1,
                            beta2,
                            epsilon,
                            &mut self.mask_state.m[flat_idx],
                            &mut self.mask_state.v[flat_idx],
                            grads.mask_grads[zone_idx][i],
                            bc1,
                            bc2,
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
        assert_eq!(opt.gate_bias_state.m.len(), 10);
        assert_eq!(opt.tau_state.m.len(), 10);
    }

    #[test]
    fn cosine_lr_warmup_ramps_linearly() {
        let lr = cosine_lr(0.001, 0, 100, 10);
        assert!(
            lr > 0.0 && lr < 0.001,
            "Step 0 should be below base LR during warmup"
        );

        let lr_mid = cosine_lr(0.001, 5, 100, 10);
        assert!(lr_mid > lr, "LR should increase during warmup");

        let lr_end_warmup = cosine_lr(0.001, 9, 100, 10);
        assert!(
            (lr_end_warmup - 0.001).abs() < 1e-5,
            "LR should reach base at end of warmup"
        );
    }

    #[test]
    fn cosine_lr_decays_after_warmup() {
        let lr_start = cosine_lr(0.001, 10, 100, 10);
        let lr_mid = cosine_lr(0.001, 55, 100, 10);
        let lr_end = cosine_lr(0.001, 99, 100, 10);

        assert!(lr_start > lr_mid, "LR should decrease after warmup");
        assert!(lr_mid > lr_end, "LR should continue decreasing");
        // Should decay to ~10% of base at the end.
        assert!(lr_end >= 0.001 * 0.09, "LR should not go below min");
        assert!(lr_end <= 0.001 * 0.2, "LR should be near min at end");
    }

    #[test]
    fn cosine_lr_handles_zero_steps() {
        let lr = cosine_lr(0.001, 0, 0, 0);
        assert!(
            (lr - 0.001).abs() < 1e-9,
            "Zero total_steps should return base LR"
        );
    }

    #[test]
    fn set_lr_updates_rate() {
        let mut opt = AdamOptimizer::new(0.001, 2, 2, 0);
        assert!((opt.lr - 0.001).abs() < 1e-9);
        opt.set_lr(0.0005);
        assert!((opt.lr - 0.0005).abs() < 1e-9);
    }

    #[test]
    fn optimizer_has_gate_and_tau_states() {
        let opt = AdamOptimizer::new(0.001, 5, 10, 3);
        assert_eq!(opt.gate_weight_state.m.len(), 10);
        assert_eq!(opt.gate_bias_state.m.len(), 5);
        assert_eq!(opt.tau_state.m.len(), 5);
        assert_eq!(opt.mask_state.m.len(), 3);
    }
}

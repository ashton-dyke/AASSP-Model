//! Forward pass state recording for BPTT and gradient accumulation.
//!
//! With LTC/CfC neurons, every neuron updates synchronously at every step.
//! The ForwardRecord stores CfcIntermediate per neuron per step (no Option<>).

use crate::neuron::CfcIntermediate;

/// Records the full forward pass for BPTT.
///
/// Stores CfcIntermediate values at each mesh step so that the backward
/// pass can compute analytical gradients without recomputation.
#[derive(Debug, Clone)]
pub struct ForwardRecord {
    /// Intermediates per step: step_intermediates[step][neuron_idx].
    /// Every neuron updates every step (synchronous CfC), so no Option<>.
    pub step_intermediates: Vec<Vec<CfcIntermediate>>,
    /// Number of neurons.
    pub neuron_count: usize,
    /// Number of steps recorded.
    pub step_count: usize,
}

impl ForwardRecord {
    pub fn new(neuron_count: usize, max_steps: usize) -> Self {
        Self {
            step_intermediates: Vec::with_capacity(max_steps),
            neuron_count,
            step_count: 0,
        }
    }

    /// Start recording a new step. Pushes a default-filled row of intermediates.
    pub fn begin_step(&mut self) {
        let default_inter = CfcIntermediate {
            x_prev: 0.0,
            f: 0.5,
            a: 0.0,
            alpha: 100.0,
            exp_term: 1.0,
            drive_sum: 0.0,
            gate_sum: 0.0,
            tau: 0.01,
        };
        self.step_intermediates
            .push(vec![default_inter; self.neuron_count]);
        self.step_count += 1;
    }

    /// Record a neuron's CfcIntermediate for the current step.
    pub fn record_neuron(&mut self, neuron_idx: usize, inter: CfcIntermediate) {
        let step = self.step_count - 1;
        self.step_intermediates[step][neuron_idx] = inter;
    }
}

/// Accumulates gradients for all trainable parameters in the mesh.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// Gradient for each neuron's drive bias.
    pub bias_grads: Vec<f32>,
    /// Gradient for each neuron's connection weights (drive pathway).
    /// weight_grads[neuron_idx] = Vec of gradients, one per connection.
    pub weight_grads: Vec<Vec<f32>>,
    /// Gradient for each neuron's gate weights (parallel to connections).
    /// gate_weight_grads[neuron_idx] = Vec of gradients, one per connection.
    pub gate_weight_grads: Vec<Vec<f32>>,
    /// Gradient for each neuron's gate bias.
    pub gate_bias_grads: Vec<f32>,
    /// Gradient for each neuron's tau.
    pub tau_grads: Vec<f32>,
    /// Gradient flowing back through neuron activations (for chain rule).
    pub dx: Vec<f32>,

    /// Gradient for overlap masks: [zone_idx] -> Vec<f32>.
    pub mask_grads: Vec<Vec<f32>>,

    /// Gradient for write-gate weights: [memory_level (0=ST, 1=MT, 2=LT)].
    pub write_gate_weight_grads: Vec<Vec<f32>>,
    /// Gradient for write-gate biases.
    pub write_gate_bias_grads: Vec<f32>,
}

impl GradientAccumulator {
    /// Create a zeroed accumulator matching the mesh dimensions.
    pub fn new(
        neuron_count: usize,
        connection_counts: &[usize],
        overlap_mask_sizes: &[usize],
        write_gate_dims: &[usize],
    ) -> Self {
        Self {
            bias_grads: vec![0.0; neuron_count],
            weight_grads: connection_counts.iter().map(|&c| vec![0.0; c]).collect(),
            gate_weight_grads: connection_counts.iter().map(|&c| vec![0.0; c]).collect(),
            gate_bias_grads: vec![0.0; neuron_count],
            tau_grads: vec![0.0; neuron_count],
            dx: vec![0.0; neuron_count],
            mask_grads: overlap_mask_sizes
                .iter()
                .map(|&s| vec![0.0; s])
                .collect(),
            write_gate_weight_grads: write_gate_dims
                .iter()
                .map(|&d| vec![0.0; d])
                .collect(),
            write_gate_bias_grads: vec![0.0; write_gate_dims.len()],
        }
    }

    /// Build an accumulator from a mesh.
    pub fn from_mesh(mesh: &crate::mesh::OverlappingMesh) -> Self {
        let neuron_count = mesh.neurons.len();
        let connection_counts: Vec<usize> =
            mesh.neurons.iter().map(|n| n.connections.len()).collect();
        let overlap_mask_sizes: Vec<usize> =
            mesh.overlap_zones.iter().map(|z| z.masks.len()).collect();
        // Write gate dims: 3 memory levels, each with dim based on causation circuit size.
        // For simplicity, use 0 if no write gates are present (they are in HierarchicalMemory).
        let write_gate_dims = vec![0usize; 3];

        Self::new(
            neuron_count,
            &connection_counts,
            &overlap_mask_sizes,
            &write_gate_dims,
        )
    }

    /// Zero all gradients.
    pub fn zero(&mut self) {
        for g in &mut self.bias_grads {
            *g = 0.0;
        }
        for wg in &mut self.weight_grads {
            for g in wg {
                *g = 0.0;
            }
        }
        for gw in &mut self.gate_weight_grads {
            for g in gw {
                *g = 0.0;
            }
        }
        for g in &mut self.gate_bias_grads {
            *g = 0.0;
        }
        for g in &mut self.tau_grads {
            *g = 0.0;
        }
        for g in &mut self.dx {
            *g = 0.0;
        }
        for mg in &mut self.mask_grads {
            for g in mg {
                *g = 0.0;
            }
        }
        for wg in &mut self.write_gate_weight_grads {
            for g in wg {
                *g = 0.0;
            }
        }
        for g in &mut self.write_gate_bias_grads {
            *g = 0.0;
        }
    }

    /// Compute the global gradient norm (L2).
    pub fn global_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for &g in &self.bias_grads {
            sum += (g as f64) * (g as f64);
        }
        for wg in &self.weight_grads {
            for &g in wg {
                sum += (g as f64) * (g as f64);
            }
        }
        for gw in &self.gate_weight_grads {
            for &g in gw {
                sum += (g as f64) * (g as f64);
            }
        }
        for &g in &self.gate_bias_grads {
            sum += (g as f64) * (g as f64);
        }
        for &g in &self.tau_grads {
            sum += (g as f64) * (g as f64);
        }
        for mg in &self.mask_grads {
            for &g in mg {
                sum += (g as f64) * (g as f64);
            }
        }
        sum.sqrt() as f32
    }

    /// Add gradients from another accumulator element-wise.
    pub fn accumulate(&mut self, other: &GradientAccumulator) {
        for (a, b) in self.bias_grads.iter_mut().zip(&other.bias_grads) {
            *a += b;
        }
        for (aw, bw) in self.weight_grads.iter_mut().zip(&other.weight_grads) {
            for (a, b) in aw.iter_mut().zip(bw) {
                *a += b;
            }
        }
        for (aw, bw) in self
            .gate_weight_grads
            .iter_mut()
            .zip(&other.gate_weight_grads)
        {
            for (a, b) in aw.iter_mut().zip(bw) {
                *a += b;
            }
        }
        for (a, b) in self.gate_bias_grads.iter_mut().zip(&other.gate_bias_grads) {
            *a += b;
        }
        for (a, b) in self.tau_grads.iter_mut().zip(&other.tau_grads) {
            *a += b;
        }
        for (a, b) in self.dx.iter_mut().zip(&other.dx) {
            *a += b;
        }
        for (am, bm) in self.mask_grads.iter_mut().zip(&other.mask_grads) {
            for (a, b) in am.iter_mut().zip(bm) {
                *a += b;
            }
        }
    }

    /// Scale all gradients by a factor (e.g. 1/mini_batch_size).
    pub fn scale(&mut self, factor: f32) {
        for g in &mut self.bias_grads {
            *g *= factor;
        }
        for wg in &mut self.weight_grads {
            for g in wg {
                *g *= factor;
            }
        }
        for gw in &mut self.gate_weight_grads {
            for g in gw {
                *g *= factor;
            }
        }
        for g in &mut self.gate_bias_grads {
            *g *= factor;
        }
        for g in &mut self.tau_grads {
            *g *= factor;
        }
        for g in &mut self.dx {
            *g *= factor;
        }
        for mg in &mut self.mask_grads {
            for g in mg {
                *g *= factor;
            }
        }
    }

    /// Clip all gradients so the global norm <= max_norm.
    pub fn clip_global_norm(&mut self, max_norm: f32) {
        let norm = self.global_norm();
        if norm <= max_norm || norm < 1e-9 {
            return;
        }
        let scale = max_norm / norm;
        for g in &mut self.bias_grads {
            *g *= scale;
        }
        for wg in &mut self.weight_grads {
            for g in wg {
                *g *= scale;
            }
        }
        for gw in &mut self.gate_weight_grads {
            for g in gw {
                *g *= scale;
            }
        }
        for g in &mut self.gate_bias_grads {
            *g *= scale;
        }
        for g in &mut self.tau_grads {
            *g *= scale;
        }
        for mg in &mut self.mask_grads {
            for g in mg {
                *g *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_record_basic() {
        let mut record = ForwardRecord::new(10, 100);
        record.begin_step();

        let inter = CfcIntermediate {
            x_prev: 0.5,
            f: 0.5,
            a: 0.3,
            alpha: 100.5,
            exp_term: 0.9,
            drive_sum: 0.3,
            gate_sum: 0.0,
            tau: 0.01,
        };
        record.record_neuron(3, inter);

        assert_eq!(record.step_count, 1);
        assert!((record.step_intermediates[0][3].x_prev - 0.5).abs() < 1e-6);
        assert!((record.step_intermediates[0][3].a - 0.3).abs() < 1e-6);
    }

    #[test]
    fn forward_record_all_neurons_populated() {
        let mut record = ForwardRecord::new(5, 10);
        record.begin_step();

        // All neurons should have default intermediates (no Option<>).
        for i in 0..5 {
            // Default f = 0.5 (sigmoid(0)).
            assert!((record.step_intermediates[0][i].f - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn gradient_accumulator_zero() {
        let mut acc = GradientAccumulator::new(4, &[2, 3, 0, 1], &[2], &[3]);
        acc.bias_grads[0] = 1.0;
        acc.gate_bias_grads[1] = 2.0;
        acc.tau_grads[2] = 0.5;
        acc.zero();
        assert!((acc.bias_grads[0]).abs() < 1e-9);
        assert!((acc.gate_bias_grads[1]).abs() < 1e-9);
        assert!((acc.tau_grads[2]).abs() < 1e-9);
    }

    #[test]
    fn clip_global_norm() {
        let mut acc = GradientAccumulator::new(2, &[0, 0], &[], &[]);
        acc.bias_grads[0] = 3.0;
        acc.bias_grads[1] = 4.0;
        // norm = 5.0
        acc.clip_global_norm(1.0);
        let norm = acc.global_norm();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn accumulate_adds_element_wise() {
        let mut a = GradientAccumulator::new(3, &[2, 1, 0], &[2], &[]);
        let mut b = GradientAccumulator::new(3, &[2, 1, 0], &[2], &[]);

        a.bias_grads[0] = 1.0;
        a.bias_grads[1] = 2.0;
        a.weight_grads[0][0] = 0.5;
        a.mask_grads[0][0] = 0.3;
        a.gate_bias_grads[0] = 0.1;
        a.tau_grads[1] = 0.2;

        b.bias_grads[0] = 3.0;
        b.bias_grads[1] = 4.0;
        b.weight_grads[0][0] = 1.5;
        b.mask_grads[0][0] = 0.7;
        b.gate_bias_grads[0] = 0.4;
        b.tau_grads[1] = 0.3;

        a.accumulate(&b);

        assert!((a.bias_grads[0] - 4.0).abs() < 1e-9);
        assert!((a.bias_grads[1] - 6.0).abs() < 1e-9);
        assert!((a.weight_grads[0][0] - 2.0).abs() < 1e-9);
        assert!((a.mask_grads[0][0] - 1.0).abs() < 1e-9);
        assert!((a.gate_bias_grads[0] - 0.5).abs() < 1e-9);
        assert!((a.tau_grads[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn scale_divides_all_grads() {
        let mut acc = GradientAccumulator::new(2, &[1, 1], &[], &[]);
        acc.bias_grads[0] = 4.0;
        acc.bias_grads[1] = 8.0;
        acc.weight_grads[0][0] = 2.0;
        acc.gate_bias_grads[0] = 1.0;
        acc.tau_grads[1] = 0.8;

        acc.scale(0.25);

        assert!((acc.bias_grads[0] - 1.0).abs() < 1e-9);
        assert!((acc.bias_grads[1] - 2.0).abs() < 1e-9);
        assert!((acc.weight_grads[0][0] - 0.5).abs() < 1e-9);
        assert!((acc.gate_bias_grads[0] - 0.25).abs() < 1e-9);
        assert!((acc.tau_grads[1] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn global_norm_includes_all_param_types() {
        let mut acc = GradientAccumulator::new(2, &[1, 1], &[1], &[]);
        acc.bias_grads[0] = 1.0;
        acc.gate_bias_grads[0] = 1.0;
        acc.tau_grads[0] = 1.0;
        acc.weight_grads[0][0] = 1.0;
        acc.gate_weight_grads[0][0] = 1.0;
        acc.mask_grads[0][0] = 1.0;

        let norm = acc.global_norm();
        // sqrt(6) = 2.449...
        assert!((norm - 6.0_f32.sqrt()).abs() < 1e-5);
    }
}

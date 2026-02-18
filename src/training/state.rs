//! Forward pass state recording for BPTT and gradient accumulation.

use crate::circuit::CircuitId;

/// Recorded state of one neuron at one timestep.
#[derive(Debug, Clone)]
pub struct NeuronSnapshot {
    /// Activation before update.
    pub x: f32,
    /// Pre-activation sum: Σ(w_i * x_i) + bias.
    pub pre_activation: f32,
    /// tanh(pre_activation).
    pub tanh_val: f32,
    /// Which circuit(s) updated this neuron and the tau used.
    pub circuit_updates: Vec<(CircuitId, f32)>, // (circuit_id, tau)
}

/// Records the full forward pass for BPTT.
///
/// Stores intermediate neuron states at each mesh step so that the backward
/// pass can compute gradients without recomputation.
#[derive(Debug, Clone)]
pub struct ForwardRecord {
    /// Snapshots per step: step_snapshots[step][neuron_idx].
    /// Only neurons that were actually updated have meaningful data.
    pub step_snapshots: Vec<Vec<Option<NeuronSnapshot>>>,
    /// Which circuits fired at each step.
    pub firing_record: Vec<Vec<CircuitId>>,
    /// Number of neurons.
    pub neuron_count: usize,
    /// Number of steps recorded.
    pub step_count: usize,
}

impl ForwardRecord {
    pub fn new(neuron_count: usize, max_steps: usize) -> Self {
        Self {
            step_snapshots: Vec::with_capacity(max_steps),
            firing_record: Vec::with_capacity(max_steps),
            neuron_count,
            step_count: 0,
        }
    }

    /// Start recording a new step.
    pub fn begin_step(&mut self) {
        self.step_snapshots.push(vec![None; self.neuron_count]);
        self.firing_record.push(Vec::new());
        self.step_count += 1;
    }

    /// Record that a circuit fired this step.
    pub fn record_firing(&mut self, circuit_id: CircuitId) {
        if let Some(last) = self.firing_record.last_mut() {
            last.push(circuit_id);
        }
    }

    /// Record a neuron's pre-update state.
    pub fn record_neuron(
        &mut self,
        neuron_idx: usize,
        x: f32,
        pre_activation: f32,
        circuit_id: CircuitId,
        tau: f32,
    ) {
        let step = self.step_count - 1;
        let tanh_val = pre_activation.tanh();

        if let Some(existing) = &mut self.step_snapshots[step][neuron_idx] {
            // Neuron updated by multiple circuits this step (overlap neuron).
            existing.circuit_updates.push((circuit_id, tau));
        } else {
            self.step_snapshots[step][neuron_idx] = Some(NeuronSnapshot {
                x,
                pre_activation,
                tanh_val,
                circuit_updates: vec![(circuit_id, tau)],
            });
        }
    }
}

/// Accumulates gradients for all trainable parameters in the mesh.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// Gradient for each neuron's bias.
    pub bias_grads: Vec<f32>,
    /// Gradient for each neuron's connection weights.
    /// weight_grads[neuron_idx] = Vec of gradients, one per connection.
    pub weight_grads: Vec<Vec<f32>>,
    /// Gradient flowing back through neuron activations (for chain rule).
    pub dx: Vec<f32>,

    /// Gradient for overlap gate weights: [zone_idx] -> GateGradient.
    pub gate_grads: Vec<GateGradient>,
    /// Gradient for overlap masks: [zone_idx] -> Vec<f32>.
    pub mask_grads: Vec<Vec<f32>>,

    /// Gradient for write-gate weights: [memory_level (0=ST, 1=MT, 2=LT)].
    pub write_gate_weight_grads: Vec<Vec<f32>>,
    /// Gradient for write-gate biases.
    pub write_gate_bias_grads: Vec<f32>,
}

/// Gradients for an overlap gate MLP (6→8→2).
#[derive(Debug, Clone)]
pub struct GateGradient {
    pub d_weights_ih: [[f32; 6]; 8],
    pub d_bias_h: [f32; 8],
    pub d_weights_ho: [[f32; 8]; 2],
    pub d_bias_o: [f32; 2],
}

impl GateGradient {
    pub fn zeros() -> Self {
        Self {
            d_weights_ih: [[0.0; 6]; 8],
            d_bias_h: [0.0; 8],
            d_weights_ho: [[0.0; 8]; 2],
            d_bias_o: [0.0; 2],
        }
    }
}

impl GradientAccumulator {
    /// Create a zeroed accumulator matching the mesh dimensions.
    pub fn new(
        neuron_count: usize,
        connection_counts: &[usize],
        overlap_zone_count: usize,
        overlap_mask_sizes: &[usize],
        write_gate_dims: &[usize],
    ) -> Self {
        Self {
            bias_grads: vec![0.0; neuron_count],
            weight_grads: connection_counts.iter().map(|&c| vec![0.0; c]).collect(),
            dx: vec![0.0; neuron_count],
            gate_grads: (0..overlap_zone_count)
                .map(|_| GateGradient::zeros())
                .collect(),
            mask_grads: overlap_mask_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            write_gate_weight_grads: write_gate_dims.iter().map(|&d| vec![0.0; d]).collect(),
            write_gate_bias_grads: vec![0.0; write_gate_dims.len()],
        }
    }

    /// Build an accumulator from a mesh.
    pub fn from_mesh(mesh: &crate::mesh::OverlappingMesh) -> Self {
        let neuron_count = mesh.neurons.len();
        let connection_counts: Vec<usize> =
            mesh.neurons.iter().map(|n| n.connections.len()).collect();
        let overlap_zone_count = mesh.overlap_zones.len();
        let overlap_mask_sizes: Vec<usize> =
            mesh.overlap_zones.iter().map(|z| z.masks.len()).collect();
        // Write gate dims: 3 memory levels, each with dim based on causation circuit size.
        // For simplicity, use 0 if no write gates are present (they are in HierarchicalMemory).
        let write_gate_dims = vec![0usize; 3];

        Self::new(
            neuron_count,
            &connection_counts,
            overlap_zone_count,
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
        for g in &mut self.dx {
            *g = 0.0;
        }
        for gg in &mut self.gate_grads {
            gg.d_weights_ih = [[0.0; 6]; 8];
            gg.d_bias_h = [0.0; 8];
            gg.d_weights_ho = [[0.0; 8]; 2];
            gg.d_bias_o = [0.0; 2];
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
        for gg in &self.gate_grads {
            for row in &gg.d_weights_ih {
                for &g in row {
                    sum += (g as f64) * (g as f64);
                }
            }
            for &g in &gg.d_bias_h {
                sum += (g as f64) * (g as f64);
            }
            for row in &gg.d_weights_ho {
                for &g in row {
                    sum += (g as f64) * (g as f64);
                }
            }
            for &g in &gg.d_bias_o {
                sum += (g as f64) * (g as f64);
            }
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
        for (a, b) in self.dx.iter_mut().zip(&other.dx) {
            *a += b;
        }
        for (ag, bg) in self.gate_grads.iter_mut().zip(&other.gate_grads) {
            for (ar, br) in ag.d_weights_ih.iter_mut().zip(&bg.d_weights_ih) {
                for (a, b) in ar.iter_mut().zip(br) {
                    *a += b;
                }
            }
            for (a, b) in ag.d_bias_h.iter_mut().zip(&bg.d_bias_h) {
                *a += b;
            }
            for (ar, br) in ag.d_weights_ho.iter_mut().zip(&bg.d_weights_ho) {
                for (a, b) in ar.iter_mut().zip(br) {
                    *a += b;
                }
            }
            for (a, b) in ag.d_bias_o.iter_mut().zip(&bg.d_bias_o) {
                *a += b;
            }
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
        for g in &mut self.dx {
            *g *= factor;
        }
        for gg in &mut self.gate_grads {
            for row in &mut gg.d_weights_ih {
                for g in row {
                    *g *= factor;
                }
            }
            for g in &mut gg.d_bias_h {
                *g *= factor;
            }
            for row in &mut gg.d_weights_ho {
                for g in row {
                    *g *= factor;
                }
            }
            for g in &mut gg.d_bias_o {
                *g *= factor;
            }
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
        for gg in &mut self.gate_grads {
            for row in &mut gg.d_weights_ih {
                for g in row {
                    *g *= scale;
                }
            }
            for g in &mut gg.d_bias_h {
                *g *= scale;
            }
            for row in &mut gg.d_weights_ho {
                for g in row {
                    *g *= scale;
                }
            }
            for g in &mut gg.d_bias_o {
                *g *= scale;
            }
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
        record.record_firing(0);
        record.record_neuron(3, 0.5, 0.2, 0, 0.01);

        assert_eq!(record.step_count, 1);
        assert!(record.step_snapshots[0][3].is_some());
        let snap = record.step_snapshots[0][3].as_ref().unwrap();
        assert!((snap.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gradient_accumulator_zero() {
        let mut acc = GradientAccumulator::new(4, &[2, 3, 0, 1], 1, &[2], &[3]);
        acc.bias_grads[0] = 1.0;
        acc.zero();
        assert!((acc.bias_grads[0]).abs() < 1e-9);
    }

    #[test]
    fn clip_global_norm() {
        let mut acc = GradientAccumulator::new(2, &[0, 0], 0, &[], &[]);
        acc.bias_grads[0] = 3.0;
        acc.bias_grads[1] = 4.0;
        // norm = 5.0
        acc.clip_global_norm(1.0);
        let norm = acc.global_norm();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn accumulate_adds_element_wise() {
        let mut a = GradientAccumulator::new(3, &[2, 1, 0], 1, &[2], &[]);
        let mut b = GradientAccumulator::new(3, &[2, 1, 0], 1, &[2], &[]);

        a.bias_grads[0] = 1.0;
        a.bias_grads[1] = 2.0;
        a.weight_grads[0][0] = 0.5;
        a.mask_grads[0][0] = 0.3;

        b.bias_grads[0] = 3.0;
        b.bias_grads[1] = 4.0;
        b.weight_grads[0][0] = 1.5;
        b.mask_grads[0][0] = 0.7;

        a.accumulate(&b);

        assert!((a.bias_grads[0] - 4.0).abs() < 1e-9);
        assert!((a.bias_grads[1] - 6.0).abs() < 1e-9);
        assert!((a.weight_grads[0][0] - 2.0).abs() < 1e-9);
        assert!((a.mask_grads[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scale_divides_all_grads() {
        let mut acc = GradientAccumulator::new(2, &[1, 1], 0, &[], &[]);
        acc.bias_grads[0] = 4.0;
        acc.bias_grads[1] = 8.0;
        acc.weight_grads[0][0] = 2.0;

        acc.scale(0.25);

        assert!((acc.bias_grads[0] - 1.0).abs() < 1e-9);
        assert!((acc.bias_grads[1] - 2.0).abs() < 1e-9);
        assert!((acc.weight_grads[0][0] - 0.5).abs() < 1e-9);
    }
}

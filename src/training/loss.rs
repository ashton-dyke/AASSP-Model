//! Loss functions for training the AASSP neural mesh.

use crate::circuit::CircuitType;
use crate::mesh::OverlappingMesh;

const EPS: f32 = 1e-7;
const READOUT_NEURONS: usize = 8;

/// Sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Get the readout value for a detection circuit: sigmoid of mean of last N neurons.
pub fn detection_readout(mesh: &OverlappingMesh, circuit_idx: usize) -> f32 {
    let circuit = &mesh.circuits[circuit_idx];
    let indices = &circuit.neuron_indices;
    let n = indices.len();
    if n == 0 {
        return 0.0;
    }

    let start = if n > READOUT_NEURONS { n - READOUT_NEURONS } else { 0 };
    let readout_indices = &indices[start..];
    let mean: f32 = readout_indices
        .iter()
        .map(|&idx| mesh.neurons[idx].x)
        .sum::<f32>()
        / readout_indices.len() as f32;

    sigmoid(mean)
}

/// Binary cross-entropy loss for a single detection circuit.
///
/// `y_pred`: circuit readout (0..1).
/// `y_true`: 1.0 if anomaly present, 0.0 otherwise.
pub fn binary_cross_entropy(y_pred: f32, y_true: f32) -> f32 {
    let p = y_pred.clamp(EPS, 1.0 - EPS);
    -(y_true * p.ln() + (1.0 - y_true) * (1.0 - p).ln())
}

/// Gradient of binary cross-entropy w.r.t. y_pred.
pub fn binary_cross_entropy_grad(y_pred: f32, y_true: f32) -> f32 {
    let p = y_pred.clamp(EPS, 1.0 - EPS);
    -(y_true / p) + (1.0 - y_true) / (1.0 - p)
}

/// Compute total detection loss across all detection circuits.
///
/// `labels`: map from circuit index to target (1.0 = anomaly, 0.0 = normal).
/// Returns (total_loss, Vec of ∂L/∂x for each neuron).
pub fn detection_loss(
    mesh: &OverlappingMesh,
    labels: &[(usize, f32)],
) -> (f32, Vec<f32>) {
    let mut total_loss = 0.0;
    let mut dl_dx = vec![0.0f32; mesh.neurons.len()];

    for &(circuit_idx, y_true) in labels {
        if circuit_idx >= mesh.circuits.len() {
            continue;
        }
        let circuit = &mesh.circuits[circuit_idx];
        if circuit.circuit_type != CircuitType::Detection {
            continue;
        }

        let y_pred = detection_readout(mesh, circuit_idx);
        total_loss += binary_cross_entropy(y_pred, y_true);

        // ∂L/∂y_pred
        let dl_dy = binary_cross_entropy_grad(y_pred, y_true);

        // ∂y_pred/∂mean = sigmoid'(mean) = y_pred * (1 - y_pred)
        let sigmoid_deriv = y_pred * (1.0 - y_pred);
        let dl_d_mean = dl_dy * sigmoid_deriv;

        // ∂mean/∂x_i = 1/N for each readout neuron
        let indices = &circuit.neuron_indices;
        let n = indices.len();
        let start = if n > READOUT_NEURONS { n - READOUT_NEURONS } else { 0 };
        let readout_indices = &indices[start..];
        let n_readout = readout_indices.len() as f32;

        for &idx in readout_indices {
            dl_dx[idx] += dl_d_mean / n_readout;
        }
    }

    (total_loss, dl_dx)
}

/// Compute causation loss (multi-label BCE over readout neurons).
///
/// Each causation circuit's last 8 neurons map to causal parameters.
/// `labels`: map from circuit index to Vec of (readout_position, target).
pub fn causation_loss(
    mesh: &OverlappingMesh,
    labels: &[(usize, Vec<(usize, f32)>)],
) -> (f32, Vec<f32>) {
    let mut total_loss = 0.0;
    let mut dl_dx = vec![0.0f32; mesh.neurons.len()];

    for (circuit_idx, param_labels) in labels {
        if *circuit_idx >= mesh.circuits.len() {
            continue;
        }
        let circuit = &mesh.circuits[*circuit_idx];
        if circuit.circuit_type != CircuitType::Causation {
            continue;
        }

        let indices = &circuit.neuron_indices;
        let n = indices.len();
        let start = if n > READOUT_NEURONS { n - READOUT_NEURONS } else { 0 };
        let readout_indices = &indices[start..];

        for &(pos, y_true) in param_labels {
            if pos >= readout_indices.len() {
                continue;
            }
            let neuron_idx = readout_indices[pos];
            let y_pred = sigmoid(mesh.neurons[neuron_idx].x);
            total_loss += binary_cross_entropy(y_pred, y_true);

            let dl_dy = binary_cross_entropy_grad(y_pred, y_true);
            let sigmoid_deriv = y_pred * (1.0 - y_pred);
            dl_dx[neuron_idx] += dl_dy * sigmoid_deriv;
        }
    }

    (total_loss, dl_dx)
}

/// Compute prediction loss (MSE between readout neurons and future WITS values).
///
/// `labels`: map from circuit index to Vec of (readout_position, target_value).
pub fn prediction_loss(
    mesh: &OverlappingMesh,
    labels: &[(usize, Vec<(usize, f32)>)],
) -> (f32, Vec<f32>) {
    let mut total_loss = 0.0;
    let mut dl_dx = vec![0.0f32; mesh.neurons.len()];
    let mut count = 0;

    for (circuit_idx, param_targets) in labels {
        if *circuit_idx >= mesh.circuits.len() {
            continue;
        }
        let circuit = &mesh.circuits[*circuit_idx];
        if circuit.circuit_type != CircuitType::Prediction {
            continue;
        }

        let indices = &circuit.neuron_indices;
        let n = indices.len();
        // Use last 14 neurons for prediction readout (one per WITS feature).
        let readout_size = 14.min(n);
        let start = n - readout_size;
        let readout_indices = &indices[start..];

        for &(pos, y_true) in param_targets {
            if pos >= readout_indices.len() {
                continue;
            }
            let neuron_idx = readout_indices[pos];
            let y_pred = mesh.neurons[neuron_idx].x;
            let diff = y_pred - y_true;
            total_loss += diff * diff;
            dl_dx[neuron_idx] += 2.0 * diff;
            count += 1;
        }
    }

    if count > 0 {
        total_loss /= count as f32;
        for g in &mut dl_dx {
            *g /= count as f32;
        }
    }

    (total_loss, dl_dx)
}

/// Compute mutual information approximation loss for overlap pre-training.
///
/// Encourages shared neurons to correlate with both circuits' private outputs.
/// Returns (loss, ∂L/∂x for each neuron).
pub fn mutual_information_loss(
    mesh: &OverlappingMesh,
    zone_idx: usize,
) -> (f32, Vec<f32>) {
    let mut dl_dx = vec![0.0f32; mesh.neurons.len()];

    if zone_idx >= mesh.overlap_zones.len() {
        return (0.0, dl_dx);
    }

    let zone = &mesh.overlap_zones[zone_idx];

    // Get circuit indices.
    let circuit_a = mesh.circuits.iter().find(|c| c.id == zone.circuit_a);
    let circuit_b = mesh.circuits.iter().find(|c| c.id == zone.circuit_b);

    let (circuit_a, circuit_b) = match (circuit_a, circuit_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return (0.0, dl_dx),
    };

    // Compute mean activations for private regions (non-shared neurons).
    let shared_set: std::collections::HashSet<usize> =
        zone.shared_neuron_indices.iter().cloned().collect();

    let private_a: Vec<f32> = circuit_a
        .neuron_indices
        .iter()
        .filter(|idx| !shared_set.contains(idx))
        .map(|&idx| mesh.neurons[idx].x)
        .collect();

    let private_b: Vec<f32> = circuit_b
        .neuron_indices
        .iter()
        .filter(|idx| !shared_set.contains(idx))
        .map(|&idx| mesh.neurons[idx].x)
        .collect();

    if private_a.is_empty() || private_b.is_empty() || zone.shared_neuron_indices.is_empty() {
        return (0.0, dl_dx);
    }

    let mean_a = private_a.iter().sum::<f32>() / private_a.len() as f32;
    let mean_b = private_b.iter().sum::<f32>() / private_b.len() as f32;

    // Shared neuron mean activation.
    let shared_vals: Vec<f32> = zone
        .shared_neuron_indices
        .iter()
        .map(|&idx| mesh.neurons[idx].x)
        .collect();
    let mean_shared = shared_vals.iter().sum::<f32>() / shared_vals.len() as f32;

    // Correlation-based loss: -corr(shared, private_a) - corr(shared, private_b)
    // Simplified: use -(mean_shared * mean_a + mean_shared * mean_b)
    // This encourages shared neurons to be co-active with both circuits.
    let loss = -(mean_shared * mean_a + mean_shared * mean_b);

    // Gradient for shared neurons: ∂L/∂x_shared = -(mean_a + mean_b) / N_shared
    let dl_d_shared = -(mean_a + mean_b) / shared_vals.len() as f32;
    for &idx in &zone.shared_neuron_indices {
        dl_dx[idx] = dl_d_shared;
    }

    (loss, dl_dx)
}

/// Compute combined sparsity loss for all overlap zones.
pub fn total_sparsity_loss(mesh: &OverlappingMesh) -> f32 {
    mesh.overlap_zones.iter().map(|z| z.sparsity_loss()).sum()
}

/// Compute overlap stability loss: Σ(mask - mask_ref)² across all zones.
pub fn stability_loss(mesh: &OverlappingMesh, mask_refs: &[Vec<f32>]) -> f32 {
    let mut loss = 0.0;
    for (zone_idx, zone) in mesh.overlap_zones.iter().enumerate() {
        if zone_idx >= mask_refs.len() {
            continue;
        }
        for (i, &mask) in zone.masks.iter().enumerate() {
            if i < mask_refs[zone_idx].len() {
                let diff = mask - mask_refs[zone_idx][i];
                loss += diff * diff;
            }
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bce_perfect_prediction() {
        // Perfect prediction: low loss.
        let loss = binary_cross_entropy(0.99, 1.0);
        assert!(loss < 0.02);

        let loss = binary_cross_entropy(0.01, 0.0);
        assert!(loss < 0.02);
    }

    #[test]
    fn bce_bad_prediction() {
        // Bad prediction: high loss.
        let loss = binary_cross_entropy(0.01, 1.0);
        assert!(loss > 4.0);
    }

    #[test]
    fn bce_grad_direction() {
        // If y_true=1 and y_pred is low, gradient should be negative
        // (push y_pred up).
        let grad = binary_cross_entropy_grad(0.1, 1.0);
        assert!(grad < 0.0);

        // If y_true=0 and y_pred is high, gradient should be positive
        // (push y_pred down).
        let grad = binary_cross_entropy_grad(0.9, 0.0);
        assert!(grad > 0.0);
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }
}

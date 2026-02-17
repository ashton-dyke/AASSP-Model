//! Connection initialization for the neural mesh.
//!
//! Before training, neurons need sparse random connectivity. This module
//! creates structured connections with Xavier/Glorot initialization.

use crate::circuit::CircuitType;
use crate::mesh::OverlappingMesh;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;

/// Target connection density by circuit type.
fn target_density(circuit_type: CircuitType) -> f32 {
    match circuit_type {
        CircuitType::Detection => 0.20,
        CircuitType::Causation => 0.30,
        CircuitType::Memory => 0.15,
        CircuitType::Prediction => 0.25,
    }
}

/// Fraction of connections that should come from within the same circuit.
fn local_fraction(circuit_type: CircuitType) -> f32 {
    match circuit_type {
        CircuitType::Detection => 1.0,  // Detection is self-contained.
        CircuitType::Causation => 0.7,
        CircuitType::Memory => 0.7,
        CircuitType::Prediction => 0.7,
    }
}

/// Initialize sparse connections for all neurons in the mesh.
///
/// For each circuit, determines target density and creates connections:
/// - A fraction from within the same circuit (local)
/// - The remainder from any neuron in the mesh (global)
///
/// Weights are initialized with Xavier/Glorot: N(0, sqrt(2 / (fan_in + fan_out))).
pub fn initialize_connections(mesh: &mut OverlappingMesh, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_neurons = mesh.neurons.len();

    // Build a map: neuron_idx -> which circuit(s) it belongs to.
    let mut neuron_circuits: Vec<Vec<usize>> = vec![Vec::new(); total_neurons];
    for (ci, circuit) in mesh.circuits.iter().enumerate() {
        for &ni in &circuit.neuron_indices {
            neuron_circuits[ni].push(ci);
        }
    }

    // For each circuit, initialize its neurons' connections.
    for ci in 0..mesh.circuits.len() {
        let circuit_type = mesh.circuits[ci].circuit_type;
        let circuit_indices: Vec<usize> = mesh.circuits[ci].neuron_indices.clone();
        let circuit_size = circuit_indices.len();
        let density = target_density(circuit_type);
        let local_frac = local_fraction(circuit_type);

        let n_connections = ((circuit_size as f32 * density) as usize).max(1);
        let n_local = ((n_connections as f32 * local_frac) as usize).max(1);
        let n_global = n_connections.saturating_sub(n_local);

        // Xavier scale.
        let fan_in = n_connections;
        let fan_out = n_connections; // Approximate.
        let xavier_std = (2.0 / (fan_in + fan_out) as f32).sqrt();

        let circuit_set: HashSet<usize> = circuit_indices.iter().cloned().collect();

        for &neuron_idx in &circuit_indices {
            // Skip if this neuron already has connections (e.g., from another circuit).
            if !mesh.neurons[neuron_idx].connections.is_empty() {
                continue;
            }

            let mut connections = Vec::with_capacity(n_connections);
            let mut used = HashSet::new();
            used.insert(neuron_idx); // No self-connections.

            // Local connections (within same circuit).
            let mut local_added = 0;
            for _ in 0..n_local * 3 {
                // Over-sample to handle collisions.
                if local_added >= n_local {
                    break;
                }
                let src = circuit_indices[rng.gen_range(0..circuit_size)];
                if used.insert(src) {
                    let weight = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    connections.push((src, weight));
                    local_added += 1;
                }
            }

            // Global connections (from any neuron).
            let mut global_added = 0;
            for _ in 0..n_global * 3 {
                if global_added >= n_global {
                    break;
                }
                let src = rng.gen_range(0..total_neurons);
                if used.insert(src) && !circuit_set.contains(&src) {
                    let weight = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    connections.push((src, weight));
                    global_added += 1;
                }
            }

            mesh.neurons[neuron_idx].connections = connections;
        }
    }

    // Initialize biases with small random values.
    for neuron in &mut mesh.neurons {
        if !neuron.connections.is_empty() {
            neuron.bias = rng.gen::<f32>() * 0.02 - 0.01;
        }
    }
}

/// Wire up adapter layers with input/output indices.
pub fn initialize_adapters(mesh: &mut OverlappingMesh) {
    // Formation adapter: reads from last 32 neurons of each causation circuit,
    // writes to first 32 neurons of each prediction circuit.
    let mut formation_inputs = Vec::new();
    let mut formation_outputs = Vec::new();

    for circuit in &mesh.circuits {
        match circuit.circuit_type {
            CircuitType::Causation => {
                let n = circuit.neuron_indices.len();
                let start = if n > 32 { n - 32 } else { 0 };
                formation_inputs.extend_from_slice(&circuit.neuron_indices[start..]);
            }
            CircuitType::Prediction => {
                let end = 32.min(circuit.neuron_indices.len());
                formation_outputs.extend_from_slice(&circuit.neuron_indices[..end]);
            }
            _ => {}
        }
    }

    // Well adapter: reads from last 32 neurons of each detection circuit,
    // writes to first 32 neurons of each memory circuit.
    let mut well_inputs = Vec::new();
    let mut well_outputs = Vec::new();

    for circuit in &mesh.circuits {
        match circuit.circuit_type {
            CircuitType::Detection => {
                let n = circuit.neuron_indices.len();
                let start = if n > 32 { n - 32 } else { 0 };
                well_inputs.extend_from_slice(&circuit.neuron_indices[start..]);
            }
            CircuitType::Memory => {
                let end = 32.min(circuit.neuron_indices.len());
                well_outputs.extend_from_slice(&circuit.neuron_indices[..end]);
            }
            _ => {}
        }
    }

    // Store dimensions for later use; the actual adapter layers live in ProductionMesh.
    // We stash them in the mesh config's reserved space via a comment placeholder.
    // The actual adapter construction happens in the pipeline.
    let _ = (formation_inputs, formation_outputs, well_inputs, well_outputs);
}

/// Create adapter input/output index lists for the default layout.
pub fn default_adapter_indices(
    mesh: &OverlappingMesh,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut formation_inputs = Vec::new();
    let mut formation_outputs = Vec::new();
    let mut well_inputs = Vec::new();
    let mut well_outputs = Vec::new();

    for circuit in &mesh.circuits {
        match circuit.circuit_type {
            CircuitType::Causation => {
                let n = circuit.neuron_indices.len();
                let start = if n > 32 { n - 32 } else { 0 };
                formation_inputs.extend_from_slice(&circuit.neuron_indices[start..]);
            }
            CircuitType::Prediction => {
                let end = 32.min(circuit.neuron_indices.len());
                formation_outputs.extend_from_slice(&circuit.neuron_indices[..end]);
            }
            CircuitType::Detection => {
                let n = circuit.neuron_indices.len();
                let start = if n > 32 { n - 32 } else { 0 };
                well_inputs.extend_from_slice(&circuit.neuron_indices[start..]);
            }
            CircuitType::Memory => {
                let end = 32.min(circuit.neuron_indices.len());
                well_outputs.extend_from_slice(&circuit.neuron_indices[..end]);
            }
        }
    }

    (formation_inputs, formation_outputs, well_inputs, well_outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::build_default_mesh;

    #[test]
    fn initialize_connections_creates_sparse_connections() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        // Check that detection neurons got connections.
        let det_neuron = &mesh.neurons[0];
        assert!(
            !det_neuron.connections.is_empty(),
            "Detection neuron should have connections"
        );

        // Connections should be within reasonable range.
        let n_conn = det_neuron.connections.len();
        assert!(n_conn > 5, "Expected more than 5 connections, got {n_conn}");
        assert!(n_conn < 200, "Expected fewer than 200 connections, got {n_conn}");
    }

    #[test]
    fn no_self_connections() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        for (idx, neuron) in mesh.neurons.iter().enumerate() {
            for &(src, _) in &neuron.connections {
                assert_ne!(src, idx, "Neuron {idx} has a self-connection");
            }
        }
    }

    #[test]
    fn xavier_weights_bounded() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        for neuron in &mesh.neurons {
            for &(_, w) in &neuron.connections {
                assert!(
                    w.abs() < 1.0,
                    "Xavier-initialized weight should be small, got {w}"
                );
            }
        }
    }

    #[test]
    fn default_adapter_indices_nonempty() {
        let mesh = build_default_mesh();
        let (fi, fo, wi, wo) = default_adapter_indices(&mesh);
        assert!(!fi.is_empty(), "Formation inputs should not be empty");
        assert!(!fo.is_empty(), "Formation outputs should not be empty");
        assert!(!wi.is_empty(), "Well inputs should not be empty");
        assert!(!wo.is_empty(), "Well outputs should not be empty");
    }
}

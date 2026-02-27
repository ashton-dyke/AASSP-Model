//! Connection initialization for the neural mesh.
//!
//! Before training, LtcNeurons need sparse random connectivity for both
//! drive and gate pathways. This module creates structured connections
//! with Xavier/Glorot initialization, initializes gate weights, and sets
//! per-neuron tau from the circuit's tau field with small random perturbation.

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
        CircuitType::Detection => 1.0, // Detection is self-contained.
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
/// Drive weights are initialized with Xavier/Glorot: N(0, sqrt(2 / (fan_in + fan_out))).
/// Gate weights are initialized with Xavier/Glorot (same connectivity as drive).
/// Gate bias is initialized to a small negative value (-0.5) to keep gates conservative.
/// Per-neuron tau is set from the circuit's tau field with small random perturbation.
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
        let circuit_tau = mesh.circuits[ci].tau;
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
            let mut gate_weights = Vec::with_capacity(n_connections);
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
                    let drive_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    let gate_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    connections.push((src, drive_w));
                    gate_weights.push(gate_w);
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
                    let drive_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    let gate_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                    connections.push((src, drive_w));
                    gate_weights.push(gate_w);
                    global_added += 1;
                }
            }

            mesh.neurons[neuron_idx].connections = connections;
            mesh.neurons[neuron_idx].gate_weights = gate_weights;

            // Set per-neuron tau from circuit tau with small random perturbation.
            // Perturbation: multiply by (1 + uniform(-0.1, 0.1))
            let perturbation = 1.0 + (rng.gen::<f32>() * 0.2 - 0.1);
            mesh.neurons[neuron_idx].tau = (circuit_tau * perturbation).max(1e-4);
        }
    }

    // Initialize biases.
    for neuron in &mut mesh.neurons {
        if !neuron.connections.is_empty() {
            // Drive bias: small random values.
            neuron.bias = rng.gen::<f32>() * 0.02 - 0.01;
            // Gate bias: small negative value to keep gates initially conservative.
            neuron.gate_bias = -0.5 + rng.gen::<f32>() * 0.1 - 0.05;
        }
    }

    // Add skip connections from input neurons to readout neurons in detection circuits.
    // CfC dynamics attenuate signals at each hop (steady state ∝ 1/alpha). Without
    // direct connections, readout neurons only see attenuated multi-hop signals that
    // lose all input-dependent variation. Skip connections give a 1-hop path so the
    // readout can actually discriminate between normal and anomalous input patterns.
    add_input_to_readout_skip_connections(mesh, &mut rng);

    // Open up the gates on readout neurons: set gate_bias near 0 so that
    // f ≈ sigmoid(0) = 0.5 instead of sigmoid(-0.5) ≈ 0.38. This allows
    // more signal throughput for the skip connections from input neurons.
    for circuit in &mesh.circuits {
        if circuit.circuit_type != CircuitType::Detection {
            continue;
        }
        let n = circuit.neuron_indices.len();
        if n > READOUT_NEURONS {
            let start = n - READOUT_NEURONS;
            for &idx in &circuit.neuron_indices[start..] {
                mesh.neurons[idx].gate_bias = 0.0;
            }
        }
    }
}

/// Number of WITS input features clamped into the first N neurons of each detection circuit.
const INPUT_NEURONS: usize = 14;
/// Number of readout neurons at the tail of each detection circuit (matches loss.rs).
const READOUT_NEURONS: usize = 8;

/// Add skip connections from input neurons to readout neurons in detection circuits.
///
/// For each detection circuit, the first `INPUT_NEURONS` neurons are clamped with
/// WITS data and the last `READOUT_NEURONS` are used for the detection readout.
/// This function adds direct connections between them so that the readout can
/// respond to input in a single CfC hop, bypassing multi-hop attenuation.
fn add_input_to_readout_skip_connections(mesh: &mut OverlappingMesh, rng: &mut StdRng) {
    for circuit in &mesh.circuits {
        if circuit.circuit_type != CircuitType::Detection {
            continue;
        }

        let indices = &circuit.neuron_indices;
        let n = indices.len();
        if n <= INPUT_NEURONS + READOUT_NEURONS {
            continue;
        }

        let input_indices: Vec<usize> = indices[..INPUT_NEURONS].to_vec();
        let readout_start = n - READOUT_NEURONS;
        let readout_indices: Vec<usize> = indices[readout_start..].to_vec();

        // Skip connection weight scale: 3x Xavier to ensure these connections
        // dominate the readout signal over the ~40 random intra-circuit connections.
        // Without this boost, the random connections (carrying tiny, non-discriminative
        // steady-state activations) drown out the input-dependent skip signal.
        let fan = INPUT_NEURONS + READOUT_NEURONS;
        let xavier_std = 3.0 * (2.0 / fan as f32).sqrt();

        for &readout_idx in &readout_indices {
            let existing: HashSet<usize> = mesh.neurons[readout_idx]
                .connections
                .iter()
                .map(|&(src, _)| src)
                .collect();

            for &input_idx in &input_indices {
                if existing.contains(&input_idx) || input_idx == readout_idx {
                    continue;
                }
                let drive_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                let gate_w = rng.gen::<f32>() * 2.0 * xavier_std - xavier_std;
                mesh.neurons[readout_idx].connections.push((input_idx, drive_w));
                mesh.neurons[readout_idx].gate_weights.push(gate_w);
            }
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
                    "Xavier-initialized drive weight should be small, got {w}"
                );
            }
            for &gw in &neuron.gate_weights {
                assert!(
                    gw.abs() < 1.0,
                    "Xavier-initialized gate weight should be small, got {gw}"
                );
            }
        }
    }

    #[test]
    fn gate_weights_parallel_to_connections() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        for neuron in &mesh.neurons {
            assert_eq!(
                neuron.connections.len(),
                neuron.gate_weights.len(),
                "Gate weights should be parallel to connections"
            );
        }
    }

    #[test]
    fn gate_bias_initialized_negative() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        // Check that non-readout neurons with connections have negative gate bias.
        // Readout neurons (last 8 per detection circuit) have gate_bias=0 for
        // better signal throughput.
        let readout_set: std::collections::HashSet<usize> = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Detection)
            .flat_map(|c| {
                let n = c.neuron_indices.len();
                let start = if n > 8 { n - 8 } else { 0 };
                c.neuron_indices[start..].iter().cloned()
            })
            .collect();

        let non_readout: Vec<_> = mesh
            .neurons
            .iter()
            .enumerate()
            .filter(|(i, n)| !n.connections.is_empty() && !readout_set.contains(i))
            .map(|(_, n)| n)
            .collect();
        assert!(!non_readout.is_empty());

        for neuron in &non_readout {
            assert!(
                neuron.gate_bias < 0.0,
                "Non-readout gate bias should be negative for conservative initial gating, got {}",
                neuron.gate_bias
            );
        }
    }

    #[test]
    fn tau_set_from_circuit() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        // Detection neurons (circuit tau=0.1) should have tau near 0.1.
        let det_neuron = &mesh.neurons[0];
        assert!(
            det_neuron.tau > 0.08 && det_neuron.tau < 0.12,
            "Detection neuron tau should be near 0.1, got {}",
            det_neuron.tau
        );

        // Causation neurons (circuit tau=0.15) should have tau near 0.15.
        let caus_neuron = &mesh.neurons[1024];
        assert!(
            caus_neuron.tau > 0.12 && caus_neuron.tau < 0.18,
            "Causation neuron tau should be near 0.15, got {}",
            caus_neuron.tau
        );
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

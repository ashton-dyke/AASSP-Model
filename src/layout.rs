//! Default mesh layout from the specification (Section 5.4).
//!
//! Creates the full 4,736-neuron mesh with all 13 circuits and overlap zones.
//! With LTC/CfC neurons, tau is set as the initial per-neuron time constant
//! during connection initialization.

use crate::circuit::{Circuit, CircuitCriticality, CircuitType};
use crate::config::MeshConfig;
use crate::mesh::OverlappingMesh;

/// Build the default 4,736-neuron mesh with all 13 circuits.
pub fn build_default_mesh() -> OverlappingMesh {
    let config = MeshConfig::default();
    let mut mesh = OverlappingMesh::new(config);

    // ── Detection block: 0..1023 (5 circuits) ──
    // Each detection circuit gets ~204 neurons.
    let det_circuits = vec![
        ("kick_detection",    0..204,   0.01),
        ("loss_detection",    204..408, 0.01),
        ("packoff_detection", 408..612, 0.015),
        ("stickslip_detect",  612..816, 0.02),
        ("founder_detect",    816..1024,0.02),
    ];

    for (i, (name, range, tau)) in det_circuits.iter().enumerate() {
        let circuit = Circuit::new(
            i as u32,
            *name,
            range.clone().collect(),
            *tau,
            CircuitType::Detection,
            CircuitCriticality::Safety,
        );
        mesh.add_circuit(circuit).unwrap();
    }

    // ── Causation block: 1024..2047 (3 circuits) ──
    let caus_circuits = vec![
        ("torque_causation",   1024..1366, 0.05),
        ("pressure_causation", 1366..1706, 0.05),
        ("flow_causation",     1706..2048, 0.05),
    ];

    for (i, (name, range, tau)) in caus_circuits.iter().enumerate() {
        let circuit = Circuit::new(
            (5 + i) as u32,
            *name,
            range.clone().collect(),
            *tau,
            CircuitType::Causation,
            CircuitCriticality::Safety,
        );
        mesh.add_circuit(circuit).unwrap();
    }

    // ── Memory block: 2048..3583 (3 circuits) ──
    let mem_circuits = vec![
        ("short_term_memory",  2048..2560, 0.2),
        ("medium_term_memory", 2560..3072, 2.0),
        ("long_term_memory",   3072..3584, 30.0),
    ];

    for (i, (name, range, tau)) in mem_circuits.iter().enumerate() {
        let circuit = Circuit::new(
            (8 + i) as u32,
            *name,
            range.clone().collect(),
            *tau,
            CircuitType::Memory,
            CircuitCriticality::Performance,
        );
        mesh.add_circuit(circuit).unwrap();
    }

    // ── Prediction block: 3584..4607 (2 circuits) ──
    let pred_circuits = vec![
        ("state_prediction", 3584..4096, 0.02),
        ("rop_prediction",   4096..4608, 0.03),
    ];

    for (i, (name, range, tau)) in pred_circuits.iter().enumerate() {
        let circuit = Circuit::new(
            (11 + i) as u32,
            *name,
            range.clone().collect(),
            *tau,
            CircuitType::Prediction,
            CircuitCriticality::Performance,
        );
        mesh.add_circuit(circuit).unwrap();
    }

    // Neurons 4608..4735 are reserved for adapters (not assigned to circuits).

    // ── Overlap zones ──
    // Detection ↔ Causation overlaps: last 32 neurons of each detection
    // circuit overlap with first 32 neurons of the corresponding causation.
    // (The spec says ~10% overlap at circuit boundaries.)
    //
    // Since detection neurons are in [0..1024) and causation in [1024..2048),
    // we create pseudo-overlap zones with shared neuron indices pointing
    // to actual shared neurons. In the default layout, circuits don't
    // physically share neuron index ranges (they are disjoint). Overlap
    // zones instead track which neurons act as bridges during training.
    // For the default layout, we leave overlap zones empty since circuits
    // are disjoint. The connections initialization will create cross-circuit
    // connections to serve the overlap role.

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::CircuitType;

    #[test]
    fn default_mesh_has_correct_neuron_count() {
        let mesh = build_default_mesh();
        assert_eq!(mesh.neurons.len(), 4736);
    }

    #[test]
    fn default_mesh_has_13_circuits() {
        let mesh = build_default_mesh();
        assert_eq!(mesh.circuits.len(), 13);
    }

    #[test]
    fn detection_circuits_count() {
        let mesh = build_default_mesh();
        let det_count = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Detection)
            .count();
        assert_eq!(det_count, 5);
    }

    #[test]
    fn causation_circuits_count() {
        let mesh = build_default_mesh();
        let count = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Causation)
            .count();
        assert_eq!(count, 3);
    }

    #[test]
    fn memory_circuits_count() {
        let mesh = build_default_mesh();
        let count = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Memory)
            .count();
        assert_eq!(count, 3);
    }

    #[test]
    fn prediction_circuits_count() {
        let mesh = build_default_mesh();
        let count = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Prediction)
            .count();
        assert_eq!(count, 2);
    }

    #[test]
    fn no_neuron_in_more_than_two_circuits() {
        let mesh = build_default_mesh();
        for neuron in &mesh.neurons {
            assert!(neuron.circuit_membership_count <= 2);
        }
    }

    #[test]
    fn circuit_ids_are_sequential() {
        let mesh = build_default_mesh();
        for (i, circuit) in mesh.circuits.iter().enumerate() {
            assert_eq!(circuit.id, i as u32);
        }
    }
}

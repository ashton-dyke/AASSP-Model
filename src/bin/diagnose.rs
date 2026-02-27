//! Diagnostic: run full training and inspect per-circuit readouts.

use sairen_mesh::adaptation::production::ProductionMesh;
use sairen_mesh::layout::build_default_mesh;
use sairen_mesh::training::loss::detection_readout;
use sairen_mesh::training::pipeline::{run_full_training, TrainingConfig};
use sairen_mesh::training::synthetic::SyntheticWellGenerator;

fn main() {
    eprintln!("=== POST-TRAINING DIAGNOSTIC ===\n");

    let config = TrainingConfig {
        stage1_steps: 200,
        stage2_rounds: 3,
        stage2_steps_per_round: 300,
        stage2_memory_steps: 50,
        stage2_prediction_steps: 50,
        stage3_steps: 400,
        mesh_steps_per_sample: 50,
        synthetic_sequences: 20,
        sequence_length: 50,
        mini_batch_size: 2,
        ..TrainingConfig::default()
    };

    eprintln!("Training...");
    let mut mesh = build_default_mesh();
    let metrics = run_full_training(&mut mesh, &config);
    eprintln!("Detection accuracy: {:.1}%\n", metrics.final_detection_accuracy * 100.0);

    // Wrap in ProductionMesh.
    let mut prod = ProductionMesh::from_trained(mesh);

    // Generate test data.
    let mut gen = SyntheticWellGenerator::new(99);
    let normal_seq = gen.normal_sequence(20);
    let kick_seq = gen.kick_sequence(30, 10, 0.9);

    // Warm up baselines with normal data.
    eprintln!("=== NORMAL DATA ===");
    for (i, wits) in normal_seq.wits_samples.iter().enumerate() {
        let output = prod.process(wits);
        if i >= 15 {
            let readouts: Vec<f32> = (0..5)
                .map(|ci| detection_readout(&prod.universal_mesh, ci))
                .collect();
            eprintln!(
                "  Sample {:2}: readouts=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}] det={} risk={:?}",
                i, readouts[0], readouts[1], readouts[2], readouts[3], readouts[4],
                output.detections.len(), output.risk_level
            );
        }
    }

    // Reset mesh state before kick data to test detection capability from fresh state.
    prod.universal_mesh.reset_activations();
    eprintln!("\n=== KICK DATA (onset at sample 10, reset) ===");
    for (i, wits) in kick_seq.wits_samples.iter().enumerate() {
        let output = prod.process(wits);
        let readouts: Vec<f32> = (0..5)
            .map(|ci| detection_readout(&prod.universal_mesh, ci))
            .collect();
        let det_info: Vec<String> = output.detections.iter()
            .map(|d| format!("{}:{:.2}", d.circuit_name, d.confidence))
            .collect();
        if i < 5 || i >= 10 {
            eprintln!(
                "  Sample {:2}: readouts=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}] det={} risk={:?} {}",
                i, readouts[0], readouts[1], readouts[2], readouts[3], readouts[4],
                output.detections.len(), output.risk_level,
                if det_info.is_empty() { String::new() } else { format!("{:?}", det_info) }
            );
        }
    }

    // Summary statistics.
    eprintln!("\n=== SUMMARY ===");
    let normal_det_counts: Vec<usize> = normal_seq.wits_samples.iter()
        .map(|w| prod.process(w).detections.len())
        .collect();
    let kick_det_counts: Vec<usize> = kick_seq.wits_samples.iter()
        .map(|w| prod.process(w).detections.len())
        .collect();

    let normal_avg = normal_det_counts.iter().sum::<usize>() as f32 / normal_det_counts.len() as f32;
    let kick_avg = kick_det_counts.iter().sum::<usize>() as f32 / kick_det_counts.len() as f32;
    eprintln!("Avg detections per sample - Normal: {:.1}, Kick: {:.1}", normal_avg, kick_avg);
}

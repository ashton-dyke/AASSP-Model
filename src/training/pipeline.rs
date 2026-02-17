//! Training pipeline: orchestrates the three-stage offline training process.

use crate::io::encoding::{encode_input, Baselines};
use crate::mesh::OverlappingMesh;
use crate::training::backward::backprop_through_time;
use crate::training::data::{causation_labels_for_sample, detection_labels_for_sample};
use crate::training::init::initialize_connections;
use crate::training::loss;
use crate::training::optimizer::AdamOptimizer;
use crate::training::state::{ForwardRecord, GradientAccumulator};
use crate::training::synthetic::{SyntheticWellGenerator, TrainingSequence};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub stage1_steps: u32,
    pub stage1_lr: f32,
    pub stage2_rounds: u32,
    pub stage2_steps_per_round: u32,
    pub stage2_lr: f32,
    pub stage2_memory_steps: u32,
    pub stage2_memory_lr: f32,
    pub stage2_prediction_steps: u32,
    pub stage2_prediction_lr: f32,
    pub stage3_steps: u32,
    pub stage3_lr: f32,
    pub sparsity_lambda: f32,
    pub stability_lambda: f32,
    pub physics_lambda: f32,
    pub max_grad_norm: f32,
    pub mesh_steps_per_sample: u32,
    pub seed: u64,
    pub synthetic_sequences: usize,
    pub sequence_length: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            stage1_steps: 5_000,
            stage1_lr: 1e-3,
            stage2_rounds: 5,
            stage2_steps_per_round: 500,
            stage2_lr: 1e-3,
            stage2_memory_steps: 200,
            stage2_memory_lr: 5e-4,
            stage2_prediction_steps: 200,
            stage2_prediction_lr: 5e-4,
            stage3_steps: 2_000,
            stage3_lr: 1e-4,
            sparsity_lambda: 0.01,
            stability_lambda: 0.1,
            physics_lambda: 0.05,
            max_grad_norm: 1.0,
            mesh_steps_per_sample: 100,
            seed: 42,
            synthetic_sequences: 120,
            sequence_length: 100,
        }
    }
}

/// Accumulated training metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub stage1_losses: Vec<f32>,
    pub stage2_detection_losses: Vec<f32>,
    pub stage2_causation_losses: Vec<f32>,
    pub stage3_losses: Vec<f32>,
    pub final_detection_accuracy: f32,
}

/// Run a single forward pass through the mesh for one WITS sample,
/// recording state for BPTT.
pub fn recorded_forward_pass(
    mesh: &mut OverlappingMesh,
    record: &mut ForwardRecord,
    steps: u32,
) {
    let dt = mesh.config.dt;

    for _ in 0..steps {
        record.begin_step();

        // Determine which circuits fire this step.
        let rng = mesh.rng.get_or_insert_with(|| StdRng::seed_from_u64(42));
        let mut firing_circuits: Vec<(u32, usize)> = Vec::new(); // (id, idx)
        for (ci, circuit) in mesh.circuits.iter_mut().enumerate() {
            if circuit.should_fire(dt, rng) {
                firing_circuits.push((circuit.id, ci));
                record.record_firing(circuit.id);
            }
        }

        // Compute updates and record state.
        let mut all_updates: Vec<(u32, Vec<(usize, f32)>)> = Vec::new();

        for &(circuit_id, ci) in &firing_circuits {
            let tau = mesh.circuits[ci].tau;
            let mut updates = Vec::new();

            for &neuron_idx in &mesh.circuits[ci].neuron_indices {
                let neuron = &mesh.neurons[neuron_idx];
                let input_sum: f32 = neuron
                    .connections
                    .iter()
                    .map(|&(src, w)| mesh.neurons[src].x * w)
                    .sum::<f32>()
                    + neuron.bias;

                // Record before update.
                record.record_neuron(neuron_idx, neuron.x, input_sum, circuit_id, tau);

                let dx = (-neuron.x + input_sum.tanh()) / tau * dt;
                updates.push((neuron_idx, dx));
            }

            mesh.circuits[ci].update_confidence(&mesh.neurons);
            all_updates.push((circuit_id, updates));
        }

        // Merge updates (same as mesh.merge_updates but we own the data).
        mesh.merge_updates(&all_updates);
        mesh.tick += 1;
    }
}

/// Run Stage 1: Overlap pre-training.
///
/// Trains only overlap masks and gate weights using mutual information loss.
fn run_stage1(
    mesh: &mut OverlappingMesh,
    config: &TrainingConfig,
    dataset: &[TrainingSequence],
    baselines: &Baselines,
    metrics: &mut TrainingMetrics,
) {
    let mut optimizer = AdamOptimizer::from_mesh(mesh, config.stage1_lr);
    let mut rng = StdRng::seed_from_u64(config.seed);

    // In Stage 1, all neuron weights are frozen — only gates/masks train.
    let all_circuit_ids: Vec<u32> = mesh.circuits.iter().map(|c| c.id).collect();

    for step in 0..config.stage1_steps {
        let seq_idx = (step as usize) % dataset.len();
        let sample_idx = rand::Rng::gen_range(&mut rng, 0..dataset[seq_idx].wits_samples.len());
        let wits = &dataset[seq_idx].wits_samples[sample_idx];

        // Encode input.
        encode_input(mesh, wits, baselines);

        // Forward pass (we only need final state for MI loss).
        for _ in 0..config.mesh_steps_per_sample {
            mesh.step();
        }

        // Compute MI loss for each overlap zone.
        let mut total_loss = 0.0;
        let mut grads = GradientAccumulator::from_mesh(mesh);

        for zone_idx in 0..mesh.overlap_zones.len() {
            let (mi_loss, mi_dl_dx) = loss::mutual_information_loss(mesh, zone_idx);
            total_loss += mi_loss;

            // Accumulate mask gradients (L1 sparsity).
            for (i, &mask) in mesh.overlap_zones[zone_idx].masks.iter().enumerate() {
                if zone_idx < grads.mask_grads.len() && i < grads.mask_grads[zone_idx].len() {
                    grads.mask_grads[zone_idx][i] +=
                        config.sparsity_lambda * mask.signum();
                }
            }

            for (i, &g) in mi_dl_dx.iter().enumerate() {
                grads.dx[i] += g;
            }
        }

        total_loss += config.sparsity_lambda * loss::total_sparsity_loss(mesh);

        // Clip and apply (neuron weights frozen via frozen_circuits).
        grads.clip_global_norm(config.max_grad_norm);
        optimizer.step(mesh, &grads, &all_circuit_ids, false);

        if step % 500 == 0 {
            metrics.stage1_losses.push(total_loss);
        }
    }
}

/// Run Stage 2: Alternating freeze circuit training.
fn run_stage2(
    mesh: &mut OverlappingMesh,
    config: &TrainingConfig,
    dataset: &[TrainingSequence],
    baselines: &Baselines,
    metrics: &mut TrainingMetrics,
) {
    let detection_ids: Vec<u32> = mesh
        .circuits
        .iter()
        .filter(|c| c.circuit_type == crate::circuit::CircuitType::Detection)
        .map(|c| c.id)
        .collect();
    let causation_ids: Vec<u32> = mesh
        .circuits
        .iter()
        .filter(|c| c.circuit_type == crate::circuit::CircuitType::Causation)
        .map(|c| c.id)
        .collect();
    let memory_ids: Vec<u32> = mesh
        .circuits
        .iter()
        .filter(|c| c.circuit_type == crate::circuit::CircuitType::Memory)
        .map(|c| c.id)
        .collect();
    let prediction_ids: Vec<u32> = mesh
        .circuits
        .iter()
        .filter(|c| c.circuit_type == crate::circuit::CircuitType::Prediction)
        .map(|c| c.id)
        .collect();

    let all_ids: Vec<u32> = mesh.circuits.iter().map(|c| c.id).collect();
    let freeze_except_causation: Vec<u32> = all_ids.iter().filter(|id| !causation_ids.contains(id)).cloned().collect();
    let freeze_except_detection: Vec<u32> = all_ids.iter().filter(|id| !detection_ids.contains(id)).cloned().collect();
    let freeze_except_memory: Vec<u32> = all_ids.iter().filter(|id| !memory_ids.contains(id)).cloned().collect();
    let freeze_except_prediction: Vec<u32> = all_ids.iter().filter(|id| !prediction_ids.contains(id)).cloned().collect();

    let mut rng = StdRng::seed_from_u64(config.seed + 1);

    for _round in 0..config.stage2_rounds {
        // Round A: Train causation, freeze detection.
        train_circuit_group(
            mesh,
            config,
            dataset,
            baselines,
            &freeze_except_causation,
            config.stage2_steps_per_round,
            config.stage2_lr,
            &mut rng,
            metrics,
            true,
        );

        // Round B: Train detection, freeze causation.
        train_circuit_group(
            mesh,
            config,
            dataset,
            baselines,
            &freeze_except_detection,
            config.stage2_steps_per_round,
            config.stage2_lr,
            &mut rng,
            metrics,
            false,
        );

        // Round C: Train memory.
        train_circuit_group(
            mesh,
            config,
            dataset,
            baselines,
            &freeze_except_memory,
            config.stage2_memory_steps,
            config.stage2_memory_lr,
            &mut rng,
            metrics,
            false,
        );

        // Round D: Train prediction.
        train_circuit_group(
            mesh,
            config,
            dataset,
            baselines,
            &freeze_except_prediction,
            config.stage2_prediction_steps,
            config.stage2_prediction_lr,
            &mut rng,
            metrics,
            false,
        );
    }
}

/// Train unfrozen circuits for a number of steps.
fn train_circuit_group(
    mesh: &mut OverlappingMesh,
    config: &TrainingConfig,
    dataset: &[TrainingSequence],
    baselines: &Baselines,
    frozen_circuits: &[u32],
    steps: u32,
    lr: f32,
    rng: &mut StdRng,
    metrics: &mut TrainingMetrics,
    is_causation_round: bool,
) {
    let mut optimizer = AdamOptimizer::from_mesh(mesh, lr);

    for step in 0..steps {
        let seq_idx = rand::Rng::gen_range(rng, 0..dataset.len());
        let seq = &dataset[seq_idx];
        let sample_idx = rand::Rng::gen_range(rng, 0..seq.wits_samples.len());

        encode_input(mesh, &seq.wits_samples[sample_idx], baselines);

        // Forward with recording.
        let mut record = ForwardRecord::new(mesh.neurons.len(), config.mesh_steps_per_sample as usize);
        recorded_forward_pass(mesh, &mut record, config.mesh_steps_per_sample);

        // Compute loss.
        let det_labels = detection_labels_for_sample(seq, sample_idx);
        let (det_loss, det_dl_dx) = loss::detection_loss(mesh, &det_labels);

        let caus_labels = causation_labels_for_sample(seq, sample_idx);
        let (caus_loss, caus_dl_dx) = loss::causation_loss(mesh, &caus_labels);

        // Combine output gradients.
        let mut dl_dx_output = vec![0.0f32; mesh.neurons.len()];
        for i in 0..dl_dx_output.len() {
            dl_dx_output[i] = det_dl_dx[i] + caus_dl_dx[i];
        }

        // BPTT.
        let mut grads = GradientAccumulator::from_mesh(mesh);
        backprop_through_time(mesh, &record, &dl_dx_output, &mut grads);

        grads.clip_global_norm(config.max_grad_norm);
        optimizer.step(mesh, &grads, frozen_circuits, true); // overlaps frozen in stage 2

        if step % 100 == 0 {
            if is_causation_round {
                metrics.stage2_causation_losses.push(caus_loss);
            } else {
                metrics.stage2_detection_losses.push(det_loss);
            }
        }
    }
}

/// Run Stage 3: Joint fine-tuning.
fn run_stage3(
    mesh: &mut OverlappingMesh,
    config: &TrainingConfig,
    dataset: &[TrainingSequence],
    baselines: &Baselines,
    mask_refs: &[Vec<f32>],
    metrics: &mut TrainingMetrics,
) {
    let mut optimizer = AdamOptimizer::from_mesh(mesh, config.stage3_lr);
    let mut rng = StdRng::seed_from_u64(config.seed + 2);

    for step in 0..config.stage3_steps {
        let seq_idx = rand::Rng::gen_range(&mut rng, 0..dataset.len());
        let seq = &dataset[seq_idx];
        let sample_idx = rand::Rng::gen_range(&mut rng, 0..seq.wits_samples.len());

        encode_input(mesh, &seq.wits_samples[sample_idx], baselines);

        let mut record = ForwardRecord::new(mesh.neurons.len(), config.mesh_steps_per_sample as usize);
        recorded_forward_pass(mesh, &mut record, config.mesh_steps_per_sample);

        // All losses combined.
        let det_labels = detection_labels_for_sample(seq, sample_idx);
        let (det_loss, det_dl_dx) = loss::detection_loss(mesh, &det_labels);

        let caus_labels = causation_labels_for_sample(seq, sample_idx);
        let (_, caus_dl_dx) = loss::causation_loss(mesh, &caus_labels);

        let mut dl_dx_output = vec![0.0f32; mesh.neurons.len()];
        for i in 0..dl_dx_output.len() {
            dl_dx_output[i] = det_dl_dx[i] + caus_dl_dx[i];
        }

        let mut grads = GradientAccumulator::from_mesh(mesh);
        backprop_through_time(mesh, &record, &dl_dx_output, &mut grads);

        // Add stability and sparsity gradients for masks.
        for (zone_idx, zone) in mesh.overlap_zones.iter().enumerate() {
            if zone_idx >= grads.mask_grads.len() || zone_idx >= mask_refs.len() {
                continue;
            }
            for (i, &mask) in zone.masks.iter().enumerate() {
                if i < grads.mask_grads[zone_idx].len() && i < mask_refs[zone_idx].len() {
                    grads.mask_grads[zone_idx][i] +=
                        config.sparsity_lambda * mask.signum()
                            + crate::training::backward::backprop_stability(
                                mask,
                                mask_refs[zone_idx][i],
                                config.stability_lambda,
                            );
                }
            }
        }

        grads.clip_global_norm(config.max_grad_norm);
        optimizer.step(mesh, &grads, &[], false); // nothing frozen

        if step % 100 == 0 {
            let stab = loss::stability_loss(mesh, mask_refs);
            let sparse = loss::total_sparsity_loss(mesh);
            let total = det_loss + config.stability_lambda * stab + config.sparsity_lambda * sparse;
            metrics.stage3_losses.push(total);
        }
    }
}

/// Evaluate per-circuit detection accuracy on a dataset subset.
///
/// For each sample, runs the mesh forward and checks whether each detection
/// circuit's readout correctly classifies the anomaly. Returns the fraction
/// of correct per-circuit predictions.
pub fn evaluate_detection_accuracy(
    mesh: &mut OverlappingMesh,
    dataset: &[TrainingSequence],
    baselines: &Baselines,
    mesh_steps: u32,
    max_samples: usize,
) -> f32 {
    let mut correct = 0u32;
    let mut total = 0u32;
    let mut count = 0;

    for seq in dataset {
        for (sample_idx, wits) in seq.wits_samples.iter().enumerate() {
            if count >= max_samples {
                break;
            }

            encode_input(mesh, wits, baselines);
            for _ in 0..mesh_steps {
                mesh.step();
            }

            let det_labels = detection_labels_for_sample(seq, sample_idx);
            for &(circuit_idx, y_true) in &det_labels {
                if circuit_idx < mesh.circuits.len() {
                    let y_pred = loss::detection_readout(mesh, circuit_idx);
                    if (y_pred > 0.5) == (y_true > 0.5) {
                        correct += 1;
                    }
                    total += 1;
                }
            }
            count += 1;
        }
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

/// Post-training: prune masks, freeze circuits, validate.
fn post_training_freeze(mesh: &mut OverlappingMesh, prune_threshold: f32) {
    // Prune overlap neurons with mask < threshold.
    for zone in &mut mesh.overlap_zones {
        let mut to_remove = Vec::new();
        for (i, &mask) in zone.masks.iter().enumerate() {
            if mask < prune_threshold {
                to_remove.push(i);
            }
        }
        // Remove in reverse order to preserve indices.
        for &i in to_remove.iter().rev() {
            zone.shared_neuron_indices.remove(i);
            zone.masks.remove(i);
        }
    }

    // Freeze all universal circuits (detection, causation, base prediction).
    for circuit in &mut mesh.circuits {
        match circuit.circuit_type {
            crate::circuit::CircuitType::Detection
            | crate::circuit::CircuitType::Causation
            | crate::circuit::CircuitType::Prediction => {
                circuit.frozen = true;
            }
            _ => {}
        }
    }
}

/// Run the complete three-stage training pipeline.
pub fn run_full_training(
    mesh: &mut OverlappingMesh,
    config: &TrainingConfig,
) -> TrainingMetrics {
    let mut metrics = TrainingMetrics::default();

    // Step 0: Initialize connections.
    initialize_connections(mesh, config.seed);

    // Generate synthetic training data.
    let mut gen = SyntheticWellGenerator::new(config.seed);
    let dataset = gen.generate_dataset(config.synthetic_sequences, config.sequence_length);

    // Build baselines from the first few sequences.
    let mut baselines = Baselines::new();
    for seq in dataset.iter().take(10) {
        for wits in &seq.wits_samples {
            baselines.update(wits);
        }
    }

    // Stage 1: Overlap pre-training.
    run_stage1(mesh, config, &dataset, &baselines, &mut metrics);

    // Save mask references for stability loss.
    let mask_refs: Vec<Vec<f32>> = mesh
        .overlap_zones
        .iter()
        .map(|z| z.masks.clone())
        .collect();

    // Stage 2: Alternating freeze.
    run_stage2(mesh, config, &dataset, &baselines, &mut metrics);

    // Stage 3: Joint fine-tuning.
    run_stage3(mesh, config, &dataset, &baselines, &mask_refs, &mut metrics);

    // Post-training freeze.
    post_training_freeze(mesh, mesh.config.mask_prune_threshold);

    // Evaluate final detection accuracy.
    metrics.final_detection_accuracy =
        evaluate_detection_accuracy(mesh, &dataset, &baselines, config.mesh_steps_per_sample, 100);

    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::build_default_mesh;

    #[test]
    fn recorded_forward_pass_records_steps() {
        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        let mut baselines = Baselines::new();
        let wits = crate::io::wits::WitsSnapshot::zeros();
        baselines.update(&wits);
        encode_input(&mut mesh, &wits, &baselines);

        let mut record = ForwardRecord::new(mesh.neurons.len(), 20);
        recorded_forward_pass(&mut mesh, &mut record, 20);

        assert_eq!(record.step_count, 20);
        // At least some neurons should have snapshots across all steps.
        let has_snapshots = record
            .step_snapshots
            .iter()
            .any(|step| step.iter().any(|s| s.is_some()));
        assert!(has_snapshots, "No neuron snapshots recorded in any step");
    }

    #[test]
    fn detection_loss_decreases_with_training() {
        use crate::circuit::CircuitType;

        let mut mesh = build_default_mesh();
        initialize_connections(&mut mesh, 42);

        // Generate a small dataset with mixed anomaly types.
        let mut gen = SyntheticWellGenerator::new(42);
        let dataset = gen.generate_dataset(12, 50);

        // Build baselines from the first few sequences.
        let mut baselines = Baselines::new();
        for seq in dataset.iter().take(6) {
            for wits in &seq.wits_samples {
                baselines.update(wits);
            }
        }

        // Only train detection circuits; freeze everything else.
        let detection_ids: Vec<u32> = mesh
            .circuits
            .iter()
            .filter(|c| c.circuit_type == CircuitType::Detection)
            .map(|c| c.id)
            .collect();
        let freeze: Vec<u32> = mesh
            .circuits
            .iter()
            .map(|c| c.id)
            .filter(|id| !detection_ids.contains(id))
            .collect();

        let mut optimizer = AdamOptimizer::from_mesh(&mesh, 1e-3);
        let mut rng = StdRng::seed_from_u64(42);
        let mut losses = Vec::new();

        for _step in 0..100 {
            let seq_idx = rand::Rng::gen_range(&mut rng, 0..dataset.len());
            let seq = &dataset[seq_idx];
            let sample_idx = rand::Rng::gen_range(&mut rng, 0..seq.wits_samples.len());

            encode_input(&mut mesh, &seq.wits_samples[sample_idx], &baselines);

            let mut record = ForwardRecord::new(mesh.neurons.len(), 50);
            recorded_forward_pass(&mut mesh, &mut record, 50);

            let det_labels = detection_labels_for_sample(seq, sample_idx);
            let (det_loss, det_dl_dx) = loss::detection_loss(&mesh, &det_labels);

            losses.push(det_loss);

            let mut grads = GradientAccumulator::from_mesh(&mesh);
            backprop_through_time(&mesh, &record, &det_dl_dx, &mut grads);
            grads.clip_global_norm(1.0);
            optimizer.step(&mut mesh, &grads, &freeze, true);
        }

        // All losses should be finite (no NaN/Inf).
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "All losses should be finite"
        );

        // Loss should decrease: compare first 10 vs last 10 steps.
        let early_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let late_avg: f32 = losses[90..].iter().sum::<f32>() / 10.0;
        assert!(
            late_avg < early_avg,
            "Detection loss should decrease: early_avg={early_avg:.4}, late_avg={late_avg:.4}"
        );
    }

    #[test]
    #[ignore] // Takes ~30s. Run with: cargo test full_pipeline_learns -- --ignored
    fn full_pipeline_learns() {
        let config = TrainingConfig {
            stage1_steps: 50,
            stage2_rounds: 2,
            stage2_steps_per_round: 100,
            stage2_memory_steps: 30,
            stage2_prediction_steps: 30,
            stage3_steps: 100,
            mesh_steps_per_sample: 50,
            synthetic_sequences: 12,
            sequence_length: 50,
            ..TrainingConfig::default()
        };

        let mut mesh = build_default_mesh();
        let metrics = run_full_training(&mut mesh, &config);

        // Stage 2 detection losses should trend downward (or at least not explode).
        if metrics.stage2_detection_losses.len() >= 4 {
            let mid = metrics.stage2_detection_losses.len() / 2;
            let first_half: f32 =
                metrics.stage2_detection_losses[..mid].iter().sum::<f32>() / mid as f32;
            let second_half: f32 = metrics.stage2_detection_losses[mid..].iter().sum::<f32>()
                / (metrics.stage2_detection_losses.len() - mid) as f32;
            assert!(
                second_half <= first_half * 1.5,
                "Stage 2 detection loss should not explode: first_half={first_half:.4}, second_half={second_half:.4}"
            );
        }

        // Stage 3 losses should be finite.
        assert!(
            metrics.stage3_losses.iter().all(|l| l.is_finite()),
            "Stage 3 losses should all be finite"
        );

        // Final detection accuracy should be above chance (50%).
        assert!(
            metrics.final_detection_accuracy > 0.5,
            "Detection accuracy should be above chance: {:.1}%",
            metrics.final_detection_accuracy * 100.0
        );
    }

    #[test]
    fn full_pipeline_smoke_test() {
        // Tiny config for fast testing.
        let config = TrainingConfig {
            stage1_steps: 5,
            stage2_rounds: 1,
            stage2_steps_per_round: 3,
            stage2_memory_steps: 2,
            stage2_prediction_steps: 2,
            stage3_steps: 3,
            mesh_steps_per_sample: 10,
            synthetic_sequences: 6,
            sequence_length: 20,
            ..TrainingConfig::default()
        };

        let mut mesh = build_default_mesh();
        let _metrics = run_full_training(&mut mesh, &config);

        // Check that training ran.
        assert!(mesh.is_healthy(), "Mesh should be healthy after training");

        // Check that circuits are frozen after training.
        let frozen_count = mesh.circuits.iter().filter(|c| c.frozen).count();
        assert!(frozen_count > 0, "Some circuits should be frozen after training");

        // Neurons should have connections.
        let connected = mesh.neurons.iter().filter(|n| !n.connections.is_empty()).count();
        assert!(connected > 100, "Neurons should have connections: {connected}");
    }
}

//! Run AASSP inference on real WITS CSV data.
//!
//! Usage: cargo run --release --bin infer_csv -- dataset/F-5_witsml.csv

use std::env;
use std::fs;

use sairen_mesh::adaptation::production::ProductionMesh;
use sairen_mesh::io::wits::WitsSnapshot;
use sairen_mesh::layout::build_default_mesh;
use sairen_mesh::physics::{compute_d_exponent, compute_ecd, compute_mse};
use sairen_mesh::training::pipeline::{run_full_training, TrainingConfig};

// ── Unit conversions (metric → US oilfield) ──

const KKGF_TO_KLBS: f32 = 2.20462;
const KNM_TO_FTLBS: f32 = 737.562;
const M_TO_FT: f32 = 3.28084;
const KPA_TO_PSI: f32 = 0.145038;
const LMIN_TO_GPM: f32 = 0.264172;
const M3_TO_BBL: f32 = 6.28981;
const GCM3_TO_PPG: f32 = 8.33;
const BIT_DIAMETER_IN: f32 = 12.25;
const NORMAL_MUD_WEIGHT_PPG: f32 = 8.33;

/// CSV column indices (from the F-5_witsml.csv header).
mod col {
    pub const TIMESTAMP: usize = 0;
    pub const BIT_DEPTH_M: usize = 1;
    // pub const TOTAL_DEPTH_M: usize = 2;
    pub const WOB_KKGF: usize = 3;
    pub const TORQUE_KNM: usize = 4;
    pub const RPM: usize = 5;
    pub const ROP_MH: usize = 6;
    pub const SPP_KPA: usize = 7;
    pub const HOOKLOAD_KKGF: usize = 8;
    pub const FLOW_IN_LMIN: usize = 9;
    pub const MUD_DENSITY_IN: usize = 10;
    pub const MUD_DENSITY_OUT: usize = 11;
    pub const ECD_GCM3: usize = 12;
    pub const TEMP_IN_C: usize = 13;
    pub const TEMP_OUT_C: usize = 14;
    pub const GAS_PCT: usize = 15;
    // pub const TOTAL_SPM: usize = 16;
    pub const TANK_VOLUME_M3: usize = 17;
    pub const RIG_MODE: usize = 18;
    // pub const BLOCK_POS: usize = 19;
}

/// Parsed CSV row ready for inference.
struct CsvRow {
    timestamp: String,
    rig_mode: String,
    wits: WitsSnapshot,
}

/// Parse a float from a CSV field, returning None for empty/unparseable values.
fn parse_f32(s: &str) -> Option<f32> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f32>().ok()
}

/// Convert Celsius to Fahrenheit.
fn c_to_f(c: f32) -> f32 {
    c * 1.8 + 32.0
}

/// Parse a single CSV row into a CsvRow, returning None if key fields are missing.
fn parse_csv_row(fields: &[&str], prev_pit_bbl: f32, dt_hours: f32) -> Option<CsvRow> {
    if fields.len() < 19 {
        return None;
    }

    let timestamp = fields[col::TIMESTAMP].trim().to_string();
    let rig_mode = fields[col::RIG_MODE].trim().to_string();

    // Parse key drilling parameters — skip rows where all are missing/zero.
    let wob_kkgf = parse_f32(fields[col::WOB_KKGF]).unwrap_or(0.0);
    let rpm = parse_f32(fields[col::RPM]).unwrap_or(0.0);
    let spp_kpa = parse_f32(fields[col::SPP_KPA]).unwrap_or(0.0);
    let flow_in_lmin = parse_f32(fields[col::FLOW_IN_LMIN]).unwrap_or(0.0);

    // Skip rows where all key parameters are zero/missing (no useful drilling data).
    if wob_kkgf.abs() < 1e-6 && rpm.abs() < 1e-6 && spp_kpa.abs() < 1e-6 && flow_in_lmin.abs() < 1e-6 {
        return None;
    }

    // Convert units.
    let wob_klbs = wob_kkgf * KKGF_TO_KLBS;
    let torque_ft_lbs = parse_f32(fields[col::TORQUE_KNM]).unwrap_or(0.0) * KNM_TO_FTLBS;
    let rop_ft_hr = parse_f32(fields[col::ROP_MH]).unwrap_or(0.0) * M_TO_FT;
    let spp_psi = spp_kpa * KPA_TO_PSI;
    let hook_load_klbs = parse_f32(fields[col::HOOKLOAD_KKGF]).unwrap_or(0.0) * KKGF_TO_KLBS;
    let flow_in_gpm = flow_in_lmin * LMIN_TO_GPM;
    let bit_depth_ft = parse_f32(fields[col::BIT_DEPTH_M]).unwrap_or(0.0) * M_TO_FT;

    // Mud properties.
    let mud_weight_in_ppg = parse_f32(fields[col::MUD_DENSITY_IN]).unwrap_or(1.03) * GCM3_TO_PPG;
    let mud_weight_out_ppg = parse_f32(fields[col::MUD_DENSITY_OUT]).unwrap_or(1.02) * GCM3_TO_PPG;
    let mud_temp_in_f = c_to_f(parse_f32(fields[col::TEMP_IN_C]).unwrap_or(20.0));
    let mud_temp_out_f = c_to_f(parse_f32(fields[col::TEMP_OUT_C]).unwrap_or(20.0));

    // Gas.
    let total_gas_pct = parse_f32(fields[col::GAS_PCT]).unwrap_or(0.0);
    let h2s_ppm = 0.0; // Not in CSV.

    // Pit volume.
    let pit_volume_bbl = parse_f32(fields[col::TANK_VOLUME_M3]).unwrap_or(0.0) * M3_TO_BBL;

    // Estimate flow_out as flow_in * 0.98 (flow_out not in CSV).
    let flow_out_gpm = flow_in_gpm * 0.98;
    let flow_balance_gpm = flow_out_gpm - flow_in_gpm;

    // Pit rate (bbl/hr) from delta pit volume.
    let pit_rate_bbl_hr = if dt_hours > 1e-6 {
        (pit_volume_bbl - prev_pit_bbl) / dt_hours
    } else {
        0.0
    };

    // ECD: use CSV column if present, else compute from physics, else fallback.
    let ecd_ppg = parse_f32(fields[col::ECD_GCM3])
        .map(|v| v * GCM3_TO_PPG)
        .or_else(|| compute_ecd(mud_weight_in_ppg, spp_psi * 0.1, bit_depth_ft))
        .unwrap_or(mud_weight_in_ppg);

    // MSE.
    let mse_psi = compute_mse(
        wob_klbs * 1000.0, // klbs → lbs
        rpm,
        torque_ft_lbs,
        rop_ft_hr,
        BIT_DIAMETER_IN,
    )
    .unwrap_or(0.0);

    // d-exponent.
    let d_exponent = compute_d_exponent(
        rop_ft_hr,
        rpm,
        wob_klbs,
        BIT_DIAMETER_IN,
        NORMAL_MUD_WEIGHT_PPG,
        mud_weight_in_ppg,
    )
    .unwrap_or(0.0);

    Some(CsvRow {
        timestamp,
        rig_mode,
        wits: WitsSnapshot {
            wob_klbs,
            rop_ft_hr,
            rpm,
            torque_ft_lbs,
            hook_load_klbs,
            spp_psi,
            flow_in_gpm,
            flow_out_gpm,
            pit_volume_bbl,
            bit_depth_ft,
            mud_weight_in_ppg,
            mud_weight_out_ppg,
            mud_temp_in_f,
            mud_temp_out_f,
            total_gas_pct,
            h2s_ppm,
            mse_psi,
            d_exponent,
            ecd_ppg,
            flow_balance_gpm,
            pit_rate_bbl_hr,
        },
    })
}

/// Parse the entire CSV file, skipping the header and invalid rows.
fn parse_csv(path: &str) -> Vec<CsvRow> {
    let contents = fs::read_to_string(path).expect("Failed to read CSV file");
    let mut rows = Vec::new();
    let mut prev_pit_bbl: f32 = 0.0;
    let dt_hours: f32 = 5.0 / 3600.0; // ~5 second sample interval

    for (line_num, line) in contents.lines().enumerate() {
        if line_num == 0 {
            continue; // Skip header.
        }

        let fields: Vec<&str> = line.split(',').collect();
        if let Some(row) = parse_csv_row(&fields, prev_pit_bbl, dt_hours) {
            prev_pit_bbl = row.wits.pit_volume_bbl;
            rows.push(row);
        }
    }

    rows
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let csv_path = args.get(1).map(|s| s.as_str()).unwrap_or("dataset/F-5_witsml.csv");

    // ── Step 1: Parse CSV ──
    eprintln!("Parsing CSV: {csv_path}");
    let rows = parse_csv(csv_path);
    eprintln!("Parsed {} valid rows (with non-zero drilling data)\n", rows.len());

    if rows.is_empty() {
        eprintln!("No valid rows found. Exiting.");
        return;
    }

    // Print sample ranges for sanity check.
    eprintln!("=== DATA RANGES (first 500 valid rows) ===");
    let check = &rows[..rows.len().min(500)];
    let wob_range = (
        check.iter().map(|r| r.wits.wob_klbs).fold(f32::MAX, f32::min),
        check.iter().map(|r| r.wits.wob_klbs).fold(f32::MIN, f32::max),
    );
    let rpm_range = (
        check.iter().map(|r| r.wits.rpm).fold(f32::MAX, f32::min),
        check.iter().map(|r| r.wits.rpm).fold(f32::MIN, f32::max),
    );
    let spp_range = (
        check.iter().map(|r| r.wits.spp_psi).fold(f32::MAX, f32::min),
        check.iter().map(|r| r.wits.spp_psi).fold(f32::MIN, f32::max),
    );
    let flow_range = (
        check.iter().map(|r| r.wits.flow_in_gpm).fold(f32::MAX, f32::min),
        check.iter().map(|r| r.wits.flow_in_gpm).fold(f32::MIN, f32::max),
    );
    let depth_range = (
        check.iter().map(|r| r.wits.bit_depth_ft).fold(f32::MAX, f32::min),
        check.iter().map(|r| r.wits.bit_depth_ft).fold(f32::MIN, f32::max),
    );
    eprintln!("  WOB (klbs):      {:.1} - {:.1}", wob_range.0, wob_range.1);
    eprintln!("  RPM:             {:.1} - {:.1}", rpm_range.0, rpm_range.1);
    eprintln!("  SPP (psi):       {:.1} - {:.1}", spp_range.0, spp_range.1);
    eprintln!("  Flow In (gpm):   {:.1} - {:.1}", flow_range.0, flow_range.1);
    eprintln!("  Bit Depth (ft):  {:.1} - {:.1}", depth_range.0, depth_range.1);
    eprintln!();

    // ── Step 2: Train model on synthetic data ──
    eprintln!("Training on synthetic data...");
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

    let mut mesh = build_default_mesh();
    let metrics = run_full_training(&mut mesh, &config);
    eprintln!(
        "Training complete. Detection accuracy: {:.1}%\n",
        metrics.final_detection_accuracy * 100.0
    );

    // ── Step 3: Wrap in ProductionMesh ──
    let mut prod = ProductionMesh::from_trained(mesh);

    // ── Step 4: Warm up baselines from first 500 valid samples ──
    let warmup_count = rows.len().min(500);
    eprintln!("Warming up baselines from first {warmup_count} samples...");
    for row in &rows[..warmup_count] {
        prod.baselines.update(&row.wits);
    }
    eprintln!(
        "Baseline WOB: mean={:.1}, std={:.1}",
        prod.baselines.wob.mean, prod.baselines.wob.std_dev
    );
    eprintln!(
        "Baseline RPM: mean={:.1}, std={:.1}",
        prod.baselines.rpm.mean, prod.baselines.rpm.std_dev
    );
    eprintln!(
        "Baseline SPP: mean={:.1}, std={:.1}\n",
        prod.baselines.spp.mean, prod.baselines.spp.std_dev
    );

    // ── Step 5: Run inference ──
    eprintln!("Running inference on {} samples...\n", rows.len());

    let mut total_detections = 0usize;
    let mut detection_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut sustained_windows: Vec<(String, String, usize)> = Vec::new(); // (start_ts, end_ts, count)
    let mut current_window_start: Option<String> = None;
    let mut current_window_count = 0usize;

    for (i, row) in rows.iter().enumerate() {
        let output = prod.process(&row.wits);

        if !output.detections.is_empty() {
            total_detections += 1;

            // Print detection.
            let det_strs: Vec<String> = output
                .detections
                .iter()
                .map(|d| format!("{:?} ({:.1}%)", d.anomaly_type, d.confidence * 100.0))
                .collect();

            println!(
                "[{:>6}] {} | {:>12} | {:?} | {}",
                i,
                row.timestamp,
                row.rig_mode,
                output.risk_level,
                det_strs.join(", ")
            );

            for d in &output.detections {
                *detection_counts
                    .entry(format!("{:?}", d.anomaly_type))
                    .or_insert(0) += 1;
            }

            // Track sustained windows.
            if current_window_start.is_none() {
                current_window_start = Some(row.timestamp.clone());
            }
            current_window_count += 1;
        } else if current_window_count > 0 {
            // Window ended.
            if current_window_count >= 3 {
                sustained_windows.push((
                    current_window_start.take().unwrap(),
                    rows[i - 1].timestamp.clone(),
                    current_window_count,
                ));
            } else {
                current_window_start = None;
            }
            current_window_count = 0;
        }
    }

    // Close any open window.
    if current_window_count >= 3 {
        if let Some(start) = current_window_start {
            sustained_windows.push((
                start,
                rows.last().unwrap().timestamp.clone(),
                current_window_count,
            ));
        }
    }

    // ── Step 6: Summary ──
    eprintln!("\n========================================");
    eprintln!("           INFERENCE SUMMARY");
    eprintln!("========================================");
    eprintln!("Total samples processed:  {}", rows.len());
    eprintln!("Samples with detections:  {}", total_detections);
    eprintln!(
        "Detection rate:           {:.2}%",
        100.0 * total_detections as f64 / rows.len() as f64
    );

    if !detection_counts.is_empty() {
        eprintln!("\nDetection breakdown:");
        let mut sorted: Vec<_> = detection_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in &sorted {
            eprintln!("  {:<20} {}", name, count);
        }
    }

    if !sustained_windows.is_empty() {
        eprintln!(
            "\nSustained anomaly windows (3+ consecutive detections):"
        );
        for (start, end, count) in &sustained_windows {
            eprintln!("  {} → {} ({} samples)", start, end, count);
        }
    }

    if total_detections == 0 {
        eprintln!("\nNo anomalies detected in this dataset.");
    }
    eprintln!("========================================");
}

//! Synthetic WITS data generator for training.
//!
//! Generates realistic drilling data with injected anomalies for offline training.

use crate::io::wits::WitsSnapshot;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Anomaly type for labelling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    Normal,
    Kick,
    Loss,
    PackOff,
    StickSlip,
    Founder,
}

/// A labelled WITS sequence for training.
#[derive(Debug, Clone)]
pub struct TrainingSequence {
    pub wits_samples: Vec<WitsSnapshot>,
    pub labels: Vec<AnomalyType>,
    pub formation: Option<String>,
    pub well_id: String,
}

/// Generates synthetic drilling WITS data.
pub struct SyntheticWellGenerator {
    rng: StdRng,
    /// Base operating point.
    pub base: BaseParams,
    /// Noise scale (fraction of base value).
    pub noise_scale: f32,
}

/// Typical operating parameters for a normal well.
#[derive(Debug, Clone)]
pub struct BaseParams {
    pub wob_klbs: f32,
    pub rop_ft_hr: f32,
    pub rpm: f32,
    pub torque_ft_lbs: f32,
    pub hook_load_klbs: f32,
    pub spp_psi: f32,
    pub flow_in_gpm: f32,
    pub flow_out_gpm: f32,
    pub pit_volume_bbl: f32,
    pub bit_depth_ft: f32,
    pub mud_weight_ppg: f32,
    pub total_gas_pct: f32,
}

impl Default for BaseParams {
    fn default() -> Self {
        Self {
            wob_klbs: 25.0,
            rop_ft_hr: 60.0,
            rpm: 120.0,
            torque_ft_lbs: 8000.0,
            hook_load_klbs: 200.0,
            spp_psi: 3000.0,
            flow_in_gpm: 400.0,
            flow_out_gpm: 400.0,
            pit_volume_bbl: 500.0,
            bit_depth_ft: 10000.0,
            mud_weight_ppg: 12.0,
            total_gas_pct: 0.5,
        }
    }
}

impl SyntheticWellGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            base: BaseParams::default(),
            noise_scale: 0.02,
        }
    }

    pub fn with_base(mut self, base: BaseParams) -> Self {
        self.base = base;
        self
    }

    /// Generate a single normal WITS snapshot with Gaussian noise.
    fn normal_snapshot(&mut self) -> WitsSnapshot {
        // Clone base params to avoid borrow conflict with self.gaussian().
        let b = self.base.clone();
        let ns = self.noise_scale;

        let wob = b.wob_klbs * (1.0 + self.gaussian() * ns);
        let rop = (b.rop_ft_hr * (1.0 + self.gaussian() * ns)).max(0.1);
        let rpm = (b.rpm * (1.0 + self.gaussian() * ns)).max(1.0);
        let torque = (b.torque_ft_lbs * (1.0 + self.gaussian() * ns)).max(0.0);
        let spp = (b.spp_psi * (1.0 + self.gaussian() * ns)).max(0.0);
        let flow_in = (b.flow_in_gpm * (1.0 + self.gaussian() * ns)).max(0.0);
        let flow_out = (b.flow_out_gpm * (1.0 + self.gaussian() * ns * 0.5)).max(0.0);
        let pit_vol = (b.pit_volume_bbl * (1.0 + self.gaussian() * ns * 0.1)).max(0.0);
        let gas = (b.total_gas_pct + self.gaussian() * 0.1).max(0.0);

        let flow_balance = flow_out - flow_in;
        let mse = if rop > 0.1 { 35000.0 + self.gaussian() * 5000.0 } else { 0.0 };

        WitsSnapshot {
            wob_klbs: wob,
            rop_ft_hr: rop,
            rpm,
            torque_ft_lbs: torque,
            hook_load_klbs: b.hook_load_klbs * (1.0 + self.gaussian() * ns),
            spp_psi: spp,
            flow_in_gpm: flow_in,
            flow_out_gpm: flow_out,
            pit_volume_bbl: pit_vol,
            bit_depth_ft: b.bit_depth_ft,
            mud_weight_in_ppg: b.mud_weight_ppg,
            mud_weight_out_ppg: b.mud_weight_ppg,
            mud_temp_in_f: 80.0 + self.gaussian() * 2.0,
            mud_temp_out_f: 120.0 + self.gaussian() * 3.0,
            total_gas_pct: gas,
            h2s_ppm: 0.0,
            mse_psi: mse.max(0.0),
            d_exponent: 1.5 + self.gaussian() * 0.1,
            ecd_ppg: b.mud_weight_ppg + 0.3 + self.gaussian() * 0.05,
            flow_balance_gpm: flow_balance,
            pit_rate_bbl_hr: self.gaussian() * 0.5,
        }
    }

    /// Generate a normal drilling sequence.
    pub fn normal_sequence(&mut self, length: usize) -> TrainingSequence {
        let wits: Vec<WitsSnapshot> = (0..length).map(|_| self.normal_snapshot()).collect();
        let labels = vec![AnomalyType::Normal; length];
        TrainingSequence {
            wits_samples: wits,
            labels,
            formation: Some("sandstone".into()),
            well_id: "synthetic_well".into(),
        }
    }

    /// Generate a sequence with a kick event injected.
    pub fn kick_sequence(&mut self, length: usize, onset: usize, severity: f32) -> TrainingSequence {
        let mut seq = self.normal_sequence(length);
        let end = (onset + 30).min(length);

        for i in onset..end {
            let progress = (i - onset) as f32 / (end - onset) as f32;
            let ramp = progress * severity;

            seq.wits_samples[i].flow_out_gpm += ramp * 30.0;
            seq.wits_samples[i].flow_balance_gpm += ramp * 30.0;
            seq.wits_samples[i].pit_volume_bbl += ramp * 2.0;
            seq.wits_samples[i].total_gas_pct += ramp * 3.0;
            seq.wits_samples[i].spp_psi -= ramp * 100.0;
            seq.labels[i] = AnomalyType::Kick;
        }

        seq
    }

    /// Generate a sequence with a mud loss event.
    pub fn loss_sequence(&mut self, length: usize, onset: usize, severity: f32) -> TrainingSequence {
        let mut seq = self.normal_sequence(length);
        let end = (onset + 25).min(length);

        for i in onset..end {
            let progress = (i - onset) as f32 / (end - onset) as f32;
            let ramp = progress * severity;

            seq.wits_samples[i].flow_out_gpm -= ramp * 25.0;
            seq.wits_samples[i].flow_balance_gpm -= ramp * 25.0;
            seq.wits_samples[i].pit_volume_bbl -= ramp * 3.0;
            seq.wits_samples[i].spp_psi += ramp * 150.0;
            seq.labels[i] = AnomalyType::Loss;
        }

        seq
    }

    /// Generate a sequence with a pack-off event.
    pub fn packoff_sequence(&mut self, length: usize, onset: usize, severity: f32) -> TrainingSequence {
        let mut seq = self.normal_sequence(length);
        let end = (onset + 15).min(length);

        for i in onset..end {
            let progress = (i - onset) as f32 / (end - onset) as f32;
            let ramp = progress * severity;

            seq.wits_samples[i].torque_ft_lbs *= 1.0 + ramp * 0.8;
            seq.wits_samples[i].spp_psi *= 1.0 + ramp * 0.4;
            seq.wits_samples[i].rop_ft_hr *= 1.0 - ramp * 0.9;
            seq.labels[i] = AnomalyType::PackOff;
        }

        seq
    }

    /// Generate a sequence with stick-slip.
    pub fn stickslip_sequence(&mut self, length: usize, onset: usize, severity: f32) -> TrainingSequence {
        let mut seq = self.normal_sequence(length);
        let end = (onset + 20).min(length);

        for i in onset..end {
            let ramp = severity;
            let phase = (i as f32 * 2.0).sin();

            seq.wits_samples[i].torque_ft_lbs *= 1.0 + phase * ramp * 0.3;
            seq.wits_samples[i].rpm *= 1.0 + phase * ramp * 0.2;
            seq.labels[i] = AnomalyType::StickSlip;
        }

        seq
    }

    /// Generate a sequence with founder condition.
    pub fn founder_sequence(&mut self, length: usize, onset: usize, severity: f32) -> TrainingSequence {
        let mut seq = self.normal_sequence(length);
        let end = (onset + 40).min(length);

        for i in onset..end {
            let progress = (i - onset) as f32 / (end - onset) as f32;
            let ramp = progress * severity;

            seq.wits_samples[i].wob_klbs *= 1.0 + ramp * 0.5;
            // ROP plateaus despite increasing WOB.
            seq.wits_samples[i].rop_ft_hr *= 1.0 - ramp * 0.3;
            seq.wits_samples[i].mse_psi *= 1.0 + ramp * 1.5;
            seq.labels[i] = AnomalyType::Founder;
        }

        seq
    }

    /// Generate a mixed training dataset.
    ///
    /// Returns sequences with a mix of normal and anomalous data.
    pub fn generate_dataset(
        &mut self,
        n_sequences: usize,
        seq_length: usize,
    ) -> Vec<TrainingSequence> {
        let mut sequences = Vec::with_capacity(n_sequences);

        for i in 0..n_sequences {
            let severity = 0.5 + self.rng.gen::<f32>() * 0.5; // 0.5-1.0
            let onset = seq_length / 3 + self.rng.gen_range(0..seq_length / 3);

            let seq = match i % 6 {
                0 => self.normal_sequence(seq_length),
                1 => self.kick_sequence(seq_length, onset, severity),
                2 => self.loss_sequence(seq_length, onset, severity),
                3 => self.packoff_sequence(seq_length, onset, severity),
                4 => self.stickslip_sequence(seq_length, onset, severity),
                5 => self.founder_sequence(seq_length, onset, severity),
                _ => unreachable!(),
            };

            sequences.push(seq);
        }

        sequences
    }

    /// Box-Muller transform for Gaussian noise.
    fn gaussian(&mut self) -> f32 {
        let u1: f32 = self.rng.gen::<f32>().max(1e-10);
        let u2: f32 = self.rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_sequence_all_normal() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.normal_sequence(100);
        assert_eq!(seq.wits_samples.len(), 100);
        assert!(seq.labels.iter().all(|l| *l == AnomalyType::Normal));
    }

    #[test]
    fn kick_sequence_has_anomaly_labels() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.kick_sequence(100, 30, 0.8);
        let n_kick = seq.labels.iter().filter(|l| **l == AnomalyType::Kick).count();
        assert!(n_kick > 0, "Should have kick labels");
    }

    #[test]
    fn kick_increases_flow_balance() {
        let mut gen = SyntheticWellGenerator::new(42);
        let seq = gen.kick_sequence(100, 30, 1.0);
        // After onset, flow_balance should be notably positive.
        let fb_before: f32 = seq.wits_samples[..30]
            .iter()
            .map(|w| w.flow_balance_gpm)
            .sum::<f32>()
            / 30.0;
        let fb_after: f32 = seq.wits_samples[50..60]
            .iter()
            .map(|w| w.flow_balance_gpm)
            .sum::<f32>()
            / 10.0;
        assert!(fb_after > fb_before + 5.0, "Kick should increase flow balance");
    }

    #[test]
    fn dataset_has_mixed_types() {
        let mut gen = SyntheticWellGenerator::new(42);
        let dataset = gen.generate_dataset(12, 100);
        assert_eq!(dataset.len(), 12);

        // Should have at least one of each type.
        let has_normal = dataset.iter().any(|s| s.labels.iter().all(|l| *l == AnomalyType::Normal));
        let has_kick = dataset.iter().any(|s| s.labels.iter().any(|l| *l == AnomalyType::Kick));
        assert!(has_normal);
        assert!(has_kick);
    }

    #[test]
    fn all_wits_values_finite() {
        let mut gen = SyntheticWellGenerator::new(42);
        let dataset = gen.generate_dataset(6, 50);
        for seq in &dataset {
            for w in &seq.wits_samples {
                let features = w.as_feature_array();
                for &f in &features {
                    assert!(f.is_finite(), "Non-finite WITS value: {f}");
                }
            }
        }
    }
}

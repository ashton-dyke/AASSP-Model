//! Input encoding: normalise WITS data into neuron activations.

use crate::circuit::CircuitType;
use crate::io::wits::WitsSnapshot;
use crate::mesh::OverlappingMesh;
use serde::{Deserialize, Serialize};

/// Running statistics for a single parameter, used for normalisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sample_count: u64,
    // Welford's online algorithm accumulators.
    m2: f64,
    running_mean: f64,
}

impl BaselineStats {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0, // Avoid division by zero before first update.
            min: f32::MAX,
            max: f32::MIN,
            sample_count: 0,
            m2: 0.0,
            running_mean: 0.0,
        }
    }

    /// Update with a new sample using Welford's online algorithm.
    pub fn update(&mut self, value: f32) {
        let val = value as f64;
        self.sample_count += 1;
        let n = self.sample_count as f64;

        let delta = val - self.running_mean;
        self.running_mean += delta / n;
        let delta2 = val - self.running_mean;
        self.m2 += delta * delta2;

        self.mean = self.running_mean as f32;
        if self.sample_count > 1 {
            self.std_dev = (self.m2 / (n - 1.0)).sqrt() as f32;
        }

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }
}

impl Default for BaselineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of baseline statistics for all 14 input parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baselines {
    pub wob: BaselineStats,
    pub rop: BaselineStats,
    pub rpm: BaselineStats,
    pub torque: BaselineStats,
    pub spp: BaselineStats,
    pub flow_in: BaselineStats,
    pub flow_out: BaselineStats,
    pub pit_volume: BaselineStats,
    pub mse: BaselineStats,
    pub d_exponent: BaselineStats,
    pub ecd: BaselineStats,
    pub flow_balance: BaselineStats,
    pub total_gas: BaselineStats,
    pub h2s: BaselineStats,
}

impl Baselines {
    pub fn new() -> Self {
        Self {
            wob: BaselineStats::new(),
            rop: BaselineStats::new(),
            rpm: BaselineStats::new(),
            torque: BaselineStats::new(),
            spp: BaselineStats::new(),
            flow_in: BaselineStats::new(),
            flow_out: BaselineStats::new(),
            pit_volume: BaselineStats::new(),
            mse: BaselineStats::new(),
            d_exponent: BaselineStats::new(),
            ecd: BaselineStats::new(),
            flow_balance: BaselineStats::new(),
            total_gas: BaselineStats::new(),
            h2s: BaselineStats::new(),
        }
    }

    /// Update all baselines from a WITS snapshot.
    pub fn update(&mut self, wits: &WitsSnapshot) {
        self.wob.update(wits.wob_klbs);
        self.rop.update(wits.rop_ft_hr);
        self.rpm.update(wits.rpm);
        self.torque.update(wits.torque_ft_lbs);
        self.spp.update(wits.spp_psi);
        self.flow_in.update(wits.flow_in_gpm);
        self.flow_out.update(wits.flow_out_gpm);
        self.pit_volume.update(wits.pit_volume_bbl);
        self.mse.update(wits.mse_psi);
        self.d_exponent.update(wits.d_exponent);
        self.ecd.update(wits.ecd_ppg);
        self.flow_balance.update(wits.flow_balance_gpm);
        self.total_gas.update(wits.total_gas_pct);
        self.h2s.update(wits.h2s_ppm);
    }

    /// Get all baselines as an array (matching WitsSnapshot::as_feature_array order).
    pub fn as_array(&self) -> [&BaselineStats; 14] {
        [
            &self.wob,
            &self.rop,
            &self.rpm,
            &self.torque,
            &self.spp,
            &self.flow_in,
            &self.flow_out,
            &self.pit_volume,
            &self.mse,
            &self.d_exponent,
            &self.ecd,
            &self.flow_balance,
            &self.total_gas,
            &self.h2s,
        ]
    }
}

impl Default for Baselines {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalise a value using baseline statistics.
/// Returns a value in approximately [-1, 1].
pub fn normalise(value: f32, baseline: &BaselineStats) -> f32 {
    if baseline.std_dev < 1e-6 {
        return 0.0;
    }
    ((value - baseline.mean) / baseline.std_dev).clamp(-3.0, 3.0) / 3.0
}

/// Encode a WITS snapshot into detection circuit input neurons.
pub fn encode_input(mesh: &mut OverlappingMesh, wits: &WitsSnapshot, baselines: &Baselines) {
    let features = wits.as_feature_array();
    let baseline_arr = baselines.as_array();

    let encoded: Vec<f32> = features
        .iter()
        .zip(baseline_arr.iter())
        .map(|(&val, bl)| normalise(val, bl))
        .collect();

    // Write to input neurons (first 14 neurons of each detection circuit).
    for circuit in &mesh.circuits {
        if circuit.circuit_type == CircuitType::Detection {
            for (i, &value) in encoded.iter().enumerate() {
                if i < circuit.neuron_indices.len() {
                    mesh.neurons[circuit.neuron_indices[i]].x = value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn baseline_stats_welford_convergence() {
        let mut stats = BaselineStats::new();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];

        for &v in &values {
            stats.update(v);
        }

        assert_abs_diff_eq!(stats.mean, 3.0, epsilon = 1e-5);
        // std_dev of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811
        assert_abs_diff_eq!(stats.std_dev, (2.5_f32).sqrt(), epsilon = 1e-4);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.sample_count, 5);
    }

    #[test]
    fn normalise_zero_variance_returns_zero() {
        let mut stats = BaselineStats::new();
        stats.std_dev = 0.0;
        assert_eq!(normalise(42.0, &stats), 0.0);
    }

    #[test]
    fn normalise_mean_returns_zero() {
        let mut stats = BaselineStats::new();
        stats.mean = 5.0;
        stats.std_dev = 2.0;
        assert_abs_diff_eq!(normalise(5.0, &stats), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn normalise_clamps_to_range() {
        let mut stats = BaselineStats::new();
        stats.mean = 0.0;
        stats.std_dev = 1.0;

        // 10 std devs away → should clamp to 1.0.
        let result = normalise(10.0, &stats);
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-6);

        let result = normalise(-10.0, &stats);
        assert_abs_diff_eq!(result, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn normalise_one_sigma() {
        let mut stats = BaselineStats::new();
        stats.mean = 0.0;
        stats.std_dev = 1.0;

        // 1 std dev → 1/3
        let result = normalise(1.0, &stats);
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn baselines_update_all_parameters() {
        let mut baselines = Baselines::new();
        let wits = WitsSnapshot {
            wob_klbs: 10.0,
            rop_ft_hr: 50.0,
            rpm: 120.0,
            torque_ft_lbs: 5000.0,
            hook_load_klbs: 200.0,
            spp_psi: 3000.0,
            flow_in_gpm: 400.0,
            flow_out_gpm: 395.0,
            pit_volume_bbl: 500.0,
            bit_depth_ft: 10000.0,
            mud_weight_in_ppg: 12.0,
            mud_weight_out_ppg: 12.1,
            mud_temp_in_f: 80.0,
            mud_temp_out_f: 120.0,
            total_gas_pct: 0.5,
            h2s_ppm: 0.0,
            mse_psi: 40000.0,
            d_exponent: 1.5,
            ecd_ppg: 12.5,
            flow_balance_gpm: -5.0,
            pit_rate_bbl_hr: -2.0,
        };

        baselines.update(&wits);
        assert_eq!(baselines.wob.sample_count, 1);
        assert_abs_diff_eq!(baselines.wob.mean, 10.0, epsilon = 1e-5);
        assert_abs_diff_eq!(baselines.rpm.mean, 120.0, epsilon = 1e-5);
    }
}

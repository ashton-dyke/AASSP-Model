//! WITS Level 0 data snapshot.

use serde::{Deserialize, Serialize};

/// A single WITS data snapshot, parsed from the raw TCP stream.
/// This is the input to the mesh at each timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitsSnapshot {
    // Primary drilling parameters
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

    // Mud properties
    pub mud_weight_in_ppg: f32,
    pub mud_weight_out_ppg: f32,
    pub mud_temp_in_f: f32,
    pub mud_temp_out_f: f32,

    // Gas monitoring
    pub total_gas_pct: f32,
    pub h2s_ppm: f32,

    // Derived physics (computed BEFORE mesh input)
    pub mse_psi: f32,
    pub d_exponent: f32,
    pub ecd_ppg: f32,
    pub flow_balance_gpm: f32,
    pub pit_rate_bbl_hr: f32,
}

impl WitsSnapshot {
    /// Create a zero-valued snapshot (useful for testing).
    pub fn zeros() -> Self {
        Self {
            wob_klbs: 0.0,
            rop_ft_hr: 0.0,
            rpm: 0.0,
            torque_ft_lbs: 0.0,
            hook_load_klbs: 0.0,
            spp_psi: 0.0,
            flow_in_gpm: 0.0,
            flow_out_gpm: 0.0,
            pit_volume_bbl: 0.0,
            bit_depth_ft: 0.0,
            mud_weight_in_ppg: 0.0,
            mud_weight_out_ppg: 0.0,
            mud_temp_in_f: 0.0,
            mud_temp_out_f: 0.0,
            total_gas_pct: 0.0,
            h2s_ppm: 0.0,
            mse_psi: 0.0,
            d_exponent: 0.0,
            ecd_ppg: 0.0,
            flow_balance_gpm: 0.0,
            pit_rate_bbl_hr: 0.0,
        }
    }

    /// Extract the 14 normalisation-target parameters as an array.
    pub fn as_feature_array(&self) -> [f32; 14] {
        [
            self.wob_klbs,
            self.rop_ft_hr,
            self.rpm,
            self.torque_ft_lbs,
            self.spp_psi,
            self.flow_in_gpm,
            self.flow_out_gpm,
            self.pit_volume_bbl,
            self.mse_psi,
            self.d_exponent,
            self.ecd_ppg,
            self.flow_balance_gpm,
            self.total_gas_pct,
            self.h2s_ppm,
        ]
    }
}

impl Default for WitsSnapshot {
    fn default() -> Self {
        Self::zeros()
    }
}

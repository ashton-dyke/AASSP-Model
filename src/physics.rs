//! Physics ground-truth calculations.
//!
//! These are computed BEFORE the mesh receives data. The mesh operates on
//! already-computed physics values. These functions are provided for reference
//! and for use in the WITS pre-processing pipeline.

use std::f32::consts::PI;

/// Compute Mechanical Specific Energy (MSE) in psi.
///
/// MSE = (WOB / A_bit) + (120 * π * RPM * Torque) / (A_bit * ROP)
///
/// Returns None if inputs are invalid (zero ROP, zero bit diameter).
pub fn compute_mse(
    wob_lbs: f32,
    rpm: f32,
    torque_ft_lbs: f32,
    rop_ft_hr: f32,
    bit_diameter_in: f32,
) -> Option<f32> {
    if rop_ft_hr.abs() < 1e-6 || bit_diameter_in.abs() < 1e-6 {
        return None;
    }

    let a_bit = PI * (bit_diameter_in / 2.0).powi(2);
    let mse = (wob_lbs / a_bit) + (120.0 * PI * rpm * torque_ft_lbs) / (a_bit * rop_ft_hr);
    Some(mse)
}

/// Compute d-exponent (corrected for mud weight).
///
/// d_exp = log10(ROP / (60 * RPM)) / log10(12 * WOB / (1000 * bit_diameter))
/// d_exp_corrected = d_exp * (mud_weight_normal / mud_weight_actual)
///
/// Returns None if inputs would cause division by zero or log of non-positive.
pub fn compute_d_exponent(
    rop_ft_hr: f32,
    rpm: f32,
    wob_klbs: f32,
    bit_diameter_in: f32,
    mud_weight_normal_ppg: f32,
    mud_weight_actual_ppg: f32,
) -> Option<f32> {
    if rpm.abs() < 1e-6 || bit_diameter_in.abs() < 1e-6 || mud_weight_actual_ppg.abs() < 1e-6 {
        return None;
    }

    let numerator_arg = rop_ft_hr / (60.0 * rpm);
    let denominator_arg = 12.0 * wob_klbs / (1.0 * bit_diameter_in);

    if numerator_arg <= 0.0 || denominator_arg <= 0.0 || denominator_arg == 1.0 {
        return None;
    }

    let d_exp = numerator_arg.log10() / denominator_arg.log10();
    let d_exp_corrected = d_exp * (mud_weight_normal_ppg / mud_weight_actual_ppg);

    if d_exp_corrected.is_finite() {
        Some(d_exp_corrected)
    } else {
        None
    }
}

/// Compute Equivalent Circulating Density (ECD) in ppg.
///
/// ECD = mud_weight + (annular_pressure_loss / (0.052 * TVD))
///
/// Returns None if TVD is zero.
pub fn compute_ecd(
    mud_weight_ppg: f32,
    annular_pressure_loss_psi: f32,
    tvd_ft: f32,
) -> Option<f32> {
    if tvd_ft.abs() < 1e-6 {
        return None;
    }

    let ecd = mud_weight_ppg + (annular_pressure_loss_psi / (0.052 * tvd_ft));
    Some(ecd)
}

/// Compute flow balance in gpm.
///
/// Positive = potential kick, negative = potential loss.
pub fn compute_flow_balance(flow_out_gpm: f32, flow_in_gpm: f32) -> f32 {
    flow_out_gpm - flow_in_gpm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_typical_values() {
        // Typical drilling: 12.25" bit, 30klbs WOB, 120 RPM, 15klbs torque, 60 ft/hr
        let mse = compute_mse(30_000.0, 120.0, 15_000.0, 60.0, 12.25).unwrap();
        // MSE should be in the tens of thousands of psi range.
        assert!(mse > 10_000.0 && mse < 200_000.0, "MSE = {mse}");
    }

    #[test]
    fn mse_zero_rop_returns_none() {
        assert!(compute_mse(30_000.0, 120.0, 15_000.0, 0.0, 12.25).is_none());
    }

    #[test]
    fn ecd_typical_values() {
        let ecd = compute_ecd(12.0, 200.0, 10_000.0).unwrap();
        // ECD should be slightly above mud weight.
        assert!(ecd > 12.0 && ecd < 13.0, "ECD = {ecd}");
    }

    #[test]
    fn flow_balance_positive_means_kick() {
        let fb = compute_flow_balance(410.0, 400.0);
        assert!(fb > 0.0); // Potential kick.
    }

    #[test]
    fn flow_balance_negative_means_loss() {
        let fb = compute_flow_balance(390.0, 400.0);
        assert!(fb < 0.0); // Potential loss.
    }
}

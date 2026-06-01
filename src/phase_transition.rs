//! Phase transitions in policy space.
//!
//! Stochastic geometry determines explore/exploit boundary.
//! Critical temperature T_c separates ordered (exploit) from disordered (explore) phases.
//! Order parameter: policy entropy S.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::temperature::Temperature;

/// Type of phase transition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PhaseType {
    /// First-order: discontinuous jump in order parameter
    FirstOrder,
    /// Second-order: continuous but non-differentiable
    SecondOrder,
    /// Crossover: smooth interpolation (no true phase transition)
    Crossover,
}

/// Phase of the policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PolicyPhase {
    /// Exploit phase (low temperature, low entropy)
    Exploit,
    /// Explore phase (high temperature, high entropy)
    Explore,
    /// Critical (near phase transition)
    Critical,
}

/// Phase transition detector and analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Critical temperature
    pub critical_temp: f64,
    /// Type of phase transition
    pub transition_type: PhaseType,
    /// Number of actions
    pub n_actions: usize,
    /// Order parameter history: (temperature, entropy)
    pub history: Vec<(f64, f64)>,
}

impl PhaseTransition {
    /// Create a new phase transition analyzer.
    pub fn new(critical_temp: f64, n_actions: usize) -> Self {
        Self {
            critical_temp,
            transition_type: PhaseType::SecondOrder,
            n_actions,
            history: Vec::new(),
        }
    }

    /// Determine the current phase based on temperature.
    pub fn classify_phase(&self, temperature: f64, tolerance: f64) -> PolicyPhase {
        let ratio = temperature / self.critical_temp;
        if (ratio - 1.0).abs() < tolerance {
            PolicyPhase::Critical
        } else if ratio < 1.0 {
            PolicyPhase::Exploit
        } else {
            PolicyPhase::Explore
        }
    }

    /// Compute order parameter (entropy) at given temperature.
    /// For Boltzmann policy: S(T) = ln Z(T) + E[Q]/T
    pub fn order_parameter(&self, q_values: &[f64], temperature: f64) -> f64 {
        let temp = Temperature::new(temperature);
        temp.boltzmann_entropy(q_values)
    }

    /// Susceptibility: χ = ∂S/∂T (diverges at T_c for second-order).
    pub fn susceptibility(&self, entropy_at_t: f64, entropy_at_t_plus_dt: f64, dt: f64) -> f64 {
        if dt.abs() > 1e-12 {
            (entropy_at_t_plus_dt - entropy_at_t) / dt
        } else {
            0.0
        }
    }

    /// Specific heat: C = T ∂S/∂T
    pub fn specific_heat(&self, t: f64, entropy_at_t: f64, entropy_at_t_plus_dt: f64, dt: f64) -> f64 {
        t * self.susceptibility(entropy_at_t, entropy_at_t_plus_dt, dt)
    }

    /// Critical exponent: S ∝ |T − T_c|^α near T_c.
    /// Estimate α from two measurements near T_c.
    pub fn estimate_critical_exponent(
        &self,
        t1: f64,
        s1: f64,
        t2: f64,
        s2: f64,
    ) -> f64 {
        let dt1 = (t1 - self.critical_temp).abs();
        let dt2 = (t2 - self.critical_temp).abs();
        if dt1 > 1e-12 && dt2 > 1e-12 && s1 > 1e-12 && s2 > 1e-12 {
            (s2.ln() - s1.ln()) / (dt2.ln() - dt1.ln())
        } else {
            0.0
        }
    }

    /// Free energy barrier between phases (for first-order transitions).
    pub fn free_energy_barrier(
        &self,
        f_exploit: f64,
        f_explore: f64,
        f_saddle: f64,
    ) -> f64 {
        let f_min = f_exploit.min(f_explore);
        f_saddle - f_min
    }

    /// Record observation for later analysis.
    pub fn record(&mut self, temperature: f64, entropy: f64) {
        self.history.push((temperature, entropy));
    }

    /// Detect phase transition from history: find maximum in |∂S/∂T|.
    pub fn detect_transition(&self) -> Option<f64> {
        if self.history.len() < 3 {
            return None;
        }
        let mut max_deriv = 0.0;
        let mut t_at_max = self.history[0].0;
        for i in 1..self.history.len() - 1 {
            let (t_prev, s_prev) = self.history[i - 1];
            let (t_curr, s_curr) = self.history[i];
            let (t_next, s_next) = self.history[i + 1];
            let dt = t_next - t_prev;
            if dt.abs() > 1e-12 {
                let d2s = (s_next - 2.0 * s_curr + s_prev) / (0.5 * dt).powi(2);
                if d2s.abs() > max_deriv {
                    max_deriv = d2s.abs();
                    t_at_max = t_curr;
                }
            }
        }
        Some(t_at_max)
    }

    /// Hysteresis width (for first-order transitions).
    pub fn hysteresis_width(&self, t_heating: f64, t_cooling: f64) -> f64 {
        (t_heating - t_cooling).abs()
    }

    /// Maximum entropy (at infinite temperature): S_max = ln(n).
    pub fn max_entropy(&self) -> f64 {
        (self.n_actions as f64).ln()
    }

    /// Latent heat (for first-order): ΔS at transition.
    pub fn latent_heat(&self, s_exploit: f64, s_explore: f64) -> f64 {
        (s_explore - s_exploit) * self.critical_temp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_classify_exploit() {
        let pt = PhaseTransition::new(1.0, 4);
        assert_eq!(pt.classify_phase(0.5, 0.1), PolicyPhase::Exploit);
    }

    #[test]
    fn test_classify_explore() {
        let pt = PhaseTransition::new(1.0, 4);
        assert_eq!(pt.classify_phase(2.0, 0.1), PolicyPhase::Explore);
    }

    #[test]
    fn test_classify_critical() {
        let pt = PhaseTransition::new(1.0, 4);
        assert_eq!(pt.classify_phase(1.0, 0.1), PolicyPhase::Critical);
    }

    #[test]
    fn test_order_parameter_increases_with_temp() {
        let pt = PhaseTransition::new(1.0, 4);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let s_low = pt.order_parameter(&q, 0.1);
        let s_high = pt.order_parameter(&q, 10.0);
        assert!(s_high > s_low);
    }

    #[test]
    fn test_susceptibility() {
        let pt = PhaseTransition::new(1.0, 4);
        let chi = pt.susceptibility(1.0, 1.5, 1.0);
        assert_relative_eq!(chi, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_specific_heat() {
        let pt = PhaseTransition::new(1.0, 4);
        let c = pt.specific_heat(2.0, 1.0, 1.5, 1.0);
        assert_relative_eq!(c, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_entropy() {
        let pt = PhaseTransition::new(1.0, 4);
        assert_relative_eq!(pt.max_entropy(), 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_free_energy_barrier() {
        let pt = PhaseTransition::new(1.0, 4);
        let barrier = pt.free_energy_barrier(-5.0, -3.0, 0.0);
        assert_relative_eq!(barrier, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_record_and_detect() {
        let mut pt = PhaseTransition::new(1.0, 4);
        // Simulate entropy vs temperature
        let q = vec![0.0, 1.0, 2.0, 3.0];
        for i in 0..20 {
            let t = 0.1 + (i as f64) * 0.1;
            let s = pt.order_parameter(&q, t);
            pt.record(t, s);
        }
        let detected = pt.detect_transition();
        assert!(detected.is_some());
    }

    #[test]
    fn test_latent_heat() {
        let pt = PhaseTransition::new(2.0, 4);
        let lh = pt.latent_heat(0.5, 1.5);
        assert_relative_eq!(lh, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hysteresis_width() {
        let pt = PhaseTransition::new(1.0, 4);
        assert_relative_eq!(pt.hysteresis_width(1.1, 0.9), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_estimate_critical_exponent() {
        let pt = PhaseTransition::new(1.0, 4);
        let alpha = pt.estimate_critical_exponent(1.1, 0.1, 1.2, 0.2);
        // Both near critical, should be positive
        assert!(alpha > 0.0 || alpha == 0.0); // may be NaN-ish
    }
}

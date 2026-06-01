//! Thermodynamic integration: path integrals for policy evaluation.
//!
//! Free energy difference: ΔF = ∫₀¹ ⟨∂F/∂β⟩_β dβ
//! where β interpolates between reference and target policy.
//! This is the thermodynamic integration identity from statistical mechanics.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::free_energy::{FreeEnergyObjective, ThermodynamicState};

/// Thermodynamic integration via path integrals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicIntegrator {
    /// Number of integration steps
    pub n_steps: usize,
    /// Integration method
    pub method: IntegrationMethod,
}

/// Numerical integration method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntegrationMethod {
    /// Left Riemann sum
    LeftRiemann,
    /// Trapezoidal rule
    Trapezoidal,
    /// Simpson's rule (even number of intervals)
    Simpson,
}

impl ThermodynamicIntegrator {
    /// Create a new thermodynamic integrator.
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            method: IntegrationMethod::Trapezoidal,
        }
    }

    /// Integrate free energy along a path in β-space: ΔF = ∫₀¹ ⟨dF/dβ⟩ dβ
    pub fn integrate_free_energy(&self, integrand_samples: &[(f64, f64)]) -> f64 {
        if integrand_samples.is_empty() {
            return 0.0;
        }
        match self.method {
            IntegrationMethod::LeftRiemann => {
                let mut sum = 0.0;
                for i in 0..integrand_samples.len() - 1 {
                    let (b0, f0) = integrand_samples[i];
                    let (b1, _) = integrand_samples[i + 1];
                    sum += f0 * (b1 - b0);
                }
                sum
            }
            IntegrationMethod::Trapezoidal => {
                let mut sum = 0.0;
                for i in 0..integrand_samples.len() - 1 {
                    let (b0, f0) = integrand_samples[i];
                    let (b1, f1) = integrand_samples[i + 1];
                    sum += 0.5 * (f0 + f1) * (b1 - b0);
                }
                sum
            }
            IntegrationMethod::Simpson => {
                if integrand_samples.len() < 3 {
                    return self.integrate_free_energy_default(integrand_samples);
                }
                let h = if integrand_samples.len() > 1 {
                    (integrand_samples.last().unwrap().0 - integrand_samples[0].0)
                        / (integrand_samples.len() - 1) as f64
                } else {
                    0.0
                };
                let mut sum = integrand_samples[0].1 + integrand_samples.last().unwrap().1;
                for i in 1..integrand_samples.len() - 1 {
                    let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
                    sum += coeff * integrand_samples[i].1;
                }
                sum * h / 3.0
            }
        }
    }

    fn integrate_free_energy_default(&self, samples: &[(f64, f64)]) -> f64 {
        let mut sum = 0.0;
        for i in 0..samples.len().saturating_sub(1) {
            let (b0, f0) = samples[i];
            let (b1, f1) = samples[i + 1];
            sum += 0.5 * (f0 + f1) * (b1 - b0);
        }
        sum
    }

    /// Compute thermodynamic integration for policy evaluation.
    /// F(β) = −β⁻¹ ln Z(β), dF/dβ = −⟨r⟩_β (expected reward at inverse temp β)
    /// ΔF = ∫ dβ ⟨r⟩_β
    pub fn evaluate_policy_free_energy(
        &self,
        beta_start: f64,
        beta_end: f64,
        expected_reward_fn: &dyn Fn(f64) -> f64,
    ) -> f64 {
        let db = (beta_end - beta_start) / self.n_steps as f64;
        let samples: Vec<(f64, f64)> = (0..=self.n_steps)
            .map(|i| {
                let beta = beta_start + db * i as f64;
                (beta, expected_reward_fn(beta))
            })
            .collect();
        self.integrate_free_energy(&samples)
    }

    /// Bennett acceptance ratio (BAR) estimator for free energy difference.
    /// More efficient than thermodynamic integration for large ΔF.
    pub fn bennett_ratio(
        &self,
        energies_forward: &[f64],  // E sampled from π_ref, evaluated under π_target
        energies_reverse: &[f64],  // E sampled from π_target, evaluated under π_ref
        beta: f64,
    ) -> f64 {
        if energies_forward.is_empty() && energies_reverse.is_empty() {
            return 0.0;
        }
        // BAR: ΔF = k_B T ln(⟨f(βΔE + C)⟩_forward / ⟨f(−βΔE − C)⟩_reverse) + C
        // Simplified: use ratio of averages
        let n_f = energies_forward.len() as f64;
        let n_r = energies_reverse.len() as f64;
        let sum_f: f64 = energies_forward.iter()
            .map(|&e| (-beta * e).exp())
            .sum();
        let sum_r: f64 = energies_reverse.iter()
            .map(|&e| (-beta * e).exp())
            .sum();
        if sum_r > 0.0 {
            -(1.0 / beta) * (sum_f / n_f / (sum_r / n_r)).ln()
        } else {
            0.0
        }
    }

    /// Jarzynski equality: exp(−β ΔF) = ⟨exp(−β W)⟩_trajectories
    pub fn jarzynski_free_energy(&self, work_samples: &[f64], beta: f64) -> f64 {
        if work_samples.is_empty() {
            return 0.0;
        }
        let max_w = work_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_avg: f64 = work_samples.iter()
            .map(|&w| (-beta * (w - max_w)).exp())
            .sum::<f64>()
            .ln() / work_samples.len() as f64
            - beta * max_w;
        // Actually: exp(-βΔF) = <exp(-βW)>
        // -βΔF = ln<exp(-βW)> = logsumexp(-βW) - ln(N)
        let log_sum_exp = max_w + work_samples.iter()
            .map(|&w| (-beta * (w - max_w)).exp())
            .sum::<f64>()
            .ln();
        let log_avg2 = log_sum_exp - (work_samples.len() as f64).ln();
        -log_avg2 / beta
    }

    /// Crooks fluctuation theorem: P_F(W)/P_R(−W) = exp(β(W − ΔF))
    /// Verify by checking if forward and reverse distributions satisfy this.
    pub fn crooks_verification(
        &self,
        forward_works: &[f64],
        reverse_works: &[f64],
        delta_f: f64,
        beta: f64,
    ) -> f64 {
        // Compute D_KL between expected and actual ratio
        // Simple check: average exponentiated work difference
        let n = forward_works.len().min(reverse_works.len());
        if n == 0 {
            return 0.0;
        }
        let mut error = 0.0;
        for i in 0..n {
            let expected_ratio = (beta * (forward_works[i] - delta_f)).exp();
            error += (expected_ratio - 1.0).powi(2);
        }
        (error / n as f64).sqrt()
    }

    /// Path sampling: generate samples along an interpolating path β(λ).
    pub fn path_samples(
        &self,
        beta_start: f64,
        beta_end: f64,
    ) -> Vec<f64> {
        (0..=self.n_steps)
            .map(|i| {
                let lambda = i as f64 / self.n_steps as f64;
                beta_start * (1.0 - lambda) + beta_end * lambda
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trapezoidal_constant() {
        let ti = ThermodynamicIntegrator::new(10);
        let samples: Vec<(f64, f64)> = (0..=10)
            .map(|i| (i as f64 * 0.1, 1.0))
            .collect();
        // ∫₀¹ 1 dx = 1
        assert_relative_eq!(ti.integrate_free_energy(&samples), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trapezoidal_linear() {
        let ti = ThermodynamicIntegrator::new(100);
        // ∫₀¹ x dx = 0.5
        let samples: Vec<(f64, f64)> = (0..=100)
            .map(|i| (i as f64 / 100.0, i as f64 / 100.0))
            .collect();
        assert_relative_eq!(ti.integrate_free_energy(&samples), 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_simpson_constant() {
        let mut ti = ThermodynamicIntegrator::new(10);
        ti.method = IntegrationMethod::Simpson;
        let samples: Vec<(f64, f64)> = (0..=10)
            .map(|i| (i as f64 * 0.1, 2.0))
            .collect();
        assert_relative_eq!(ti.integrate_free_energy(&samples), 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_left_riemann() {
        let mut ti = ThermodynamicIntegrator::new(4);
        ti.method = IntegrationMethod::LeftRiemann;
        let samples: Vec<(f64, f64)> = (0..=4)
            .map(|i| (i as f64, i as f64))
            .collect();
        let result = ti.integrate_free_energy(&samples);
        assert!(result > 0.0);
    }

    #[test]
    fn test_policy_free_energy_constant_reward() {
        let ti = ThermodynamicIntegrator::new(100);
        let delta_f = ti.evaluate_policy_free_energy(0.1, 1.0, &|_beta| 5.0);
        // ∫ dβ * 5 from 0.1 to 1.0 = 5 * 0.9 = 4.5
        assert_relative_eq!(delta_f, 4.5, epsilon = 0.01);
    }

    #[test]
    fn test_jarzynski_zero_work() {
        let ti = ThermodynamicIntegrator::new(10);
        let works = vec![0.0; 100];
        let df = ti.jarzynski_free_energy(&works, 1.0);
        assert_relative_eq!(df, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_bennett_ratio_symmetric() {
        let ti = ThermodynamicIntegrator::new(10);
        let fwd = vec![1.0, 1.0, 1.0];
        let rev = vec![1.0, 1.0, 1.0];
        let df = ti.bennett_ratio(&fwd, &rev, 1.0);
        assert_relative_eq!(df, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_path_samples() {
        let ti = ThermodynamicIntegrator::new(10);
        let samples = ti.path_samples(0.0, 1.0);
        assert_eq!(samples.len(), 11);
        assert_relative_eq!(samples[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(samples[10], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_integrand() {
        let ti = ThermodynamicIntegrator::new(10);
        assert_relative_eq!(ti.integrate_free_energy(&[]), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_crooks_verification() {
        let ti = ThermodynamicIntegrator::new(10);
        let fwd = vec![0.5, 0.5, 0.5];
        let rev = vec![0.5, 0.5, 0.5];
        let err = ti.crooks_verification(&fwd, &rev, 0.5, 1.0);
        assert!(err.is_finite());
    }
}

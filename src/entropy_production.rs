//! Entropy production tracking: second law of thermodynamics in policy space.
//!
//! ΔS_total = ΔS_policy + ΔS_environment ≥ 0
//! Entropy production rate: σ = dS/dt ≥ 0

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Entropy production tracker for policy optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyProduction {
    /// Policy entropy history
    pub entropy_history: Vec<f64>,
    /// Time steps
    pub time_steps: Vec<usize>,
    /// Cumulative entropy production
    pub cumulative_production: f64,
}

impl EntropyProduction {
    /// Create a new entropy production tracker.
    pub fn new() -> Self {
        Self {
            entropy_history: Vec::new(),
            time_steps: Vec::new(),
            cumulative_production: 0.0,
        }
    }

    /// Shannon entropy of a discrete distribution.
    pub fn shannon_entropy(probs: &[f64]) -> f64 {
        -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Differential entropy of a Gaussian: H = d/2 (1 + ln(2πσ²))
    pub fn gaussian_entropy(dim: usize, sigma: f64) -> f64 {
        (dim as f64) / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma * sigma).ln())
    }

    /// Entropy production rate: σ = dS/dt ≈ (S_{t+1} − S_t) / Δt
    pub fn production_rate(&self, dt: usize) -> f64 {
        if self.entropy_history.len() < 2 {
            return 0.0;
        }
        let n = self.entropy_history.len();
        let ds = self.entropy_history[n - 1] - self.entropy_history[n - 2];
        ds / dt.max(1) as f64
    }

    /// Record entropy observation.
    pub fn record(&mut self, entropy: f64, time_step: usize) {
        if !self.entropy_history.is_empty() {
            let ds = entropy - self.entropy_history.last().unwrap();
            self.cumulative_production += ds.max(0.0); // production is non-negative
        }
        self.entropy_history.push(entropy);
        self.time_steps.push(time_step);
    }

    /// Second law check: total entropy change should be non-negative.
    pub fn second_law_satisfied(&self) -> bool {
        if self.entropy_history.len() < 2 {
            return true;
        }
        // In a closed system: ΔS ≥ 0
        // In RL: policy entropy can decrease (exploitation), but
        // total entropy (policy + environment) must increase
        self.cumulative_production >= -1e-10
    }

    /// Kullback-Leibler divergence as entropy production.
    /// D_KL(π_new || π_old) ≥ 0 is a form of entropy production.
    pub fn kl_entropy_production(p_new: &[f64], p_old: &[f64]) -> f64 {
        p_new.iter().zip(p_old.iter())
            .filter(|(_, q)| **q > 0.0)
            .map(|(p, q)| {
                if *p > 0.0 {
                    p * (p / q).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Relative entropy production (irreversibility measure).
    pub fn relative_entropy_production(
        forward_transition: &[(usize, usize, f64)], // (from, to, prob)
        reverse_transition: &[(usize, usize, f64)],
    ) -> f64 {
        let mut sigma = 0.0;
        for (i, t_fwd) in forward_transition.iter().enumerate() {
            if let Some(t_rev) = reverse_transition.get(i) {
                if t_fwd.2 > 0.0 && t_rev.2 > 0.0 {
                    sigma += t_fwd.2 * (t_fwd.2 / t_rev.2).ln();
                }
            }
        }
        sigma
    }

    /// Entropy flow to environment: Φ = dS_env/dt
    pub fn entropy_flow(&self, reward_rate: f64, temperature: f64) -> f64 {
        if temperature > 0.0 {
            reward_rate / temperature
        } else {
            0.0
        }
    }

    /// Total entropy balance: dS/dt = σ + Φ (production + flow)
    pub fn entropy_balance(
        &self,
        reward_rate: f64,
        temperature: f64,
        dt: usize,
    ) -> (f64, f64, f64) {
        let sigma = self.production_rate(dt);
        let phi = self.entropy_flow(reward_rate, temperature);
        let total = sigma + phi;
        (total, sigma, phi)
    }

    /// Efficiency: η = σ_useful / σ_total (thermodynamic efficiency of learning)
    pub fn thermodynamic_efficiency(
        useful_production: f64,
        total_production: f64,
    ) -> f64 {
        if total_production > 0.0 {
            (useful_production / total_production).min(1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shannon_entropy_uniform() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = EntropyProduction::shannon_entropy(&p);
        assert_relative_eq!(h, 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let p = vec![1.0, 0.0];
        assert_relative_eq!(EntropyProduction::shannon_entropy(&p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_entropy() {
        let h = EntropyProduction::gaussian_entropy(1, 1.0);
        // H = 0.5 * (1 + ln(2π))
        assert_relative_eq!(h, 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln()), epsilon = 1e-10);
    }

    #[test]
    fn test_production_rate_empty() {
        let ep = EntropyProduction::new();
        assert_relative_eq!(ep.production_rate(1), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_record_and_production() {
        let mut ep = EntropyProduction::new();
        ep.record(1.0, 0);
        ep.record(1.5, 1);
        assert_relative_eq!(ep.production_rate(1), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_second_law_satisfied() {
        let mut ep = EntropyProduction::new();
        ep.record(1.0, 0);
        ep.record(1.5, 1);
        ep.record(2.0, 2);
        assert!(ep.second_law_satisfied());
    }

    #[test]
    fn test_kl_entropy_production() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert_relative_eq!(EntropyProduction::kl_entropy_production(&p, &q), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_entropy_production_positive() {
        let p = vec![1.0, 0.0];
        let q = vec![0.5, 0.5];
        let kl = EntropyProduction::kl_entropy_production(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_entropy_flow() {
        let ep = EntropyProduction::new();
        let phi = ep.entropy_flow(10.0, 2.0);
        assert_relative_eq!(phi, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_balance() {
        let mut ep = EntropyProduction::new();
        ep.record(1.0, 0);
        ep.record(1.5, 1);
        let (total, sigma, phi) = ep.entropy_balance(10.0, 2.0, 1);
        assert!(total.is_finite());
    }

    #[test]
    fn test_thermodynamic_efficiency() {
        assert_relative_eq!(EntropyProduction::thermodynamic_efficiency(0.5, 1.0), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_thermodynamic_efficiency_perfect() {
        assert_relative_eq!(EntropyProduction::thermodynamic_efficiency(1.0, 1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cumulative_production() {
        let mut ep = EntropyProduction::new();
        ep.record(1.0, 0);
        ep.record(2.0, 1); // +1.0
        ep.record(1.5, 2); // -0.5, but production only counts positive
        assert_relative_eq!(ep.cumulative_production, 1.0, epsilon = 1e-10);
    }
}

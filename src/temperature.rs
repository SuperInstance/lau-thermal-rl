//! Temperature as exploration parameter.
//!
//! High T → explore (entropy dominates)
//! Low T → exploit (reward dominates)
//! T → 0: greedy policy (zero-temperature limit)
//! T → ∞: uniform random (maximum entropy)

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

/// Temperature controller for thermal RL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Temperature {
    /// Current temperature
    pub value: f64,
    /// Minimum temperature (prevents division by zero)
    pub min_temp: f64,
    /// Maximum temperature
    pub max_temp: f64,
    /// Decay schedule: T_{t+1} = T_t * decay_rate
    pub decay_rate: f64,
    /// Annealing schedule type
    pub schedule: AnnealingSchedule,
}

/// Temperature annealing schedule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AnnealingSchedule {
    /// Constant temperature
    Constant,
    /// Linear decay: T_t = T_0 (1 − t/T_max)
    Linear { t_max: f64 },
    /// Exponential decay: T_t = T_0 · exp(−λt)
    Exponential { lambda: f64 },
    /// Cosine annealing: T_t = T_min + ½(T_max − T_min)(1 + cos(πt/T_max))
    Cosine { t_max: f64 },
    /// Inverse sqrt: T_t = T_0 / √(1 + t)
    InverseSqrt,
}

impl Temperature {
    /// Create a new temperature controller.
    pub fn new(initial: f64) -> Self {
        Self {
            value: initial,
            min_temp: 1e-6,
            max_temp: 1e6,
            decay_rate: 0.999,
            schedule: AnnealingSchedule::Constant,
        }
    }

    /// Get inverse temperature β = 1/T.
    pub fn beta(&self) -> f64 {
        1.0 / self.value.max(self.min_temp)
    }

    /// Boltzmann weights: π(a) ∝ exp(Q(a)/T)
    pub fn boltzmann_weights(&self, q_values: &[f64]) -> Vec<f64> {
        let max_q = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t = self.value.max(self.min_temp);
        let weights: Vec<f64> = q_values.iter()
            .map(|q| ((q - max_q) / t).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / q_values.len() as f64; q_values.len()]
        }
    }

    /// Gumbel-softmax sample (differentiable approximation).
    pub fn gumbel_sample(&self, logits: &[f64], rng_samples: &[f64]) -> Vec<f64> {
        let t = self.value.max(self.min_temp);
        let gumbels: Vec<f64> = rng_samples.iter()
            .map(|&u| {
                let u_clamped = u.clamp(1e-10, 1.0 - 1e-10);
                -(-u_clamped.ln()).ln()
            })
            .collect();
        let perturbed: Vec<f64> = logits.iter().zip(gumbels.iter())
            .map(|(&l, &g)| (l + g) / t)
            .collect();
        let max_val = perturbed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = perturbed.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Exploration probability: P(explore) ≈ 1 − exp(−T/T_ref)
    pub fn exploration_probability(&self, t_ref: f64) -> f64 {
        1.0 - (-self.value / t_ref).exp()
    }

    /// Entropy at current temperature for Boltzmann over n actions.
    /// S = ln Z + E[Q]/T where Z = Σ exp(Q/T)
    pub fn boltzmann_entropy(&self, q_values: &[f64]) -> f64 {
        let probs = self.boltzmann_weights(q_values);
        -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Step the temperature according to the schedule.
    pub fn step(&mut self, t: usize) {
        self.value = match self.schedule {
            AnnealingSchedule::Constant => self.value,
            AnnealingSchedule::Linear { t_max } => {
                self.value * (1.0 - t as f64 / t_max).max(self.min_temp)
            }
            AnnealingSchedule::Exponential { lambda } => {
                self.value * (-lambda * t as f64).exp()
            }
            AnnealingSchedule::Cosine { t_max } => {
                self.min_temp + 0.5 * (self.max_temp - self.min_temp) * (1.0 + (std::f64::consts::PI * t as f64 / t_max).cos())
            }
            AnnealingSchedule::InverseSqrt => {
                self.value / (1.0 + t as f64).sqrt()
            }
        };
        self.value = self.value.clamp(self.min_temp, self.max_temp);
    }

    /// Thermodynamic beta schedule: β_t = β_0 + (β_max − β_0) · t/T_max
    pub fn beta_schedule(&self, step: usize, total_steps: usize) -> f64 {
        let beta_0 = 1.0 / self.max_temp;
        let beta_max = 1.0 / self.min_temp;
        let frac = step as f64 / total_steps as f64;
        beta_0 + (beta_max - beta_0) * frac
    }

    /// Check if temperature is in the "critical" regime near phase transition.
    pub fn is_critical(&self, critical_temp: f64, tolerance: f64) -> bool {
        (self.value - critical_temp).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beta_inverse() {
        let t = Temperature::new(2.0);
        assert_relative_eq!(t.beta(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_clamped() {
        let mut t = Temperature::new(0.0);
        t.min_temp = 1e-6;
        assert!(t.beta().is_finite());
    }

    #[test]
    fn test_boltzmann_uniform_at_high_temp() {
        let mut t = Temperature::new(1000.0);
        t.min_temp = 1e-10;
        let q = vec![1.0, 2.0, 3.0];
        let probs = t.boltzmann_weights(&q);
        // At high T, should be approximately uniform
        for p in &probs {
            assert_relative_eq!(*p, 1.0 / 3.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_boltzmann_greedy_at_low_temp() {
        let mut t = Temperature::new(0.001);
        t.min_temp = 1e-10;
        let q = vec![1.0, 2.0, 3.0];
        let probs = t.boltzmann_weights(&q);
        // At low T, should be nearly one-hot on argmax
        assert!(probs[2] > 0.99);
    }

    #[test]
    fn test_boltzmann_sums_to_one() {
        let t = Temperature::new(1.0);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let probs = t.boltzmann_weights(&q);
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exploration_probability() {
        let t = Temperature::new(1.0);
        let p = t.exploration_probability(1.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_exploration_probability_high_temp() {
        let t = Temperature::new(100.0);
        let p = t.exploration_probability(1.0);
        assert!(p > 0.9);
    }

    #[test]
    fn test_boltzmann_entropy_uniform() {
        let mut t = Temperature::new(1000.0);
        t.min_temp = 1e-10;
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let h = t.boltzmann_entropy(&q);
        assert_relative_eq!(h, 4.0_f64.ln(), epsilon = 0.01);
    }

    #[test]
    fn test_constant_schedule() {
        let mut t = Temperature::new(1.0);
        t.schedule = AnnealingSchedule::Constant;
        for i in 0..10 {
            t.step(i);
        }
        assert_relative_eq!(t.value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_schedule() {
        let mut t = Temperature::new(1.0);
        t.schedule = AnnealingSchedule::Exponential { lambda: 0.1 };
        t.step(1);
        assert!(t.value < 1.0);
    }

    #[test]
    fn test_inverse_sqrt_schedule() {
        let mut t = Temperature::new(1.0);
        t.schedule = AnnealingSchedule::InverseSqrt;
        let initial = t.value;
        t.step(3);
        assert!(t.value < initial);
    }

    #[test]
    fn test_is_critical() {
        let t = Temperature::new(1.0);
        assert!(t.is_critical(1.0, 0.1));
        assert!(!t.is_critical(2.0, 0.1));
    }

    #[test]
    fn test_beta_schedule() {
        let t = Temperature::new(1.0);
        let b0 = t.beta_schedule(0, 100);
        let b100 = t.beta_schedule(100, 100);
        assert!(b100 > b0);
    }

    #[test]
    fn test_gumbel_sample_sums_to_one() {
        let t = Temperature::new(1.0);
        let logits = vec![1.0, 2.0, 3.0];
        let rng = vec![0.5, 0.3, 0.8];
        let sample = t.gumbel_sample(&logits, &rng);
        let sum: f64 = sample.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

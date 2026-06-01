//! Thermal policy: policy parameterized by temperature.
//!
//! Combines reward maximization with entropy regularization.
//! At each temperature, the optimal policy minimizes free energy.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::temperature::Temperature;
use crate::free_energy::FreeEnergyObjective;

/// A policy parameterized by temperature (thermal/Gibbs policy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPolicy {
    /// Temperature controller
    pub temperature: Temperature,
    /// Number of actions
    pub n_actions: usize,
    /// Mean parameters (for Gaussian policy)
    pub mu: DVector<f64>,
    /// Standard deviation (for Gaussian policy)
    pub sigma: f64,
}

impl ThermalPolicy {
    /// Create a new thermal policy.
    pub fn new(n_actions: usize, dim: usize, initial_temp: f64) -> Self {
        Self {
            temperature: Temperature::new(initial_temp),
            n_actions,
            mu: DVector::zeros(dim),
            sigma: 1.0,
        }
    }

    /// Discrete action probabilities: π(a) ∝ exp(Q(a)/T)
    pub fn action_probabilities(&self, q_values: &[f64]) -> Vec<f64> {
        self.temperature.boltzmann_weights(q_values)
    }

    /// Sample action (deterministic for testing: argmax).
    pub fn greedy_action(&self, q_values: &[f64]) -> usize {
        q_values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Stochastic action (Boltzmann).
    pub fn boltzmann_action(&self, q_values: &[f64], random_value: f64) -> usize {
        let probs = self.action_probabilities(q_values);
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if random_value <= cumsum {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Policy entropy at current temperature.
    pub fn entropy(&self, q_values: &[f64]) -> f64 {
        self.temperature.boltzmann_entropy(q_values)
    }

    /// Update mean parameters.
    pub fn update_mu(&mut self, delta: &DVector<f64>) {
        self.mu += delta;
    }

    /// Set temperature.
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature.value = temp;
    }

    /// Free energy of the policy.
    pub fn free_energy(&self, expected_reward: f64, kl_divergence: f64) -> f64 {
        expected_reward - self.temperature.value * kl_divergence
    }

    /// KL divergence from this policy to another Gaussian.
    pub fn kl_to(&self, other: &ThermalPolicy) -> f64 {
        let d = self.mu.nrows() as f64;
        let diff = &self.mu - &other.mu;
        d * (other.sigma / self.sigma).ln()
            + (&diff.transpose() * &diff)[(0, 0)] / (2.0 * other.sigma * other.sigma)
            + d * (self.sigma * self.sigma) / (2.0 * other.sigma * other.sigma)
            - d / 2.0
    }

    /// Gradient of log-policy w.r.t. mu (score function).
    pub fn score_function(&self, action_mu: &DVector<f64>) -> DVector<f64> {
        (action_mu - &self.mu) / (self.sigma * self.sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_action_probabilities_sum() {
        let policy = ThermalPolicy::new(4, 2, 1.0);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let probs = policy.action_probabilities(&q);
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_greedy_action() {
        let policy = ThermalPolicy::new(4, 2, 1.0);
        let q = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(policy.greedy_action(&q), 3);
    }

    #[test]
    fn test_boltzmann_action_low_temp() {
        let mut policy = ThermalPolicy::new(3, 2, 0.01);
        let q = vec![1.0, 2.0, 3.0];
        // At very low temp, should almost always pick argmax
        let action = policy.boltzmann_action(&q, 0.5);
        assert_eq!(action, 2);
    }

    #[test]
    fn test_entropy_high_temp() {
        let mut policy = ThermalPolicy::new(4, 2, 100.0);
        policy.temperature.min_temp = 1e-10;
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let h = policy.entropy(&q);
        assert_relative_eq!(h, 4.0_f64.ln(), epsilon = 0.01);
    }

    #[test]
    fn test_free_energy() {
        let policy = ThermalPolicy::new(4, 2, 1.0);
        let f = policy.free_energy(10.0, 2.0);
        assert_relative_eq!(f, 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_update_mu() {
        let mut policy = ThermalPolicy::new(2, 3, 1.0);
        let delta = DVector::from_vec(vec![1.0, 0.0, -1.0]);
        policy.update_mu(&delta);
        assert_relative_eq!(policy.mu[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(policy.mu[2], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_to_self() {
        let p1 = ThermalPolicy::new(2, 2, 1.0);
        let p2 = p1.clone();
        let kl = p1.kl_to(&p2);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_positive() {
        let p1 = ThermalPolicy::new(2, 2, 1.0);
        let mut p2 = ThermalPolicy::new(2, 2, 1.0);
        p2.mu = DVector::from_vec(vec![1.0, 0.0]);
        let kl = p1.kl_to(&p2);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_score_function() {
        let policy = ThermalPolicy::new(2, 2, 1.0);
        let action = DVector::from_vec(vec![1.0, 0.0]);
        let score = policy.score_function(&action);
        assert!(score.norm() > 0.0);
    }

    #[test]
    fn test_set_temperature() {
        let mut policy = ThermalPolicy::new(2, 2, 1.0);
        policy.set_temperature(5.0);
        assert_relative_eq!(policy.temperature.value, 5.0, epsilon = 1e-10);
    }
}

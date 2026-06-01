//! Free energy RL objective: KL-regularized reward as Helmholtz free energy.
//!
//! J(θ) = E[Σ γᵗ rₜ] − β D_KL(Π_θ ‖ Π_ref) = F = U − TS
//! where U = expected return, S = Shannon entropy, β = 1/(k_B T).

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Thermodynamic state of a policy at a given parameterization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Internal energy: expected discounted return U = E[Σ γᵗ rₜ]
    pub internal_energy: f64,
    /// Shannon entropy of the policy S = -Σ π log π
    pub entropy: f64,
    /// Inverse temperature β = 1/(k_B T)
    pub beta: f64,
    /// Temperature T (exploration parameter)
    pub temperature: f64,
    /// Boltzmann constant (set to 1 in natural units)
    pub k_b: f64,
}

impl ThermodynamicState {
    /// Create a new thermodynamic state.
    pub fn new(internal_energy: f64, entropy: f64, temperature: f64) -> Self {
        let k_b = 1.0;
        let beta = if temperature > 0.0 { 1.0 / (k_b * temperature) } else { f64::INFINITY };
        Self { internal_energy, entropy, beta, temperature, k_b }
    }

    /// Helmholtz free energy F = U − TS.
    pub fn helmholtz_free_energy(&self) -> f64 {
        self.internal_energy - self.temperature * self.entropy
    }

    /// KL-regularized objective J(θ) = U − β⁻¹ D_KL = F in the correspondence.
    /// With β = 1/T, this gives J = U − T · D_KL.
    pub fn kl_regularized_objective(&self, kl_divergence: f64) -> f64 {
        self.internal_energy - (1.0 / self.beta) * kl_divergence
    }

    /// Thermodynamic pressure P = −∂F/∂V (volume = parameter space volume).
    pub fn pressure(&self, volume: f64) -> f64 {
        if volume > 0.0 {
            -(self.helmholtz_free_energy() / volume)
        } else {
            0.0
        }
    }

    /// Heat capacity at constant volume: C_V = T (∂S/∂T)_V.
    pub fn heat_capacity(&self, entropy_at_higher_temp: f64, delta_t: f64) -> f64 {
        if delta_t > 0.0 {
            self.temperature * (entropy_at_higher_temp - self.entropy) / delta_t
        } else {
            0.0
        }
    }

    /// Gibbs free energy G = F + PV.
    pub fn gibbs_free_energy(&self, pressure: f64, volume: f64) -> f64 {
        self.helmholtz_free_energy() + pressure * volume
    }

    /// Enthalpy H = U + PV.
    pub fn enthalpy(&self, pressure: f64, volume: f64) -> f64 {
        self.internal_energy + pressure * volume
    }
}

/// Free energy RL objective linking KL-regularized optimization to thermodynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyObjective {
    /// Discount factor γ
    pub discount_factor: f64,
    /// Inverse temperature β
    pub beta: f64,
    /// Reference policy parameters (for KL computation)
    pub ref_params: DVector<f64>,
    /// Dimension of parameter space
    pub param_dim: usize,
}

impl FreeEnergyObjective {
    /// Create a new free energy objective.
    pub fn new(discount_factor: f64, beta: f64, param_dim: usize) -> Self {
        Self {
            discount_factor,
            beta,
            ref_params: DVector::zeros(param_dim),
            param_dim,
        }
    }

    /// Compute discounted return U = Σ γᵗ rₜ.
    pub fn discounted_return(&self, rewards: &[f64]) -> f64 {
        rewards.iter().enumerate()
            .map(|(t, r)| self.discount_factor.powi(t as i32) * r)
            .sum()
    }

    /// Compute KL divergence D_KL(Π_θ ‖ Π_ref) for Gaussian policies.
    /// For two Gaussians N(μ₁, σ₁²) and N(μ₂, σ₂²):
    /// D_KL = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − 1/2
    pub fn kl_divergence_gaussian(
        &self,
        mu_theta: &DVector<f64>,
        sigma_theta: f64,
        mu_ref: &DVector<f64>,
        sigma_ref: f64,
    ) -> f64 {
        let d = mu_theta.nrows() as f64;
        let diff = mu_theta - mu_ref;
        let kl = d * (sigma_ref / sigma_theta).ln()
            + (diff.transpose() * &diff)[(0, 0)] / (2.0 * sigma_ref * sigma_ref)
            + d * (sigma_theta * sigma_theta) / (2.0 * sigma_ref * sigma_ref)
            - d / 2.0;
        kl.max(0.0)
    }

    /// Compute KL divergence for discrete distributions.
    pub fn kl_divergence_discrete(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter())
            .filter(|(_, q_i)| **q_i > 0.0)
            .map(|(p_i, q_i)| {
                if *p_i > 0.0 {
                    p_i * (p_i / q_i).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Full free energy objective: F = U − β⁻¹ D_KL.
    pub fn evaluate(
        &self,
        rewards: &[f64],
        mu_theta: &DVector<f64>,
        sigma_theta: f64,
        mu_ref: &DVector<f64>,
        sigma_ref: f64,
    ) -> ThermodynamicState {
        let u = self.discounted_return(rewards);
        let kl = self.kl_divergence_gaussian(mu_theta, sigma_theta, mu_ref, sigma_ref);
        let temperature = 1.0 / self.beta;
        // Entropy of Gaussian: S = d/2 (1 + ln(2πσ²))
        let d = mu_theta.nrows() as f64;
        let entropy = d / 2.0 * (1.0 + (2.0 * std::f64::consts::PI * sigma_theta * sigma_theta).ln());
        ThermodynamicState::new(u, entropy, temperature)
    }

    /// Gradient of the free energy w.r.t. parameters θ.
    /// ∇F = ∇U − β⁻¹ ∇D_KL
    pub fn gradient(
        &self,
        reward_grad: &DVector<f64>,
        mu_theta: &DVector<f64>,
        mu_ref: &DVector<f64>,
        sigma_ref: f64,
    ) -> DVector<f64> {
        let kl_grad = (mu_theta - mu_ref) / (sigma_ref * sigma_ref);
        reward_grad - (1.0 / self.beta) * &kl_grad
    }

    /// Shannon entropy of a discrete distribution.
    pub fn discrete_entropy(probs: &[f64]) -> f64 {
        -probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Cross entropy H(p, q) = −Σ p log q.
    pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
        -p.iter().zip(q.iter())
            .filter(|(_, q_i)| **q_i > 0.0)
            .map(|(p_i, q_i)| *p_i * q_i.ln())
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_helmholtz_free_energy_basic() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        // F = U - TS = 10 - 1*2 = 8
        assert_relative_eq!(state.helmholtz_free_energy(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_free_energy_zero_entropy() {
        let state = ThermodynamicState::new(5.0, 0.0, 2.0);
        assert_relative_eq!(state.helmholtz_free_energy(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_free_energy_zero_temperature() {
        let state = ThermodynamicState::new(10.0, 3.0, 0.0);
        // F = U - 0*S = U
        assert_relative_eq!(state.helmholtz_free_energy(), 10.0, epsilon = 1e-10);
        assert_eq!(state.beta, f64::INFINITY);
    }

    #[test]
    fn test_free_energy_high_temperature() {
        let state = ThermodynamicState::new(10.0, 3.0, 100.0);
        // F = 10 - 100*3 = -290
        assert_relative_eq!(state.helmholtz_free_energy(), -290.0, epsilon = 1e-10);
    }

    #[test]
    fn test_beta_inverse_temperature() {
        let state = ThermodynamicState::new(1.0, 1.0, 2.0);
        assert_relative_eq!(state.beta, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_discounted_return_no_discount() {
        let obj = FreeEnergyObjective::new(1.0, 1.0, 2);
        let rewards = vec![1.0, 1.0, 1.0];
        assert_relative_eq!(obj.discounted_return(&rewards), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_discounted_return_with_discount() {
        let obj = FreeEnergyObjective::new(0.9, 1.0, 2);
        let rewards = vec![1.0, 1.0, 1.0];
        let expected = 1.0 + 0.9 + 0.81;
        assert_relative_eq!(obj.discounted_return(&rewards), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        let obj = FreeEnergyObjective::new(0.99, 1.0, 2);
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let kl = obj.kl_divergence_gaussian(&mu, 1.0, &mu, 1.0);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_different_means() {
        let obj = FreeEnergyObjective::new(0.99, 1.0, 2);
        let mu1 = DVector::from_vec(vec![1.0, 0.0]);
        let mu2 = DVector::from_vec(vec![0.0, 0.0]);
        let kl = obj.kl_divergence_gaussian(&mu1, 1.0, &mu2, 1.0);
        // KL = 0 + 1/(2*1) + 0.5 - 0.5 = 0.5
        assert!(kl > 0.0);
    }

    #[test]
    fn test_kl_divergence_discrete() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert_relative_eq!(FreeEnergyObjective::kl_divergence_discrete(&p, &q), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_discrete_different() {
        let p = vec![1.0, 0.0];
        let q = vec![0.5, 0.5];
        let kl = FreeEnergyObjective::kl_divergence_discrete(&p, &q);
        // KL = 1*ln(1/0.5) = ln(2)
        assert_relative_eq!(kl, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_discrete_entropy_uniform() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = FreeEnergyObjective::discrete_entropy(&p);
        assert_relative_eq!(h, 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_discrete_entropy_deterministic() {
        let p = vec![1.0, 0.0, 0.0];
        assert_relative_eq!(FreeEnergyObjective::discrete_entropy(&p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_regularized_objective() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        // J = U - T * D_KL = 10 - 1*0.5 = 9.5
        let obj = state.kl_regularized_objective(0.5);
        assert_relative_eq!(obj, 9.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pressure() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        // F = 8, P = -F/V = -8/4 = -2
        let p = state.pressure(4.0);
        assert_relative_eq!(p, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gibbs_free_energy() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        // G = F + PV = 8 + 2*3 = 14
        let g = state.gibbs_free_energy(2.0, 3.0);
        assert_relative_eq!(g, 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_enthalpy() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        let h = state.enthalpy(1.5, 2.0);
        assert_relative_eq!(h, 13.0, epsilon = 1e-10);
    }

    #[test]
    fn test_heat_capacity() {
        let state = ThermodynamicState::new(10.0, 2.0, 1.0);
        let cv = state.heat_capacity(2.5, 1.0);
        // C_V = T * (S2 - S1) / dT = 1 * (2.5-2) / 1 = 0.5
        assert_relative_eq!(cv, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let ce = FreeEnergyObjective::cross_entropy(&p, &q);
        assert_relative_eq!(ce, std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_evaluate_full() {
        let obj = FreeEnergyObjective::new(0.99, 1.0, 2);
        let rewards = vec![1.0, 1.0, 1.0];
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let state = obj.evaluate(&rewards, &mu, 1.0, &mu, 1.0);
        assert_relative_eq!(state.internal_energy, obj.discounted_return(&rewards), epsilon = 1e-10);
        assert!(state.entropy > 0.0);
    }

    #[test]
    fn test_gradient_direction() {
        let obj = FreeEnergyObjective::new(0.99, 1.0, 2);
        let reward_grad = DVector::from_vec(vec![1.0, 0.0]);
        let mu_theta = DVector::from_vec(vec![1.0, 0.0]);
        let mu_ref = DVector::from_vec(vec![0.0, 0.0]);
        let grad = obj.gradient(&reward_grad, &mu_theta, &mu_ref, 1.0);
        // grad = [1,0] - 1*[1,0] = [0,0]
        assert_relative_eq!(grad[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 0.0, epsilon = 1e-10);
    }
}

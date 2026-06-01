//! Natural policy gradient: covariant derivative ∇̃J = G⁻¹∇J.
//!
//! The natural gradient corrects for the geometry of parameter space
//! by pre-multiplying the Euclidean gradient by the inverse Fisher metric.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::fisher_metric::FisherMetric;

/// Natural gradient descent optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalGradient {
    /// Fisher metric handler
    pub fisher: FisherMetric,
    /// Learning rate (step size in natural gradient space)
    pub learning_rate: f64,
    /// Damping factor for Fisher matrix inversion (regularization)
    pub damping: f64,
    /// Whether to use trust-region constraint (KL ≤ δ)
    pub trust_region: bool,
    /// Maximum KL divergence for trust region
    pub max_kl: f64,
}

impl NaturalGradient {
    /// Create a new natural gradient optimizer.
    pub fn new(dim: usize, learning_rate: f64) -> Self {
        Self {
            fisher: FisherMetric::new(dim),
            learning_rate,
            damping: 1e-8,
            trust_region: false,
            max_kl: 0.01,
        }
    }

    /// Compute the natural gradient: ∇̃J = G⁻¹∇J
    pub fn natural_gradient(
        &self,
        euclidean_grad: &DVector<f64>,
        fisher_matrix: &DMatrix<f64>,
    ) -> DVector<f64> {
        let damped_fisher = fisher_matrix + DMatrix::from_diagonal_element(
            fisher_matrix.nrows(), fisher_matrix.ncols(), self.damping,
        );
        match damped_fisher.clone().try_inverse() {
            Some(finv) => finv * euclidean_grad,
            None => euclidean_grad.clone(), // fallback to Euclidean
        }
    }

    /// Compute natural gradient update step: Δθ = −η G⁻¹∇J
    pub fn step(
        &self,
        params: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        fisher_matrix: &DMatrix<f64>,
    ) -> DVector<f64> {
        let nat_grad = self.natural_gradient(euclidean_grad, fisher_matrix);
        let delta = params - self.learning_rate * &nat_grad;
        delta
    }

    /// Trust-region natural gradient step.
    /// Scales step to satisfy KL(π_old || π_new) ≤ δ.
    pub fn trust_region_step(
        &self,
        params: &DVector<f64>,
        euclidean_grad: &DVector<f64>,
        fisher_matrix: &DMatrix<f64>,
        sigma: f64,
    ) -> DVector<f64> {
        let nat_grad = self.natural_gradient(euclidean_grad, fisher_matrix);
        let step = self.learning_rate * &nat_grad;
        // KL ≈ ½ Δθ^T G Δθ for small steps
        let kl_approx = 0.5 * (&step.transpose() * fisher_matrix * &step)[(0, 0)];
        let scale = if kl_approx > self.max_kl {
            (self.max_kl / kl_approx).sqrt()
        } else {
            1.0
        };
        params - scale * &step
    }

    /// Compute the effective step size in Fisher-Rao geometry.
    pub fn fisher_rao_step_size(
        &self,
        old_params: &DVector<f64>,
        new_params: &DVector<f64>,
        fisher_matrix: &DMatrix<f64>,
    ) -> f64 {
        let diff = new_params - old_params;
        (diff.transpose() * fisher_matrix * &diff)[(0, 0)].sqrt()
    }

    /// Covariant Hessian: H̃ = G⁻¹ H where H is the Euclidean Hessian.
    pub fn covariant_hessian(
        &self,
        hessian: &DMatrix<f64>,
        fisher_matrix: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let damped_fisher = fisher_matrix + DMatrix::from_diagonal_element(
            fisher_matrix.nrows(), fisher_matrix.ncols(), self.damping,
        );
        match damped_fisher.clone().try_inverse() {
            Some(finv) => finv * hessian,
            None => hessian.clone(),
        }
    }

    /// Convergence metric: norm of natural gradient in Fisher metric.
    pub fn convergence_criterion(
        &self,
        euclidean_grad: &DVector<f64>,
        fisher_matrix: &DMatrix<f64>,
    ) -> f64 {
        let nat = self.natural_gradient(euclidean_grad, fisher_matrix);
        (nat.transpose() * fisher_matrix * &nat)[(0, 0)].sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_natural_gradient_identity_fisher() {
        let ng = NaturalGradient::new(2, 0.01);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        let fisher = DMatrix::identity(2, 2);
        let nat = ng.natural_gradient(&grad, &fisher);
        assert_relative_eq!(nat[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(nat[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_natural_gradient_scaled() {
        let ng = NaturalGradient::new(2, 0.01);
        let grad = DVector::from_vec(vec![2.0, 0.0]);
        let fisher = DMatrix::from_diagonal_element(2, 2, 2.0);
        let nat = ng.natural_gradient(&grad, &fisher);
        assert_relative_eq!(nat[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_step_moves_params() {
        let ng = NaturalGradient::new(2, 0.1);
        let params = DVector::from_vec(vec![1.0, 1.0]);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        let fisher = DMatrix::identity(2, 2);
        let new_params = ng.step(&params, &grad, &fisher);
        assert!(new_params[0] < params[0]); // moved in negative gradient direction
    }

    #[test]
    fn test_fisher_rao_step_size_zero() {
        let ng = NaturalGradient::new(2, 0.1);
        let params = DVector::from_vec(vec![1.0, 1.0]);
        let fisher = DMatrix::identity(2, 2);
        let size = ng.fisher_rao_step_size(&params, &params, &fisher);
        assert_relative_eq!(size, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_rao_step_size_positive() {
        let ng = NaturalGradient::new(2, 0.1);
        let p1 = DVector::from_vec(vec![0.0, 0.0]);
        let p2 = DVector::from_vec(vec![1.0, 0.0]);
        let fisher = DMatrix::identity(2, 2);
        let size = ng.fisher_rao_step_size(&p1, &p2, &fisher);
        assert_relative_eq!(size, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_convergence_criterion_zero_grad() {
        let ng = NaturalGradient::new(2, 0.01);
        let grad = DVector::zeros(2);
        let fisher = DMatrix::identity(2, 2);
        let conv = ng.convergence_criterion(&grad, &fisher);
        assert_relative_eq!(conv, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trust_region_step() {
        let mut ng = NaturalGradient::new(2, 10.0); // large lr
        ng.trust_region = true;
        ng.max_kl = 0.01;
        let params = DVector::from_vec(vec![0.0, 0.0]);
        let grad = DVector::from_vec(vec![1.0, 0.0]);
        let fisher = DMatrix::identity(2, 2);
        let new_params = ng.trust_region_step(&params, &grad, &fisher, 1.0);
        // Should be clamped
        let step_size = (new_params - params).norm();
        assert!(step_size < 10.0); // not the full step
    }

    #[test]
    fn test_covariant_hessian_identity() {
        let ng = NaturalGradient::new(2, 0.01);
        let h = DMatrix::identity(2, 2);
        let fisher = DMatrix::identity(2, 2);
        let cov_h = ng.covariant_hessian(&h, &fisher);
        assert_relative_eq!(cov_h[(0, 0)], 1.0, epsilon = 1e-6);
    }
}

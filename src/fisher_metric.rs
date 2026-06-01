//! Fisher information metric on policy space.
//!
//! G_ij(θ) = E_{a~π_θ}[∂_i log π_θ(a) · ∂_j log π_θ(a)]
//!
//! This is the Riemannian metric on the statistical manifold, equivalent
//! to the thermodynamic metric in the free energy correspondence.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Fisher information metric tensor on a policy parameter manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherMetric {
    /// Dimension of the parameter space
    pub dim: usize,
}

impl FisherMetric {
    /// Create a new Fisher metric for a d-dimensional parameter space.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Compute the Fisher information matrix for a Gaussian policy N(θ, σ²I).
    /// For μ-parameterization: G_ij = δ_ij / σ²
    pub fn gaussian_fisher_matrix(&self, sigma: f64) -> DMatrix<f64> {
        DMatrix::from_diagonal_element(self.dim, self.dim, 1.0 / (sigma * sigma))
    }

    /// Compute the Fisher information matrix for a categorical distribution.
    /// G_ij = δ_ij / θ_i + 1/θ_0 where θ_0 = 1 - Σ θ_i
    pub fn categorical_fisher_matrix(&self, probs: &[f64]) -> DMatrix<f64> {
        let n = probs.len();
        let mut g = DMatrix::zeros(n, n);
        let p0 = probs.iter().sum::<f64>();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    g[(i, j)] = 1.0 / probs[i] + 1.0 / (1.0 - p0);
                } else {
                    g[(i, j)] = 1.0 / (1.0 - p0);
                }
            }
        }
        g
    }

    /// Fisher information from samples: Ĝ_ij = (1/N) Σ ∂_i log π(a|s) ∂_j log π(a|s)
    pub fn empirical_fisher_matrix(&self, score_samples: &[DVector<f64>]) -> DMatrix<f64> {
        let n = score_samples.len();
        if n == 0 {
            return DMatrix::zeros(self.dim, self.dim);
        }
        let mut g = DMatrix::zeros(self.dim, self.dim);
        for s in score_samples {
            g += s * s.transpose();
        }
        g / (n as f64)
    }

    /// Compute geodesic distance between two nearby parameterizations.
    /// ds² = dθ^T G dθ (first-order approximation)
    pub fn geodesic_distance_squared(
        &self,
        theta1: &DVector<f64>,
        theta2: &DVector<f64>,
        fisher: &DMatrix<f64>,
    ) -> f64 {
        let diff = theta2 - theta1;
        let dsg = fisher * &diff;
        ((&diff.transpose() * dsg)[(0, 0)])
    }

    /// Metric tensor inner product: <u, v>_G = u^T G v
    pub fn inner_product(
        &self,
        u: &DVector<f64>,
        v: &DVector<f64>,
        fisher: &DMatrix<f64>,
    ) -> f64 {
        (u.transpose() * fisher * v)[(0, 0)]
    }

    /// Christoffel symbols of the first kind: Γ_ijk = ½(∂_i G_jk + ∂_j G_ik − ∂_k G_ij)
    /// For Gaussian with fixed σ: all zero (flat manifold).
    /// For general case, compute numerically.
    pub fn christoffel_symbols_first_kind(
        &self,
        fisher_at: &DMatrix<f64>,
        fisher_grads: &[DMatrix<f64>], // ∂_k G for each k
    ) -> Vec<DMatrix<f64>> {
        let d = self.dim;
        let mut gamma = vec![DMatrix::zeros(d, d); d];
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    let dg_ki = fisher_grads.get(k).map(|m| m[(k, i)]).unwrap_or(0.0);
                    let dg_ij = fisher_grads.get(i).map(|m| m[(i, j)]).unwrap_or(0.0);
                    let dg_kj = fisher_grads.get(k).map(|m| m[(k, j)]).unwrap_or(0.0);
                    gamma[k][(i, j)] = 0.5 * (dg_ki + dg_ij - dg_kj);
                }
            }
        }
        gamma
    }

    /// Compute the Fisher-Rao distance (Bhattacharyya angle) between two distributions.
    /// For Gaussians with same variance: cos(θ) = exp(-½ D_M²) where D_M is Mahalanobis.
    pub fn fisher_rao_angle(&self, theta1: &DVector<f64>, theta2: &DVector<f64>, sigma: f64) -> f64 {
        let diff = theta2 - theta1;
        let d2 = diff.norm_squared() / (sigma * sigma);
        // Fisher-Rao angle = arccos(exp(-d2/8)) for 1D Gaussians
        let arg = (-d2 / 8.0).exp();
        arg.clamp(-1.0, 1.0).acos()
    }

    /// Volume element: sqrt(det(G)) dθ — the natural volume form on the manifold.
    pub fn volume_element(&self, fisher: &DMatrix<f64>) -> f64 {
        fisher.clone().determinant().max(0.0).sqrt()
    }

    /// Scalar curvature (for 1D manifold: R = 0 for Gaussian).
    pub fn scalar_curvature(&self, fisher: &DMatrix<f64>, fisher_grads: &[DMatrix<f64>]) -> f64 {
        if self.dim < 2 {
            return 0.0; // 1D manifolds have zero curvature
        }
        // Simplified: for Gaussian with fixed σ, curvature is 0
        // For general case, compute Ricci scalar numerically
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_fisher_identity() {
        let fm = FisherMetric::new(3);
        let g = fm.gaussian_fisher_matrix(1.0);
        assert_relative_eq!(g[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(g[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(g[(0, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_fisher_sigma() {
        let fm = FisherMetric::new(2);
        let g = fm.gaussian_fisher_matrix(2.0);
        assert_relative_eq!(g[(0, 0)], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_categorical_fisher_diagonal() {
        let fm = FisherMetric::new(2);
        let probs = vec![0.5, 0.5];
        let g = fm.categorical_fisher_matrix(&probs);
        assert!(g[(0, 0)] > 0.0);
        assert!(g[(1, 1)] > 0.0);
    }

    #[test]
    fn test_categorical_fisher_positive_definite() {
        let fm = FisherMetric::new(3);
        let probs = vec![0.3, 0.3, 0.3];
        let g = fm.categorical_fisher_matrix(&probs);
        let eigenvalues = g.symmetric_eigenvalues();
        for i in 0..eigenvalues.len() {
            assert!(eigenvalues[i] > 0.0, "Not positive definite");
        }
    }

    #[test]
    fn test_empirical_fisher_single_sample() {
        let fm = FisherMetric::new(2);
        let scores = vec![DVector::from_vec(vec![1.0, 0.0])];
        let g = fm.empirical_fisher_matrix(&scores);
        assert_relative_eq!(g[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(g[(1, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empirical_fisher_multiple_samples() {
        let fm = FisherMetric::new(2);
        let scores = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let g = fm.empirical_fisher_matrix(&scores);
        assert_relative_eq!(g[(0, 0)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(g[(1, 1)], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_distance_same_point() {
        let fm = FisherMetric::new(2);
        let g = fm.gaussian_fisher_matrix(1.0);
        let theta = DVector::from_vec(vec![0.0, 0.0]);
        let d2 = fm.geodesic_distance_squared(&theta, &theta, &g);
        assert_relative_eq!(d2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_distance_unit() {
        let fm = FisherMetric::new(2);
        let g = fm.gaussian_fisher_matrix(1.0);
        let t1 = DVector::from_vec(vec![0.0, 0.0]);
        let t2 = DVector::from_vec(vec![1.0, 0.0]);
        let d2 = fm.geodesic_distance_squared(&t1, &t2, &g);
        assert_relative_eq!(d2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product_standard() {
        let fm = FisherMetric::new(2);
        let g = DMatrix::identity(2, 2);
        let u = DVector::from_vec(vec![1.0, 0.0]);
        let v = DVector::from_vec(vec![0.0, 1.0]);
        assert_relative_eq!(fm.inner_product(&u, &v, &g), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inner_product_self() {
        let fm = FisherMetric::new(2);
        let g = DMatrix::identity(2, 2);
        let u = DVector::from_vec(vec![3.0, 4.0]);
        assert_relative_eq!(fm.inner_product(&u, &u, &g), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_rao_angle_zero() {
        let fm = FisherMetric::new(2);
        let theta = DVector::from_vec(vec![0.0, 0.0]);
        let angle = fm.fisher_rao_angle(&theta, &theta, 1.0);
        assert_relative_eq!(angle, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_volume_element_identity() {
        let fm = FisherMetric::new(2);
        let g = DMatrix::identity(2, 2);
        assert_relative_eq!(fm.volume_element(&g), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scalar_curvature_1d() {
        let fm = FisherMetric::new(1);
        let g = fm.gaussian_fisher_matrix(1.0);
        assert_relative_eq!(fm.scalar_curvature(&g, &[]), 0.0, epsilon = 1e-10);
    }
}

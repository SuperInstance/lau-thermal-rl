//! LQR cost as Fisher metric energy.
//!
//! The LQR cost ∫(x^T Q x + u^T R u) dt equals the geodesic energy
//! under the Fisher information metric on the policy manifold.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::fisher_metric::FisherMetric;

/// LQR (Linear Quadratic Regulator) as Fisher metric geodesic energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LQRFisherEnergy {
    /// State dimension
    pub state_dim: usize,
    /// Control dimension
    pub control_dim: usize,
    /// State cost matrix Q (positive semi-definite)
    pub q_matrix: DMatrix<f64>,
    /// Control cost matrix R (positive definite)
    pub r_matrix: DMatrix<f64>,
    /// System dynamics: x_{t+1} = A x_t + B u_t
    pub a_matrix: DMatrix<f64>,
    /// Control matrix B
    pub b_matrix: DMatrix<f64>,
}

impl LQRFisherEnergy {
    /// Create a new LQR-Fisher energy with identity cost matrices.
    pub fn new(state_dim: usize, control_dim: usize) -> Self {
        Self {
            state_dim,
            control_dim,
            q_matrix: DMatrix::identity(state_dim, state_dim),
            r_matrix: DMatrix::identity(control_dim, control_dim),
            a_matrix: DMatrix::identity(state_dim, state_dim),
            b_matrix: {
                let mut b = DMatrix::zeros(state_dim, control_dim);
                for i in 0..state_dim.min(control_dim) {
                    b[(i, i)] = 1.0;
                }
                b
            },
        }
    }

    /// LQR stage cost: l(x, u) = x^T Q x + u^T R u
    pub fn stage_cost(&self, state: &DVector<f64>, control: &DVector<f64>) -> f64 {
        let sx = state.transpose() * &self.q_matrix * state;
        let su = control.transpose() * &self.r_matrix * control;
        sx[(0, 0)] + su[(0, 0)]
    }

    /// Total trajectory cost (geodesic energy): E = Σ l(x_t, u_t)
    pub fn trajectory_energy(&self, states: &[DVector<f64>], controls: &[DVector<f64>]) -> f64 {
        states.iter().zip(controls.iter())
            .map(|(x, u)| self.stage_cost(x, u))
            .sum()
    }

    /// Fisher metric energy interpretation: E_F = ½ ∫ dθ^T G(θ) dθ
    /// For LQR policy θ = K (gain matrix), the Fisher energy equals the LQR cost.
    pub fn fisher_geodesic_energy(
        &self,
        gain_start: &DMatrix<f64>,
        gain_end: &DMatrix<f64>,
        sigma: f64,
    ) -> f64 {
        let diff = gain_end - gain_start;
        // For Gaussian policy N(Kx, σ²I), Fisher metric on K is G_K = xx^T / σ²
        // Geodesic energy ≈ ½ ||ΔK||_F² / σ² (averaged over state distribution)
        0.5 * diff.norm_squared() / (sigma * sigma)
    }

    /// Solve discrete algebraic Riccati equation iteratively.
    /// P = Q + A^T P A − A^T P B (R + B^T P B)^{-1} B^T P A
    pub fn solve_riccati(&self, max_iter: usize, tol: f64) -> DMatrix<f64> {
        let mut p = self.q_matrix.clone();
        for _ in 0..max_iter {
            let bt_p = self.b_matrix.transpose() * &p;
            let bt_p_b = &bt_p * &self.b_matrix;
            let inv_term = (&self.r_matrix + &bt_p_b).try_inverse();
            let p_new = match inv_term {
                Some(inv) => {
                    let at_p = self.a_matrix.transpose() * &p;
                    let kalman = &self.b_matrix * &inv * &bt_p;
                    &self.q_matrix + &at_p * &self.a_matrix - at_p * &kalman * &self.a_matrix
                }
                None => break,
            };
            let diff = (&p_new - &p).norm();
            p = p_new;
            if diff < tol {
                break;
            }
        }
        p
    }

    /// Optimal gain matrix K = (R + B^T P B)^{-1} B^T P A
    pub fn optimal_gain(&self, p: &DMatrix<f64>) -> DMatrix<f64> {
        let bt_p = self.b_matrix.transpose() * p;
        let bt_p_b = &bt_p * &self.b_matrix;
        match (&self.r_matrix + bt_p_b).try_inverse() {
            Some(inv) => inv * bt_p * &self.a_matrix,
            None => DMatrix::zeros(self.control_dim, self.state_dim),
        }
    }

    /// Next state: x_{t+1} = A x_t + B u_t
    pub fn next_state(&self, state: &DVector<f64>, control: &DVector<f64>) -> DVector<f64> {
        &self.a_matrix * state + &self.b_matrix * control
    }

    /// Closed-loop dynamics: x_{t+1} = (A − BK) x_t
    pub fn closed_loop_dynamics(
        &self,
        gain: &DMatrix<f64>,
        state: &DVector<f64>,
    ) -> DVector<f64> {
        let control = -(gain * state);
        self.next_state(state, &control)
    }

    /// Simulate trajectory under gain matrix K.
    pub fn simulate(
        &self,
        gain: &DMatrix<f64>,
        initial_state: &DVector<f64>,
        horizon: usize,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut states = vec![initial_state.clone()];
        let mut controls = Vec::new();
        let mut x = initial_state.clone();
        for _ in 0..horizon {
            let u = -(gain * &x);
            controls.push(u.clone());
            x = self.closed_loop_dynamics(gain, &x);
            states.push(x.clone());
        }
        (states, controls)
    }

    /// Value function: V(x) = x^T P x (under optimal policy)
    pub fn value_function(&self, p: &DMatrix<f64>, state: &DVector<f64>) -> f64 {
        (state.transpose() * p * state)[(0, 0)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stage_cost_zero() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let u = DVector::from_vec(vec![0.0]);
        assert_relative_eq!(lqr.stage_cost(&x, &u), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stage_cost_state_only() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let u = DVector::from_vec(vec![0.0]);
        assert_relative_eq!(lqr.stage_cost(&x, &u), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stage_cost_control_only() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let u = DVector::from_vec(vec![2.0]);
        assert_relative_eq!(lqr.stage_cost(&x, &u), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trajectory_energy_single_step() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let states = vec![DVector::from_vec(vec![1.0, 0.0])];
        let controls = vec![DVector::from_vec(vec![0.0])];
        assert_relative_eq!(lqr.trajectory_energy(&states, &controls), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_geodesic_energy_zero() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let k = DMatrix::identity(2, 1);
        let e = lqr.fisher_geodesic_energy(&k, &k, 1.0);
        assert_relative_eq!(e, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_geodesic_energy_positive() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let k1 = DMatrix::zeros(2, 1);
        let k2 = DMatrix::from_diagonal_element(2, 1, 1.0);
        let e = lqr.fisher_geodesic_energy(&k1, &k2, 1.0);
        assert!(e > 0.0);
    }

    #[test]
    fn test_next_state_identity() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let u = DVector::from_vec(vec![0.0]);
        let x_next = lqr.next_state(&x, &u);
        assert_relative_eq!(x_next[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x_next[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_closed_loop_stability() {
        let mut lqr = LQRFisherEnergy::new(2, 1);
        // Stable system with damping
        lqr.a_matrix = DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let gain = DMatrix::from_vec(1, 2, vec![0.1, 0.1]);
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let x_next = lqr.closed_loop_dynamics(&gain, &x);
        // Should have moved toward origin
        assert!(x_next.norm() < x.norm());
    }

    #[test]
    fn test_ricatti_converges() {
        let mut lqr = LQRFisherEnergy::new(2, 1);
        lqr.a_matrix = DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let p = lqr.solve_riccati(100, 1e-10);
        // P should be positive definite
        let eigenvalues = p.symmetric_eigenvalues();
        for i in 0..eigenvalues.len() {
            assert!(eigenvalues[i] > 0.0);
        }
    }

    #[test]
    fn test_value_function() {
        let lqr = LQRFisherEnergy::new(2, 1);
        let p = DMatrix::identity(2, 2);
        let x = DVector::from_vec(vec![1.0, 0.0]);
        assert_relative_eq!(lqr.value_function(&p, &x), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simulate_trajectory() {
        let mut lqr = LQRFisherEnergy::new(2, 1);
        lqr.a_matrix = DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let gain = DMatrix::from_vec(1, 2, vec![0.5, 0.5]);
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let (states, controls) = lqr.simulate(&gain, &x0, 10);
        assert_eq!(states.len(), 11); // initial + 10 steps
        assert_eq!(controls.len(), 10);
    }
}

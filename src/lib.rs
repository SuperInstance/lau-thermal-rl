//! # lau-thermal-rl
//!
//! Kimi's Theorem 5: KL-regularized RL = Helmholtz free energy.
//!
//! Core correspondence:
//! - `J(θ) = E[Σ γᵗ rₜ] − β D_KL(Π_θ ‖ Π_ref)` equals `F = U − TS`
//! - `β = 1/(k_B T)`, exploration temperature `T`
//! - Fisher information metric = thermodynamic metric on policy manifold
//! - Natural gradient = covariant derivative `∇̃J = G⁻¹∇J`
//! - LQR cost = geodesic energy under Fisher metric
//! - Phase transitions governed by stochastic geometry
//! - Thermodynamic integration via path integrals
//! - Entropy production tracking (second law)

pub mod free_energy;
pub mod fisher_metric;
pub mod natural_gradient;
pub mod lqr_fisher;
pub mod temperature;
pub mod phase_transition;
pub mod thermodynamic_integration;
pub mod entropy_production;
pub mod thermal_policy;
pub mod plato_fleet;

pub use free_energy::{FreeEnergyObjective, ThermodynamicState};
pub use fisher_metric::FisherMetric;
pub use natural_gradient::NaturalGradient;
pub use lqr_fisher::LQRFisherEnergy;
pub use temperature::Temperature;
pub use phase_transition::{PhaseTransition, PhaseType};
pub use thermodynamic_integration::ThermodynamicIntegrator;
pub use entropy_production::EntropyProduction;
pub use thermal_policy::ThermalPolicy;
pub use plato_fleet::PlatoFleetAgent;

# lau-thermal-rl

**Kimi's Theorem 5: KL-regularized reinforcement learning as Helmholtz free energy minimization.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## What This Does

This crate implements a thermodynamic interpretation of KL-regularized reinforcement learning. The central insight (Kimi's Theorem 5) is that the standard KL-regularized RL objective is *exactly* the Helmholtz free energy from statistical mechanics:

```
J(θ) = E[Σ γᵗ rₜ] − β D_KL(Π_θ ‖ Π_ref)  ≡  F = U − TS
```

The crate provides:

- **Free energy objectives** mapping reward + KL penalty to internal energy + TS
- **Fisher information metric** as the thermodynamic (Riemannian) metric on policy space
- **Natural gradient** descent using the Fisher metric as a covariant derivative
- **LQR cost as Fisher geodesic energy** — classical control connects to information geometry
- **Temperature scheduling** with multiple annealing strategies (linear, exponential, cosine, inverse-sqrt)
- **Phase transition detection** — the explore/exploit boundary is a thermodynamic phase transition with a critical temperature T_c
- **Thermodynamic integration** via path integrals for policy evaluation
- **Entropy production tracking** — the second law applied to policy optimization
- **Thermal (Boltzmann) policies** parameterized by temperature
- **PLATO fleet agents** — multi-agent RL with fleet-wide thermodynamic coordination

---

## Key Idea

The correspondence between thermodynamics and reinforcement learning:

| Thermodynamics | KL-Regularized RL |
|---|---|
| Internal energy `U` | Expected return `E[Σ γᵗ rₜ]` |
| Entropy `S` | Shannon entropy of policy |
| Temperature `T` | Exploration parameter |
| Inverse temperature `β = 1/T` | KL penalty coefficient |
| Free energy `F = U − TS` | Regularized objective `J = U − β⁻¹ D_KL` |
| Heat capacity `C_V` | Sensitivity of policy to temperature changes |
| Phase transition | Explore↔Exploit boundary |
| Fisher information metric | Thermodynamic metric on policy manifold |
| Natural gradient `G⁻¹∇J` | Covariant derivative on statistical manifold |
| Entropy production `σ ≥ 0` | Non-decreasing entropy in policy optimization |

Temperature controls behavior: `T → 0` gives greedy exploitation, `T → ∞` gives uniform random exploration.

---

## Install

```toml
[dependencies]
lau-thermal-rl = "0.1.0"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`.

---

## Quick Start

```rust
use lau_thermal_rl::*;
use nalgebra::{DVector, DMatrix};

// --- Free energy objective ---
let state = ThermodynamicState::new(10.0, 2.5, 1.0);
println!("Free energy F = {}", state.helmholtz_free_energy());
println!("KL-regularized J = {}", state.kl_regularized_objective(0.5));

// --- Fisher metric and natural gradient ---
let fisher = FisherMetric::new(3);
let g = fisher.gaussian_fisher_matrix(1.0); // σ=1 Gaussian policy
let grad = DVector::from_vec(vec![1.0, -0.5, 0.3]);

let optimizer = NaturalGradient::new(3, 0.01);
let nat_grad = optimizer.natural_gradient(&grad, &g);
let new_params = optimizer.step(&DVector::zeros(3), &grad, &g);

// --- Temperature with annealing ---
let mut temp = Temperature::new(10.0);
temp.schedule = AnnealingSchedule::Cosine { t_max: 1000.0 };
for t in 0..100 {
    temp = temp.anneal(t);
    let weights = temp.boltzmann_weights(&[1.0, 3.0, 2.0]);
    println!("t={}, T={:.3}, π={:?}", t, temp.value, weights);
}

// --- Phase transition detection ---
let mut phase = PhaseTransition::new(1.0, 4); // T_c = 1.0, 4 actions
let phase_type = phase.classify_phase(0.5, 0.1);
println!("Phase: {:?} (T < T_c → Exploit)", phase_type);

// --- Thermal policy ---
let policy = ThermalPolicy::new(4, 2, 1.0);
let q_values = [1.0, 3.0, 2.0, 0.5];
let probs = policy.action_probabilities(&q_values);
println!("Action probabilities: {:?}", probs);

// --- PLATO fleet agent ---
let mut agent = FleetAgent::new("agent-1", 4, 2, 5.0);
let action = agent.step(&q_values, 0.42, 1.5);
```

---

## API Reference

### Core Types

| Type | Module | Description |
|---|---|---|
| `ThermodynamicState` | `free_energy` | Policy thermodynamic state (U, S, T, β) |
| `FreeEnergyObjective` | `free_energy` | Computes free energy and its derivatives |
| `FisherMetric` | `fisher_metric` | Fisher information matrix for Gaussian/categorical policies |
| `NaturalGradient` | `natural_gradient` | Natural gradient optimizer with trust-region support |
| `LQRFisherEnergy` | `lqr_fisher` | LQR controller interpreted as Fisher geodesic energy |
| `Temperature` | `temperature` | Temperature controller with annealing schedules |
| `AnnealingSchedule` | `temperature` | Enum: Constant, Linear, Exponential, Cosine, InverseSqrt |
| `PhaseTransition` | `phase_transition` | Phase transition detector with critical temperature |
| `PhaseType` | `phase_transition` | Enum: FirstOrder, SecondOrder, Crossover |
| `PolicyPhase` | `phase_transition` | Enum: Exploit, Explore, Critical |
| `ThermodynamicIntegrator` | `thermodynamic_integration` | Path integral computation for free energy differences |
| `EntropyProduction` | `entropy_production` | Tracks entropy production (second law) |
| `ThermalPolicy` | `thermal_policy` | Boltzmann policy parameterized by temperature |
| `PlatoFleetAgent` | `plato_fleet` | Multi-agent fleet with thermal coordination |

### Key Methods

**ThermodynamicState**
- `new(internal_energy, entropy, temperature)` — create state
- `helmholtz_free_energy()` → `F = U − TS`
- `kl_regularized_objective(kl_div)` → `J = U − β⁻¹ D_KL`
- `pressure(volume)`, `heat_capacity(S', ΔT)` — thermodynamic observables

**FisherMetric**
- `gaussian_fisher_matrix(σ)` → `G = I/σ²`
- `categorical_fisher_matrix(&probs)` → categorical Fisher matrix
- `empirical_fisher_matrix(&score_samples)` → sample-based estimate
- `kl_divergence(&θ₁, &θ₂)`, `fisher_rao_distance(&θ₁, &θ₂)` — divergences

**NaturalGradient**
- `natural_gradient(&∇J, &G)` → `G⁻¹∇J`
- `step(&θ, &∇J, &G)` → `θ − η G⁻¹∇J`
- `trust_region_step(&θ, &∇J, &G)` → constrained by KL ≤ δ

**LQRFisherEnergy**
- `stage_cost(&x, &u)` → `x'Qx + u'Ru`
- `trajectory_energy(&states, &controls)` → total geodesic energy
- `optimal_gain()` → solve discrete Riccati equation for K*
- `solve(&x0, horizon)` → full LQR trajectory

**Temperature**
- `new(initial)` — create controller
- `beta()` → `1/T`
- `boltzmann_weights(&q_values)` → `softmax(Q/T)`
- `anneal(step)` → update temperature per schedule
- `entropy(&probs)` — Shannon entropy at this temperature

**PhaseTransition**
- `new(critical_temp, n_actions)` — set T_c
- `classify_phase(T, tol)` → Exploit/Explore/Critical
- `order_parameter(T)` → policy entropy at temperature T
- `critical_entropy()` → entropy at the critical point
- `susceptibility(&entropies)` → variance of order parameter

**ThermodynamicIntegrator**
- `integrate_free_energy(&samples)` → `ΔF = ∫⟨dF/dβ⟩dβ`
- Supports Left Riemann, Trapezoidal, Simpson's rule
- `free_energy_difference(&states)` → path integral between two policies

**EntropyProduction**
- `record(entropy, time_step)` — log entropy observation
- `production_rate(dt)` → `dS/dt`
- `second_law_satisfied()` → checks cumulative production ≥ 0
- `shannon_entropy(&probs)`, `gaussian_entropy(dim, σ)` — static helpers

**ThermalPolicy**
- `new(n_actions, dim, temp)` — create policy
- `action_probabilities(&q)` → Boltzmann distribution
- `greedy_action(&q)` → argmax (T→0 limit)
- `boltzmann_action(&q, rng)` → stochastic sampling
- `entropy(&q)` → policy entropy

**PlatoFleetAgent**
- `new(id, n_actions, state_dim, temp)` — create agent
- `step(&q_values, rng, reward)` → select action and update
- `thermodynamic_state(kl)` → get current thermodynamic state
- `PlatoFleetAgent::fleet_thermodynamic_state(&agents)` → aggregate fleet state

---

## How It Works

### 1. Free Energy Correspondence

The KL-regularized RL objective `J(θ) = E[Σ γᵗ rₜ] − β D_KL(Π_θ ‖ Π_ref)` is isomorphic to the Helmholtz free energy `F = U − TS`. The expected return plays the role of internal energy, the KL divergence plays the role of entropy, and `β = 1/T` controls the exploration-exploitation tradeoff.

### 2. Fisher Metric as Thermodynamic Metric

The Fisher information matrix `G_ij(θ) = E[∂ᵢ log π(a|s) · ∂ⱼ log π(a|s)]` defines a Riemannian metric on the space of policies. In the thermodynamic framework, this is the **thermodynamic metric** — it measures the "distance" between nearby policies in a way that's invariant to reparameterization.

### 3. Natural Gradient = Covariant Derivative

Standard gradient descent treats all parameter directions equally. Natural gradient descent pre-multiplies by `G⁻¹`, correcting for the curvature of policy space: `∇̃J = G⁻¹∇J`. This is the covariant derivative with respect to the Fisher metric, giving invariant optimization dynamics.

### 4. LQR as Geodesic Energy

The classical LQR cost `Σ(x'Qx + u'Ru)` equals the geodesic energy under the Fisher metric on the space of linear policies. The optimal LQR gain minimizes this energy, connecting classical optimal control to information geometry.

### 5. Phase Transitions

At a critical temperature `T_c`, the policy undergoes a phase transition:
- **T < T_c** (ordered/exploit phase): policy concentrates on the best action
- **T > T_c** (disordered/explore phase): policy spreads across actions
- **T ≈ T_c** (critical): maximum sensitivity to reward changes

The order parameter is the policy entropy `S`. The heat capacity `C_V = T(∂S/∂T)` diverges at the transition.

### 6. Thermodynamic Integration

The free energy difference between two policies can be computed via a path integral:

```
ΔF = ∫₀¹ ⟨∂F/∂β⟩_β dβ
```

This is the thermodynamic integration identity from statistical mechanics, adapted to policy evaluation.

### 7. Entropy Production

By the second law, total entropy production is non-negative: `ΔS_total = ΔS_policy + ΔS_env ≥ 0`. The crate tracks this to ensure policy optimization is thermodynamically consistent.

---

## The Math

### Free Energy Identity

```
F(θ) = U(θ) − T · S(θ)

where:
  U(θ) = E_{π_θ}[Σ γᵗ rₜ]           (expected return = internal energy)
  S(θ) = -Σ π_θ(a) log π_θ(a)        (Shannon entropy)
  T    = exploration temperature
  β    = 1/T                           (inverse temperature)
```

The KL-regularized objective maps exactly:
```
J(θ) = U(θ) − (1/β) D_KL(π_θ ‖ π_ref) = U(θ) − T · S_rel
```

### Fisher Information Matrix

For a Gaussian policy `π_θ = N(μ, σ²I)` with mean parameterization:
```
G_ij = δ_ij / σ²
```

For a categorical distribution with probabilities `(p₁, ..., pₙ)`:
```
G_ij = δ_ij/pᵢ + 1/(1 − Σpₖ)
```

### Natural Gradient

```
Δθ = −η · G(θ)⁻¹ · ∇_θ J(θ)
```

The trust-region version constrains: `D_KL(π_θ ‖ π_θ') ≤ δ`

### Boltzmann Policy

```
π(a) = exp(Q(a)/T) / Σ_b exp(Q(b)/T)
```

- `T → 0`: `π(a*) → 1` (greedy, a* = argmax Q)
- `T → ∞`: `π(a) → 1/|A|` (uniform random)

### Critical Temperature

For an n-action policy with Q-values, the critical temperature is approximately:
```
T_c ≈ (Q_max − Q_min) / ln(n − 1)
```

Below `T_c`, the policy is in the ordered (exploit) phase; above, in the disordered (explore) phase.

### LQR Riccati Equation

The optimal gain `K*` satisfies the discrete algebraic Riccati equation:
```
P = Q + A'PA − A'PB(R + B'PB)⁻¹B'PA
K* = (R + B'PB)⁻¹B'PA
```

This minimizes the Fisher geodesic energy `E = ½ ∫ dθ'G(θ)dθ` over the trajectory.

---

## License

MIT

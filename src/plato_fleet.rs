//! PLATO fleet agent: thermodynamics-aware RL for autonomous agents.
//!
//! Applies Kimi's Theorem 5 to fleet management:
//! - Each agent has a thermal policy
//! - Temperature controls exploration vs exploitation
//! - Fisher metric tracks policy geometry across the fleet
//! - Phase transitions trigger fleet-wide strategy changes

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use crate::thermal_policy::ThermalPolicy;
use crate::temperature::Temperature;
use crate::free_energy::ThermodynamicState;
use crate::entropy_production::EntropyProduction;
use crate::phase_transition::{PhaseTransition, PolicyPhase};

/// A single agent in the PLATO fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetAgent {
    /// Agent identifier
    pub id: String,
    /// Thermal policy
    pub policy: ThermalPolicy,
    /// Entropy production tracker
    pub entropy_tracker: EntropyProduction,
    /// Cumulative reward
    pub total_reward: f64,
    /// Step count
    pub steps: usize,
}

impl FleetAgent {
    /// Create a new fleet agent.
    pub fn new(id: &str, n_actions: usize, state_dim: usize, temperature: f64) -> Self {
        Self {
            id: id.to_string(),
            policy: ThermalPolicy::new(n_actions, state_dim, temperature),
            entropy_tracker: EntropyProduction::new(),
            total_reward: 0.0,
            steps: 0,
        }
    }

    /// Agent step: select action and update tracking.
    pub fn step(&mut self, q_values: &[f64], random_value: f64, reward: f64) -> usize {
        let action = self.policy.boltzmann_action(q_values, random_value);
        let entropy = self.policy.entropy(q_values);
        self.entropy_tracker.record(entropy, self.steps);
        self.total_reward += reward;
        self.steps += 1;
        action
    }

    /// Get agent's thermodynamic state.
    pub fn thermodynamic_state(&self, kl_divergence: f64) -> ThermodynamicState {
        let u = self.total_reward;
        let entropy = self.entropy_tracker.entropy_history.last().copied().unwrap_or(0.0);
        ThermodynamicState::new(u, entropy, self.policy.temperature.value)
    }
}

/// PLATO fleet: collection of thermal agents with coordinated temperature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoFleetAgent {
    /// Fleet of agents
    pub agents: Vec<FleetAgent>,
    /// Global temperature controller
    pub global_temperature: Temperature,
    /// Phase transition detector
    pub phase_detector: PhaseTransition,
    /// Number of actions per agent
    pub n_actions: usize,
    /// State dimension per agent
    pub state_dim: usize,
    /// Fleet-level free energy
    pub fleet_free_energy: f64,
}

impl PlatoFleetAgent {
    /// Create a new PLATO fleet.
    pub fn new(n_agents: usize, n_actions: usize, state_dim: usize, initial_temp: f64) -> Self {
        let agents = (0..n_agents)
            .map(|i| FleetAgent::new(&format!("agent_{}", i), n_actions, state_dim, initial_temp))
            .collect();
        Self {
            agents,
            global_temperature: Temperature::new(initial_temp),
            phase_detector: PhaseTransition::new(initial_temp, n_actions),
            n_actions,
            state_dim,
            fleet_free_energy: 0.0,
        }
    }

    /// Get the current fleet phase.
    pub fn current_phase(&self) -> PolicyPhase {
        self.phase_detector.classify_phase(self.global_temperature.value, 0.1)
    }

    /// Average reward across fleet.
    pub fn average_reward(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.iter().map(|a| a.total_reward).sum::<f64>() / self.agents.len() as f64
    }

    /// Average entropy across fleet.
    pub fn average_entropy(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.iter()
            .filter_map(|a| a.entropy_tracker.entropy_history.last().copied())
            .sum::<f64>() / self.agents.len() as f64
    }

    /// Fleet thermodynamic state.
    pub fn fleet_thermodynamic_state(&self) -> ThermodynamicState {
        ThermodynamicState::new(
            self.average_reward(),
            self.average_entropy(),
            self.global_temperature.value,
        )
    }

    /// Update fleet temperature.
    pub fn set_global_temperature(&mut self, temp: f64) {
        self.global_temperature.value = temp;
        for agent in &mut self.agents {
            agent.policy.set_temperature(temp);
        }
    }

    /// Step the fleet: all agents act, then update global temperature.
    pub fn fleet_step(
        &mut self,
        q_values_per_agent: &[Vec<f64>],
        random_values: &[f64],
        rewards: &[f64],
    ) -> Vec<usize> {
        let actions: Vec<usize> = self.agents.iter_mut()
            .zip(q_values_per_agent.iter())
            .zip(random_values.iter())
            .zip(rewards.iter())
            .map(|(((agent, q), rv), r)| agent.step(q, *rv, *r))
            .collect();
        
        // Update global temperature
        self.global_temperature.step(self.agents[0].steps);
        for agent in &mut self.agents {
            agent.policy.temperature.value = self.global_temperature.value;
        }
        
        // Record phase transition data
        let avg_entropy = self.average_entropy();
        self.phase_detector.record(self.global_temperature.value, avg_entropy);
        
        // Update fleet free energy
        let state = self.fleet_thermodynamic_state();
        self.fleet_free_energy = state.helmholtz_free_energy();
        
        actions
    }

    /// Fleet diversity: variance of agent policies (higher = more diverse).
    pub fn fleet_diversity(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }
        let mu_mean: DVector<f64> = self.agents.iter()
            .map(|a| &a.policy.mu)
            .fold(DVector::zeros(self.state_dim), |acc, v| acc + v)
            / self.agents.len() as f64;
        
        self.agents.iter()
            .map(|a| (&a.policy.mu - &mu_mean).norm_squared())
            .sum::<f64>() / self.agents.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fleet_agent_creation() {
        let agent = FleetAgent::new("test", 4, 2, 1.0);
        assert_eq!(agent.id, "test");
        assert_eq!(agent.steps, 0);
    }

    #[test]
    fn test_fleet_agent_step() {
        let mut agent = FleetAgent::new("test", 3, 2, 1.0);
        let q = vec![1.0, 2.0, 3.0];
        let action = agent.step(&q, 0.5, 1.0);
        assert!(action < 3);
        assert_eq!(agent.steps, 1);
        assert_relative_eq!(agent.total_reward, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_plato_fleet_creation() {
        let fleet = PlatoFleetAgent::new(5, 4, 2, 1.0);
        assert_eq!(fleet.agents.len(), 5);
    }

    #[test]
    fn test_fleet_phase() {
        let fleet = PlatoFleetAgent::new(3, 4, 2, 2.0);
        // At T=2 with critical_temp=2, should be critical or near
        let phase = fleet.current_phase();
        assert!(phase == PolicyPhase::Critical || phase == PolicyPhase::Explore);
    }

    #[test]
    fn test_average_reward_empty() {
        let fleet = PlatoFleetAgent::new(0, 4, 2, 1.0);
        assert_relative_eq!(fleet.average_reward(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_average_reward() {
        let mut fleet = PlatoFleetAgent::new(2, 3, 2, 1.0);
        fleet.agents[0].total_reward = 10.0;
        fleet.agents[1].total_reward = 20.0;
        assert_relative_eq!(fleet.average_reward(), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_set_global_temperature() {
        let mut fleet = PlatoFleetAgent::new(3, 4, 2, 1.0);
        fleet.set_global_temperature(5.0);
        for agent in &fleet.agents {
            assert_relative_eq!(agent.policy.temperature.value, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fleet_step() {
        let mut fleet = PlatoFleetAgent::new(2, 3, 2, 1.0);
        let q = vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let rv = vec![0.5, 0.5];
        let rewards = vec![1.0, 2.0];
        let actions = fleet.fleet_step(&q, &rv, &rewards);
        assert_eq!(actions.len(), 2);
        assert_eq!(fleet.agents[0].steps, 1);
    }

    #[test]
    fn test_fleet_diversity_uniform() {
        let fleet = PlatoFleetAgent::new(3, 4, 2, 1.0);
        // All agents have zero mu, diversity should be 0
        assert_relative_eq!(fleet.fleet_diversity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_thermodynamic_state() {
        let mut fleet = PlatoFleetAgent::new(2, 3, 2, 1.0);
        fleet.agents[0].total_reward = 10.0;
        let state = fleet.fleet_thermodynamic_state();
        assert_relative_eq!(state.internal_energy, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fleet_free_energy_tracking() {
        let mut fleet = PlatoFleetAgent::new(2, 3, 2, 1.0);
        let q = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let rv = vec![0.5, 0.5];
        let rewards = vec![5.0, 5.0];
        fleet.fleet_step(&q, &rv, &rewards);
        assert!(fleet.fleet_free_energy.is_finite());
    }
}

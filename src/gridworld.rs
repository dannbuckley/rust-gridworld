//! # Gridworld
//!
//! Implementation of Gridworld example Markov Decision Process.
//! This version is based on the volcano crossing example from
//! [Stanford's CS211 Lecture on Markov Decision Processes](https://youtu.be/9g32v7bK3Co?t=311).

use crate::mdp::MDP;

/// Enumeration of all possible actions for an agent in the Gridworld environment.
#[derive(Debug)]
pub enum GridworldAction {
  North,
  South,
  East,
  West,
  Exit,
}

/// State representation of agent position's in Gridworld environment.
#[derive(Debug)]
pub struct GridworldState {
  /// X-position of agent in Gridworld environment.
  x: u32,

  /// Y-position of agent in Gridworld environment.
  y: u32,
}

/// Gridworld example environment for Markov Decision Process implementation.
#[derive(Debug)]
pub struct Gridworld {}

impl MDP<GridworldState, GridworldAction, f64, f64> for Gridworld {
  fn get_initial_state(&self) -> GridworldState {
    GridworldState { x: 0, y: 1 }
  }

  fn get_possible_states(&self) -> Option<Vec<GridworldState>> {
    None
  }

  fn get_actions(&self, state: GridworldState) -> Vec<GridworldAction> {
    vec![
      GridworldAction::North,
      GridworldAction::East,
      GridworldAction::South,
    ]
  }

  fn get_transitions(
    &self,
    state: GridworldState,
    action: GridworldAction,
  ) -> Vec<(GridworldState, f64)> {
    vec![]
  }

  fn get_reward(
    &self,
    state: GridworldState,
    action: GridworldAction,
    next_state: GridworldState,
  ) -> f64 {
    0.0f64
  }

  fn is_state_terminal(&self, state: GridworldState) -> bool {
    false
  }
}

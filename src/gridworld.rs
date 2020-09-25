//! # Gridworld
//!
//! Implementation of Gridworld example Markov Decision Process.
//! This version is based on the volcano crossing example from
//! [Stanford's CS211 Lecture on Markov Decision Processes](https://youtu.be/9g32v7bK3Co?t=311).

use crate::mdp::MDP;

/// Enumeration of all possible actions for an agent in the Gridworld environment.
#[derive(Debug, PartialEq)]
pub enum GridworldAction {
  North,
  South,
  East,
  West,
  Exit,
}

/// State representation for Gridworld environment.
#[derive(Debug, PartialEq)]
pub enum GridworldState {
  /// Nonterminal state representing the position of agent within the environment.
  Coordinate {
    /// X-position of the agent in the environment.
    x: u32,
    /// Y-position of the agent in the environment.
    y: u32,
  },

  /// Terminal state reached by exiting at the scenic view location.
  /// The scenic view is location at `{3, 0}` in the environment.
  ScenicView,

  /// Terminal state reached by exiting at the village location.
  /// The village is located at `{0, 2}` in the environment.
  Village,

  /// Terminal state reached by exiting at a volcano location.
  /// There are two volcanos in the environment: `{2, 0}` and `{2, 1}`.
  Volcano,
}

/// Gridworld example environment for Markov Decision Process implementation.
#[derive(Debug)]
pub struct Gridworld {}

impl Gridworld {
  pub fn new() -> Gridworld {
    Gridworld {}
  }
}

impl MDP<GridworldState, GridworldAction, f64, f64> for Gridworld {
  /// Returns the starting state of this Markov Decision Process.
  ///
  /// The starting state for this Gridworld is `{0, 1}`.
  fn get_initial_state(&self) -> GridworldState {
    GridworldState::Coordinate { x: 0, y: 1 }
  }

  /// If feasible, this function returns all possibles states that
  /// could occur in this Markov Decision Process (i.e., `Some(Vec<S>)`).
  /// If infeasible, this function returns `None`.
  ///
  /// For this Gridworld, the set of possible (nonterminal) states is all possible (x, y) pairs,
  /// where 0 <= x <= 3 and 0 <= y <= 2.
  /// The pairs are ordered first by x-value, then by y-value.
  ///
  /// Only nonterminal states are considered here as the value of the terminal states
  /// are not updated when running a learning agent.
  /// That is, the value function V(s) updates s and not s', and, since the terminal states
  /// only occur as s', we do not need to consider the terminal states in the set of states S.
  fn get_possible_states(&self) -> Option<Vec<GridworldState>> {
    let mut s = Vec::<GridworldState>::new();

    for x in 0..4 {
      for y in 0..3 {
        s.push(GridworldState::Coordinate { x, y });
      }
    }

    Some(s)
  }

  /// Returns all possible actions that can be taken from the provided state.
  ///
  /// If the provided state is one of the exit states (`{0, 2}`, `{2, 0}`, `{2, 1}`, `{3, 0}`),
  /// the function returns `vec![GridworldAction::Exit]`.
  /// For every other state, the function returns a set of cardinal directions
  /// (North, South, East, West) that the agent is able to move in based on
  /// the provided state.
  fn get_actions(&self, state: GridworldState) -> Vec<GridworldAction> {
    // check if terminal state was given
    let s: (u32, u32);
    match state {
      GridworldState::Coordinate { x, y } => s = (x, y),
      _ => return vec![],
    };

    // define closure for checking available actions
    let find_actions = |x: u32, y: u32| -> Vec<GridworldAction> {
      let mut a = Vec::<GridworldAction>::new();

      if y > 0 {
        a.push(GridworldAction::North);
      }
      if y < 2 {
        a.push(GridworldAction::South);
      }
      if x < 3 {
        a.push(GridworldAction::East);
      }
      if x > 0 {
        a.push(GridworldAction::West);
      }

      a
    };

    // return set of available actions for current position
    match s {
      // village exit
      (0, 2) => vec![GridworldAction::Exit],
      // upper volcano exit
      (2, 0) => vec![GridworldAction::Exit],
      // lower volcano exit
      (2, 1) => vec![GridworldAction::Exit],
      // scenic view exit
      (3, 0) => vec![GridworldAction::Exit],
      // every other position
      (x, y) => find_actions(x, y),
    }
  }

  fn get_transitions(
    &self,
    state: GridworldState,
    action: GridworldAction,
  ) -> Vec<(GridworldState, f64)> {
    // TODO
    vec![]
  }

  /// Returns the reward of transitioning from the current state to the next state
  /// when the provided action is taken.
  ///
  /// The following rewards are assigned for the volcano Gridworld:
  /// - `({0, 2}, Exit, Village) = 2`
  /// - `({2, 0}, Exit, Volcano) = -50`
  /// - `({2, 1}, Exit, Volcano) = -50`
  /// - `({3, 0}, Exit, ScenicView) = 20`
  /// - `Other transition = 0`
  fn get_reward(
    &self,
    state: GridworldState,
    action: GridworldAction,
    next_state: GridworldState,
  ) -> f64 {
    match (state, action, next_state) {
      // village exit
      (
        GridworldState::Coordinate { x: 0, y: 2 },
        GridworldAction::Exit,
        GridworldState::Village,
      ) => 2.0f64,
      // upper volcano exit
      (
        GridworldState::Coordinate { x: 2, y: 0 },
        GridworldAction::Exit,
        GridworldState::Volcano,
      ) => -50.0f64,
      // lower volcano exit
      (
        GridworldState::Coordinate { x: 2, y: 1 },
        GridworldAction::Exit,
        GridworldState::Volcano,
      ) => -50.0f64,
      // scenic view exit
      (
        GridworldState::Coordinate { x: 3, y: 0 },
        GridworldAction::Exit,
        GridworldState::ScenicView,
      ) => 20.0f64,
      // every other position
      _ => 0.0f64,
    }
  }

  /// Returns true if the provided state is the terminal state;
  /// false otherwise.
  ///
  /// The terminal states for this Gridworld are:
  /// - `GridworldState::ScenicView`
  /// - `GridworldState::Village`
  /// - `GridworldState::Volcano`
  fn is_state_terminal(&self, state: GridworldState) -> bool {
    match state {
      GridworldState::ScenicView => true,
      GridworldState::Village => true,
      GridworldState::Volcano => true,
      // every other position
      _ => false,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn gridworld_initial_state_test() {
    let g = Gridworld::new();

    // initial state should be {0, 1}
    assert_eq!(
      g.get_initial_state(),
      GridworldState::Coordinate { x: 0, y: 1 }
    );
  }

  #[test]
  fn gridworld_initial_state_actions_test() {
    let g = Gridworld::new();
    let g_init = g.get_initial_state();

    // initial state should have north, south, and east as possible actions
    assert_eq!(
      g.get_actions(g_init),
      vec![
        GridworldAction::North,
        GridworldAction::South,
        GridworldAction::East
      ]
    );
  }

  #[test]
  fn gridworld_possible_states_test() {
    let g = Gridworld::new();

    // the set of possible states should contain all possible (x, y) pairs
    assert_eq!(
      g.get_possible_states(),
      Some(vec![
        GridworldState::Coordinate { x: 0, y: 0 },
        GridworldState::Coordinate { x: 0, y: 1 },
        GridworldState::Coordinate { x: 0, y: 2 },
        GridworldState::Coordinate { x: 1, y: 0 },
        GridworldState::Coordinate { x: 1, y: 1 },
        GridworldState::Coordinate { x: 1, y: 2 },
        GridworldState::Coordinate { x: 2, y: 0 },
        GridworldState::Coordinate { x: 2, y: 1 },
        GridworldState::Coordinate { x: 2, y: 2 },
        GridworldState::Coordinate { x: 3, y: 0 },
        GridworldState::Coordinate { x: 3, y: 1 },
        GridworldState::Coordinate { x: 3, y: 2 },
      ])
    )
  }

  #[test]
  fn gridworld_exit_states_test() {
    let g = Gridworld::new();

    // village exit should only have exit action
    assert_eq!(
      g.get_actions(GridworldState::Coordinate { x: 0, y: 2 }),
      vec![GridworldAction::Exit]
    );

    // upper volcano exit should only have exit action
    assert_eq!(
      g.get_actions(GridworldState::Coordinate { x: 2, y: 0 }),
      vec![GridworldAction::Exit]
    );

    // lower volcano exit should only have exit action
    assert_eq!(
      g.get_actions(GridworldState::Coordinate { x: 2, y: 1 }),
      vec![GridworldAction::Exit]
    );

    // scenic view exit should only have exit action
    assert_eq!(
      g.get_actions(GridworldState::Coordinate { x: 3, y: 0 }),
      vec![GridworldAction::Exit]
    );
  }

  #[test]
  fn gridworld_get_terminal_state_actions() {
    let g = Gridworld::new();

    // no actions should be returned for terminal states
    assert_eq!(g.get_actions(GridworldState::ScenicView), vec![]);
    assert_eq!(g.get_actions(GridworldState::Village), vec![]);
    assert_eq!(g.get_actions(GridworldState::Volcano), vec![]);
  }

  #[test]
  fn gridworld_is_state_terminal_test() {
    let g = Gridworld::new();
    let g_init = g.get_initial_state();

    // initial state should not be terminal (i.e., return false)
    assert!(!g.is_state_terminal(g_init));

    // terminal states should show as terminal (i.e., return true)
    assert!(g.is_state_terminal(GridworldState::ScenicView));
    assert!(g.is_state_terminal(GridworldState::Village));
    assert!(g.is_state_terminal(GridworldState::Volcano));
  }

  #[test]
  fn gridworld_terminal_state_rewards_test() {
    let g = Gridworld::new();

    // village exit should have reward of 2
    assert_eq!(
      g.get_reward(
        GridworldState::Coordinate { x: 0, y: 2 },
        GridworldAction::Exit,
        GridworldState::Village
      ),
      2.0f64
    );

    // upper volcano exit should have reward of -50
    assert_eq!(
      g.get_reward(
        GridworldState::Coordinate { x: 2, y: 0 },
        GridworldAction::Exit,
        GridworldState::Volcano
      ),
      -50.0f64
    );

    // lower volcano exit should have reward of -50
    assert_eq!(
      g.get_reward(
        GridworldState::Coordinate { x: 2, y: 1 },
        GridworldAction::Exit,
        GridworldState::Volcano
      ),
      -50.0f64
    );

    // scenic view exit should have reward of 20
    assert_eq!(
      g.get_reward(
        GridworldState::Coordinate { x: 3, y: 0 },
        GridworldAction::Exit,
        GridworldState::ScenicView
      ),
      20.0f64
    );
  }
}

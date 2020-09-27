//! # Gridworld
//!
//! Implementation of Gridworld example Markov Decision Process.
//! This version is based on the volcano crossing example from
//! [Stanford's CS211 Lecture on Markov Decision Processes](https://youtu.be/9g32v7bK3Co?t=311).

use crate::mdp::{LearningAgent, ValueIterationAgent, MDP};
use std::collections::HashMap;

/// Enumeration of all possible actions for an agent in the Gridworld environment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridworldAction {
  /// Moves the agent one space north (i.e., `y -= 1`)
  North,
  /// Moves the agent one space south (i.e., `y += 1`)
  South,
  /// Moves the agent one space east (i.e., `x += 1`)
  East,
  /// Moves the agent one space west (i.e., `x -= 1`)
  West,
  /// Moves the agent to a terminal state (i.e., `ScenicView`, `Village`, or `Volcano`)
  Exit,
}

/// State representation for Gridworld environment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
pub struct Gridworld {
  /// Discount factor for value iteration agent.
  gamma: f64,
  /// Number of iterations for value iteration agent.
  iterations: usize,
  /// Value container for value iteration agent.
  pub values: HashMap<GridworldState, f64>,
}

impl Gridworld {
  /// Create a Gridworld object with `gamma = 1.0` (discount factor) and `iterations = 100`.
  pub fn default() -> Gridworld {
    let mut g = Gridworld {
      gamma: 1.0,
      iterations: 100,
      values: HashMap::<GridworldState, f64>::new(),
    };
    g.initialize_values();
    g
  }

  /// Create a Gridworld object with the provided `gamma` (discount factor) and `iterations`.
  pub fn new(gamma: f64, iterations: usize) -> Gridworld {
    let mut g = Gridworld {
      gamma,
      iterations,
      values: HashMap::<GridworldState, f64>::new(),
    };
    g.initialize_values();
    g
  }

  /// Set `V_0(s) = 0.0` for all s in the set of states S
  fn initialize_values(&mut self) {
    let states = self.get_possible_states().unwrap();
    // set V_0(s) = 0 for all s
    for s in states {
      self.values.insert(s, 0.0f64);
    }

    // add value of 0 for terminal states
    self.values.insert(GridworldState::ScenicView, 0.0f64);
    self.values.insert(GridworldState::Village, 0.0f64);
    self.values.insert(GridworldState::Volcano, 0.0f64);
  }
}

impl LearningAgent for Gridworld {
  fn run(&mut self) {
    let states = self.get_possible_states().unwrap();
    let num_states = states.len();

    // run value iteration
    for i in 0..self.iterations {
      // get state and policy action
      let s = states[i % num_states];
      let a = match self.calculate_policy_action(s) {
        Some(a) => a,
        None => continue,
      };

      // update value for state
      let new_val = self.calculate_q_value(s, a);
      let e = self.values.entry(s).or_insert(0.0f64);
      *e = new_val;
    }
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
  /// For this Gridworld, the set of possible (nonterminal) states is all possible `{x, y}` pairs,
  /// where `0 <= x <= 3` and `0 <= y <= 2`.
  /// The pairs are ordered first by x-value, then by y-value.
  ///
  /// Only nonterminal states are considered here as the value of the terminal states
  /// are not updated when running a learning agent.
  /// That is, the value function V(s) updates s and not s', and, since the terminal states
  /// only occur as s', we do not need to consider the terminal states in the set of states S.
  fn get_possible_states(&self) -> Option<Vec<GridworldState>> {
    let mut s = Vec::<GridworldState>::new();

    // generate Coordinate objects for all possible {x, y} pairs
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
  /// (`North`, `South`, `East`, `West`) that the agent is able to move in based on
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

  /// Returns all possible transitions that arise from performing the provided action
  /// from the provided state.
  ///
  /// The Exit action taken from the village, the volcanos, or the scenic view will result in
  /// the respective terminal state with 100% probability.
  /// A movement action taken from any state but the exit states will result in the agent
  /// moving one space in the requested direction with 100% probability.
  ///
  /// TODO: add support for randomness
  fn get_transitions(
    &self,
    state: GridworldState,
    action: GridworldAction,
  ) -> Vec<(GridworldState, f64)> {
    // define closure for transition states and probabilities
    let get_transition_state = |x: u32, y: u32| -> Vec<(GridworldState, f64)> {
      match action {
        // movement action transitions
        GridworldAction::North => vec![(GridworldState::Coordinate { x, y: y - 1 }, 1.0f64)],
        GridworldAction::South => vec![(GridworldState::Coordinate { x, y: y + 1 }, 1.0f64)],
        GridworldAction::East => vec![(GridworldState::Coordinate { x: x + 1, y }, 1.0f64)],
        GridworldAction::West => vec![(GridworldState::Coordinate { x: x - 1, y }, 1.0f64)],
        // exit action transitions
        GridworldAction::Exit if x == 0 && y == 2 => vec![(GridworldState::Village, 1.0f64)],
        GridworldAction::Exit if x == 2 && y == 0 => vec![(GridworldState::Volcano, 1.0f64)],
        GridworldAction::Exit if x == 2 && y == 1 => vec![(GridworldState::Volcano, 1.0f64)],
        GridworldAction::Exit if x == 3 && y == 0 => vec![(GridworldState::ScenicView, 1.0f64)],
        // exit action only valid at exit states
        GridworldAction::Exit => vec![],
      }
    };

    match state {
      GridworldState::Coordinate { x, y } => get_transition_state(x, y),
      // no valid transitions from terminal states
      _ => vec![],
    }
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

impl ValueIterationAgent<GridworldState, GridworldAction, f64> for Gridworld {
  fn calculate_q_value(&self, state: GridworldState, action: GridworldAction) -> f64 {
    let transitions = self.get_transitions(state, action);
    // Q(s,a) = sum_{s' in S} { P(s'|s,a) * [R(s,a,s') + (gamma * V(s'))] }
    transitions
      .iter()
      .map(|t| {
        t.1 * (self.get_reward(state, action, t.0) + (self.gamma * *self.values.get(&t.0).unwrap()))
      })
      .sum()
  }

  fn calculate_policy_action(&self, state: GridworldState) -> Option<GridworldAction> {
    let actions = self.get_actions(state);
    // get Q-values for each action
    let q: Vec<(usize, f64)> = actions
      .clone()
      .iter()
      .enumerate()
      .map(|a| (a.0, self.calculate_q_value(state, *a.1)))
      .collect();
    // find best action (max_{a in actions} Q(s,a))
    let mut opt_pair: (usize, f64) = q[0];
    for i in 1..q.len() {
      if q[i].1 > opt_pair.1 {
        opt_pair = q[i];
      }
    }
    Some(actions[opt_pair.0])
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn gridworld_default_test() {
    let g = Gridworld::default();

    assert_eq!(g.gamma, 1.0f64);
    assert_eq!(g.iterations, 100);
  }

  #[test]
  fn gridworld_new_test() {
    let g = Gridworld::new(0.5f64, 1000);

    assert_eq!(g.gamma, 0.5f64);
    assert_eq!(g.iterations, 1000);
  }

  #[test]
  fn gridworld_value_initialization_test() {
    let g = Gridworld::default();
    let mut g_states = g.get_possible_states().unwrap();

    // add states whose values don't get updated
    g_states.extend(
      vec![
        GridworldState::ScenicView,
        GridworldState::Village,
        GridworldState::Volcano,
      ]
      .iter(),
    );

    // V_0(s) = 0 for all states
    for s in g_states {
      assert_eq!(*g.values.get(&s).unwrap(), 0.0f64);
    }
  }

  #[test]
  fn gridworld_initial_state_test() {
    let g = Gridworld::default();

    // initial state should be {0, 1}
    assert_eq!(
      g.get_initial_state(),
      GridworldState::Coordinate { x: 0, y: 1 }
    );
  }

  #[test]
  fn gridworld_initial_state_actions_test() {
    let g = Gridworld::default();
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
    let g = Gridworld::default();

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
    let g = Gridworld::default();

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
    let g = Gridworld::default();

    // no actions should be returned for terminal states
    assert_eq!(g.get_actions(GridworldState::ScenicView), vec![]);
    assert_eq!(g.get_actions(GridworldState::Village), vec![]);
    assert_eq!(g.get_actions(GridworldState::Volcano), vec![]);
  }

  #[test]
  fn gridworld_is_state_terminal_test() {
    let g = Gridworld::default();
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
    let g = Gridworld::default();

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

  #[test]
  fn gridworld_initial_state_transitions() {
    let g = Gridworld::default();
    let g_init = g.get_initial_state();
    let g_init_actions = g.get_actions(g_init);
    let g_init_transitions: Vec<Vec<(GridworldState, f64)>> = g_init_actions
      .iter()
      .map(|a| g.get_transitions(g_init, *a))
      .collect();

    assert_eq!(
      g_init_transitions[0],
      vec![(GridworldState::Coordinate { x: 0, y: 0 }, 1.0f64)]
    );
    assert_eq!(
      g_init_transitions[1],
      vec![(GridworldState::Coordinate { x: 0, y: 2 }, 1.0f64)]
    );
    assert_eq!(
      g_init_transitions[2],
      vec![(GridworldState::Coordinate { x: 1, y: 1 }, 1.0f64)]
    );
  }

  #[test]
  fn gridworld_q_exit_test() {
    let g = Gridworld::default();

    // village exit
    assert_eq!(
      g.calculate_q_value(
        GridworldState::Coordinate { x: 0, y: 2 },
        GridworldAction::Exit
      ),
      2.0f64
    );

    // upper volcano exit
    assert_eq!(
      g.calculate_q_value(
        GridworldState::Coordinate { x: 2, y: 0 },
        GridworldAction::Exit
      ),
      -50.0f64
    );

    // lower volcano exit
    assert_eq!(
      g.calculate_q_value(
        GridworldState::Coordinate { x: 2, y: 1 },
        GridworldAction::Exit
      ),
      -50.0f64
    );

    // scenic view exit
    assert_eq!(
      g.calculate_q_value(
        GridworldState::Coordinate { x: 3, y: 0 },
        GridworldAction::Exit
      ),
      20.0f64
    );
  }

  #[test]
  fn gridworld_default_value_iteration_test() {
    // default world, gamma = 1.0
    let mut g = Gridworld::default();
    g.run();
    let g_states = g.get_possible_states().unwrap();

    // V*({0, 0}) = 20.0
    assert_eq!(*g.values.get(&g_states[0]).unwrap(), 20.0f64);
    // V*({0, 1}) = 20.0
    assert_eq!(*g.values.get(&g_states[1]).unwrap(), 20.0f64);
    // V*({0, 2}) = 2.0
    assert_eq!(*g.values.get(&g_states[2]).unwrap(), 2.0f64);
    // V*({1, 0}) = 20.0
    assert_eq!(*g.values.get(&g_states[3]).unwrap(), 20.0f64);
    // V*({1, 1}) = 20.0
    assert_eq!(*g.values.get(&g_states[4]).unwrap(), 20.0f64);
    // V*({1, 2}) = 20.0
    assert_eq!(*g.values.get(&g_states[5]).unwrap(), 20.0f64);
    // V*({2, 0}) = -50.0
    assert_eq!(*g.values.get(&g_states[6]).unwrap(), -50.0f64);
    // V*({2, 1}) = -50.0
    assert_eq!(*g.values.get(&g_states[7]).unwrap(), -50.0f64);
    // V*({2, 2}) = 20.0
    assert_eq!(*g.values.get(&g_states[8]).unwrap(), 20.0f64);
    // V*({3, 0}) = 20.0
    assert_eq!(*g.values.get(&g_states[9]).unwrap(), 20.0f64);
    // V*({3, 1}) = 20.0
    assert_eq!(*g.values.get(&g_states[10]).unwrap(), 20.0f64);
    // V*({3, 2}) = 20.0
    assert_eq!(*g.values.get(&g_states[11]).unwrap(), 20.0f64);
  }
}

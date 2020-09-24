//! # Markov Decision Processes
//!
//! This module contains traits helpful for implementing Markov Decision Processes.

use std::vec::Vec;

/// Generic Markov Decision Process trait.
///
/// - S: Type for representing States.
/// - A: Type for representing Actions.
/// - P: Numeric type for representing transition probabilities.
/// - R: Numeric type for representing transition rewards.
pub trait MDP<S, A, P, R> {
  /// Returns the starting state of this Markov Decision Process.
  fn get_initial_state(&self) -> S;

  /// If feasible, this function returns all possibles states that
  /// could occur in this Markov Decision Process (i.e., `Some(Vec<S>)`).
  /// If infeasible, this function returns `None`.
  fn get_possible_states(&self) -> Option<Vec<S>>;

  /// Returns all possible actions that can be taken from the provided state.
  fn get_actions(&self, state: S) -> Vec<A>;

  /// Returns all possible transitions that arise from performing the provided action
  /// from the provided state.
  fn get_transitions(&self, state: S, action: A) -> Vec<(S, P)>;

  /// Returns the reward of transitioning from the current state to the next state
  /// when the provided action is taken.
  fn get_reward(&self, state: S, action: A, next_state: S) -> R;

  /// Returns true if the provided state is the terminal state;
  /// false otherwise.
  fn is_state_terminal(&self, state: S) -> bool;
}

/// Generic Learning Agent trait.
pub trait LearningAgent {
  /// Runs this learning agent to completion.
  fn run();
}

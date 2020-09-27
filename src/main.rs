extern crate rust_gridworld;

use rust_gridworld::gridworld::Gridworld;
use rust_gridworld::mdp::{LearningAgent, MDP};

fn main() {
    let mut g = Gridworld::default();
    println!("Initial state: {:?}", g.get_initial_state());
    g.run();
    println!("Values after value-iteration: {:?}", g.values);
}

extern crate rust_gridworld;

use rust_gridworld::gridworld::Gridworld;
use rust_gridworld::mdp::MDP;

fn main() {
    let g = Gridworld::new();
    println!("{:?}", g.get_initial_state());
}

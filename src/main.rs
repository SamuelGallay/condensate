pub mod condensate;
pub mod utils;
pub mod sum;
pub mod gpu;
pub mod array;

use crate::condensate::Parameters;
use clap::Parser;
use std::process::Command;

#[derive(Parser)]
struct Cli {
    /// The space grid size
    #[arg(short, long)]
    n: Option<usize>,

    /// The number of iterations
    #[arg(short, long)]
    iter: Option<u64>,

    /// The physical size of the domain
    #[arg(short, long)]
    length: Option<f32>,

    /// Omega, the rotation
    #[arg(short, long)]
    omega: Option<f32>,

    /// Beta, the repulsion
    #[arg(short, long)]
    beta: Option<f32>,

    /// Gamma, the shape of the trap
    #[arg(short, long)]
    gamma: Option<f32>,

    /// Courant–Friedrichs–Lewy condition, less than 1.0
    #[arg(short, long)]
    cfl: Option<f32>,
}

fn main() {
    Command::new("mkdir")
        .args(["-p", "plot", "archive"])
        .spawn().unwrap();
    let cli = Cli::parse();
    //println!("{:?}", cli.n);
    let mut p = Parameters {
        n: cli.n.unwrap_or(usize::pow(2, 8)),
        length: cli.length.unwrap_or(40.0),
        omega: cli.omega.unwrap_or(1.3),
        beta: cli.beta.unwrap_or(8000.0),
        gamma: cli.gamma.unwrap_or(1.0),
        cfl: cli.cfl.unwrap_or(1.0),
        niter: cli.iter.unwrap_or(1000),
        dx: 0.0,
        dt: 0.0,
        final_time: 0.0,
    };
    p.dx = p.length / p.n as f32;
    p.dt = p.cfl * p.dx / p.omega / p.length;
    p.final_time = p.niter as f32 * p.dt;
    
    
    gpu::test();
    condensate::condensate(p);   
    //sum::benchmark();
}

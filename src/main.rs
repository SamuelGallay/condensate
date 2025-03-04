pub mod allocator;
pub mod array;
pub mod condensate;
pub mod gpu;
pub mod sum;
pub mod utils;

use crate::condensate::Parameters;
use clap::Parser;
use std::fs;
use std::string::String;

#[derive(Parser)]
struct Cli {
    /// The space grid size
    #[arg(short, long)]
    n: Option<u64>,

    /// The number of iterations
    #[arg(short, long)]
    iter: Option<u64>,

    /// The physical size of the domain
    #[arg(short, long)]
    length: Option<f64>,

    /// Omega, the rotation
    #[arg(short, long)]
    omega: Option<f64>,

    /// Beta, the repulsion
    #[arg(short, long)]
    beta: Option<f64>,

    /// Gamma, the shape of the trap
    #[arg(short, long)]
    gamma: Option<f64>,

    /// Precision
    #[arg(short, long)]
    precision: Option<f64>,

    /// Title  
    #[arg(short, long)]
    title: Option<String>,

    /// Toggle the gradient flow version
    #[arg(long)]
    gradientflow: bool,
}

fn main() {
    let _ = fs::remove_dir_all("temp");
    let _ = fs::create_dir("temp");
    let _ = fs::create_dir("archive");
    let _ = fs::create_dir("plot");
    let cli = Cli::parse();
    //println!("{:?}", cli.n);
    let mut p = Parameters {
        n: cli.n.unwrap_or(u64::pow(2, 8)),
        length: cli.length.unwrap_or(20.0),
        omega: cli.omega.unwrap_or(1.3),
        beta: cli.beta.unwrap_or(2000.0),
        gamma: cli.gamma.unwrap_or(1.0),
        niter: cli.iter.unwrap_or(1000),
        precision: cli.precision.unwrap_or(10.0),
        dx: 0.0,
    };
    p.dx = p.length / p.n as f64;
    println!("{:?}", p);

    gpu::test(p);
    if cli.gradientflow {
        condensate::old_condensate(p, cli.title);
    } else {
        condensate::condensate(p, cli.title);
    }
    //sum::benchmark();
}

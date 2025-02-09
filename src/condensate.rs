extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate ocl_vkfft_sys;
extern crate rand;

use crate::gpu::Gpu;
use crate::{sum, utils};

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use num::complex::Complex32;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft_sys::VkFFTConfiguration;
use std::time::Instant;
use utils::get_from_gpu;

const SRC: &str = include_str!("kernels.cl");

pub struct Parameters {
    pub n: usize,
    pub niter: u64,
    pub length: f32,
    pub omega: f32,
    pub beta: f32,
    pub gamma: f32,
    pub cfl: f32,
    pub dx: f32,
    pub dt: f32,
    pub final_time: f32,
}
pub fn condensate(p: Parameters) -> Result<()> {
    ocl_vkfft_sys::say_hello();

    let name = format!(
        "{}-{}-{}-{}-{}-{}",
        p.n, p.niter, p.length, p.omega, p.beta, p.gamma
    );

    println!("Final time: {}", p.final_time);
    println!("Transport CFL: {}", p.dt * p.omega * p.length / p.dx);

    let g = Gpu::new(SRC);

    let phi0 = utils::init(p.n, 1.0, p.length);
    println!("Init l2 norm: {}", utils::l2_norm(&phi0, p.dx));
    //let mut phi_back_data = Array2::<Complex32>::zeros((N, N));

    let phi_buffer = g.new_buffer(p.n);
    let newphi_buffer = g.new_buffer(p.n);
    let diff_buffer = g.new_buffer(p.n);
    let phi2hat_buffer = g.new_buffer(p.n);
    let phihat_buffer = g.new_buffer(p.n);
    let dxphi_buffer = g.new_buffer(p.n);
    let dyphi_buffer = g.new_buffer(p.n);
    phi_buffer
        .write(phi0.as_slice().ok_or(anyhow!("Oh no!"))?)
        .queue(&g.queue)
        .enq()?;
    let mut s = Complex32::new(0.0, 0.0);
    for i in 0..p.n {
        for j in 0..p.n {
            s += phi0[[i, j]];
        }
    }
    //println!("CPU sum: {:?}", s);
    utils::plot_from_gpu(&phi_buffer, "plot/in.png")?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [p.n as u64, p.n as u64, 0, 0],
        numberBatches: 1,
        device: &mut g.device.as_ptr(),
        context: &mut g.context.as_ptr(),
        bufferSize: &mut (8 * (p.n * p.n) as u64), // 8 = sizeof(Complex<f32>)
        normalize: 1,
        isInputFormatted: 1,
        ..Default::default()
    };

    let app = ocl_vkfft::App::new(config);
    let builder = ocl_vkfft::Builder::new(&app, &g.queue);

    // ------------------------------------------------------------------------- //

    g.queue.finish()?;
    println!("Initialization complete. (fake)");
    let pb = ProgressBar::new(p.niter);

    let instant = Instant::now();
    unsafe {
        for _ in 0..p.niter {
            builder.fft(&phi_buffer, &phihat_buffer, -1);

            ocl::Kernel::builder()
                .program(&g.program)
                .queue(g.queue.clone())
                .name("diffusion")
                .global_work_size([p.n, p.n])
                .disable_arg_type_check()
                .arg(&phihat_buffer)
                .arg(&dxphi_buffer)
                .arg(&dyphi_buffer)
                .arg(p.n as i32)
                .arg(&p.length)
                .arg(&p.dt)
                .build()?
                .enq()?;

            builder.fft(&dxphi_buffer, &dxphi_buffer, 1);
            builder.fft(&dyphi_buffer, &dyphi_buffer, 1);
            builder.fft(&phihat_buffer, &newphi_buffer, 1);

            ocl::Kernel::builder()
                .program(&g.program)
                .queue(g.queue.clone())
                .name("rotation")
                .global_work_size([p.n, p.n])
                .disable_arg_type_check()
                .arg(&newphi_buffer)
                .arg(&dxphi_buffer)
                .arg(&dyphi_buffer)
                .arg(&phi2hat_buffer)
                .arg(p.n as i32)
                .arg(p.length)
                .arg(p.omega)
                .arg(p.beta)
                .arg(p.dt)
                .build()?
                .enq()?;

            let sum = sum::sum(&g, &phi2hat_buffer, p.n * p.n);
            let norm = sum.re.sqrt() * p.dx;

            ocl::Kernel::builder()
                .program(&g.program)
                .queue(g.queue.clone())
                .name("rescale")
                .global_work_size([p.n, p.n])
                .disable_arg_type_check()
                .arg(&newphi_buffer)
                .arg(&phi_buffer)
                .arg(&diff_buffer)
                .arg(p.n as i32)
                .arg(p.dx)
                .arg(norm)
                .arg(&phi2hat_buffer)
                .build()?
                .enq()?;

            g.queue.finish()?;
            pb.inc(1);
        }
    }
    g.queue.finish()?;
    println!("Loop time: {:?}", instant.elapsed());

    // ------------------------------------------------------------------------- //
    utils::plot_from_gpu(&phi_buffer, "plot/out.png")?;
    utils::plot_from_gpu(&phi_buffer, format!("archive/{}.png", name).as_str())?;
    println!(
        "End l2 norm: {}",
        utils::l2_norm(&get_from_gpu(&phi_buffer)?, p.dx)
    );

    app.delete();

    println!("Exiting main function.");
    Ok(())
}

extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate ocl_vkfft_sys;
extern crate rand;

use crate::utils;

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft_sys::VkFFTConfiguration;
use std::process::Command;
use std::time::Instant;
use utils::get_from_gpu;
use utils::new_buffer;

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
    let _ = Command::new("mkdir")
        .args(["-p", "plot", "archive"])
        .spawn()?;
    let name = format!(
        "{}-{}-{}-{}-{}-{}",
        p.n, p.niter, p.length, p.omega, p.beta, p.gamma
    );

    println!("Final time: {}", p.final_time);
    println!("Transport CFL: {}", p.dt * p.omega * p.length / p.dx);

    let platform = ocl::Platform::first()?;
    let device = ocl::Device::first(&platform)?;
    let context = ocl::Context::builder().build()?;
    let program = ocl::Program::builder().src(SRC).build(&context)?;
    let queue = ocl::Queue::new(&context, device, None)?;

    let phi0 = utils::init(p.n, 1.0, p.length);
    println!("Init l2 norm: {}", utils::l2_norm(&phi0, p.dx));
    //let mut phi_back_data = Array2::<Complex32>::zeros((N, N));

    let phi_buffer = new_buffer(&queue, p.n)?;
    let newphi_buffer = new_buffer(&queue, p.n)?;
    let diff_buffer = new_buffer(&queue, p.n)?;
    let phi2hat_buffer = new_buffer(&queue, p.n)?;
    let phihat_buffer = new_buffer(&queue, p.n)?;
    let dxphi_buffer = new_buffer(&queue, p.n)?;
    let dyphi_buffer = new_buffer(&queue, p.n)?;
    let sumresult_buffer = new_buffer(&queue, p.n)?;
    phi_buffer
        .write(phi0.as_slice().ok_or(anyhow!("Oh no!"))?)
        .enq()?;
    utils::plot_from_gpu(&phi_buffer, "plot/in.png")?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [p.n as u64, p.n as u64, 0, 0],
        numberBatches: 1,
        device: &mut device.as_ptr(),
        context: &mut context.as_ptr(),
        bufferSize: &mut (8 * (p.n * p.n) as u64), // 8 = sizeof(Complex<f32>)
        normalize: 1,
        isInputFormatted: 1,
        ..Default::default()
    };

    let app = ocl_vkfft::App::new(config);
    let builder = ocl_vkfft::Builder::new(&app, &queue);

    // ------------------------------------------------------------------------- //

    queue.finish()?;
    println!("Initialization complete. (fake)");
    let pb = ProgressBar::new(p.niter);

    let instant = Instant::now();
    unsafe {
        for _ in 0..p.niter {
            builder.fft(&phi_buffer, &phihat_buffer, -1);

            ocl::Kernel::builder()
                .program(&program)
                .queue(queue.clone())
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
                .program(&program)
                .queue(queue.clone())
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

            builder.fft(&phi2hat_buffer, &phi2hat_buffer, -1);

            ocl::Kernel::builder()
                .program(&program)
                .queue(queue.clone())
                .name("rescale")
                .global_work_size([p.n, p.n])
                .disable_arg_type_check()
                .arg(&newphi_buffer)
                .arg(&phi_buffer)
                .arg(&diff_buffer)
                .arg(&phi2hat_buffer)
                .arg(p.n as i32)
                .arg(p.length)
                .build()?
                .enq()?;

            queue.finish()?;
            pb.inc(1);
        }
    }
    queue.finish()?;
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

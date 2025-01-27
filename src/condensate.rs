extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

use crate::utils;

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};
use std::process::Command;
use std::time::Instant;
use std::time::SystemTime;
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
    let phi2hat_buffer = new_buffer(&queue, p.n)?;
    let phihat_buffer = new_buffer(&queue, p.n)?;
    let dxphi_buffer = new_buffer(&queue, p.n)?;
    let dyphi_buffer = new_buffer(&queue, p.n)?;
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

    let mut app = VkFFTApplication {
        ..Default::default()
    };

    let res = unsafe { ocl_vkfft::initializeVkFFT(&mut app, config) };
    assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

    // ------------------------------------------------------------------------- //
    let mut launch_phihat = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut phi_buffer.as_ptr(),
        buffer: &mut phihat_buffer.as_ptr(),
        ..Default::default()
    };

    let kernel_diffusion = unsafe {
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
    };

    let mut launch_dxphi = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut dxphi_buffer.as_ptr(),
        buffer: &mut dxphi_buffer.as_ptr(),
        ..Default::default()
    };

    let mut launch_dyphi = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut dyphi_buffer.as_ptr(),
        buffer: &mut dyphi_buffer.as_ptr(),
        ..Default::default()
    };

    let mut launch_phi = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut phihat_buffer.as_ptr(),
        buffer: &mut phi_buffer.as_ptr(),
        ..Default::default()
    };

    let kernel_rotation = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("rotation")
            .global_work_size([p.n, p.n])
            .disable_arg_type_check()
            .arg(&phi_buffer)
            .arg(&dxphi_buffer)
            .arg(&dyphi_buffer)
            .arg(&phi2hat_buffer)
            .arg(p.n as i32)
            .arg(p.length)
            .arg(p.omega)
            .arg(p.beta)
            .arg(p.dt)
            .build()?
    };

    let mut launch_phi2hat = VkFFTLaunchParams {
        commandQueue: &mut queue.as_ptr(),
        inputBuffer: &mut phi2hat_buffer.as_ptr(),
        buffer: &mut phi2hat_buffer.as_ptr(),
        ..Default::default()
    };

    let kernel_rescale = unsafe {
        ocl::Kernel::builder()
            .program(&program)
            .queue(queue.clone())
            .name("rescale")
            .global_work_size([p.n, p.n])
            .disable_arg_type_check()
            .arg(&phi_buffer)
            .arg(&phi2hat_buffer)
            .arg(p.n as i32)
            .arg(p.length)
            .build()?
    };

    // ------------------------------------------------------------------------- //

    let _sys_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    queue.finish()?;
    println!("Initialization complete. (fake)");
    let pb = ProgressBar::new(p.niter);

    // ------------------------------------------------------------------------- //
    let instant = Instant::now();
    unsafe {
        for _ in 0..p.niter {
            let res = ocl_vkfft::VkFFTAppend(&mut app, -1, &mut launch_phihat);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            kernel_diffusion.enq()?;

            let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_dxphi);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_dyphi);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            let res = ocl_vkfft::VkFFTAppend(&mut app, 1, &mut launch_phi);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            kernel_rotation.enq()?;

            let res = ocl_vkfft::VkFFTAppend(&mut app, -1, &mut launch_phi2hat);
            assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

            kernel_rescale.enq()?;

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

    unsafe {
        ocl_vkfft::deleteVkFFT(&mut app);
    }

    println!("Exiting main function.");
    Ok(())
}

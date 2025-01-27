extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate rand;

pub mod utils;

use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
//use ndarray::Array2;
//use num::complex::Complex32;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};
//use std::f32::consts::PI;
use std::process::Command;
use std::time::Instant;
use std::time::SystemTime;
use utils::get_from_gpu;
use utils::new_buffer;
//use std::thread;
//use core::time;

const SRC: &str = include_str!("kernels.cl");
const N:usize = usize::pow(2, 10);
const L: f32 = 40.0;


fn condensate() -> Result<()> {
    let _ = Command::new("mkdir").args(["-p", "plot"]).spawn()?;
    let _gamma2: f32 = 1.0;
    let omega: f32 = 1.3;
    let beta: f32 = 8000.0;
    let cfl: f32 = 1.0;
    let dx = L / N as f32;
    let dt: f32 = cfl * dx / omega / L;
    let niter = 1000;
    let final_time = niter as f32 * dt;

    println!("Final time: {}", final_time);
    println!("Transport CFL: {}", dt * omega * L / dx);

    let platform = ocl::Platform::first()?;
    let device = ocl::Device::first(&platform)?;
    let context = ocl::Context::builder().build()?;
    let program = ocl::Program::builder().src(SRC).build(&context)?;
    let queue = ocl::Queue::new(&context, device, None)?;

    let phi0 = utils::init(N, 1.0, L);
    println!("Init l2 norm: {}", utils::l2_norm(&phi0, dx));
    //let mut phi_back_data = Array2::<Complex32>::zeros((N, N));

    let phi_buffer = new_buffer(&queue, N)?;
    let phi2hat_buffer = new_buffer(&queue, N)?;
    let phihat_buffer = new_buffer(&queue, N)?;
    let dxphi_buffer = new_buffer(&queue, N)?;
    let dyphi_buffer = new_buffer(&queue, N)?;
    phi_buffer
        .write(phi0.as_slice().ok_or(anyhow!("Oh no!"))?)
        .enq()?;
    utils::plot_from_gpu(&phi_buffer, "plot/in.png")?;

    let config = VkFFTConfiguration {
        FFTdim: 2,
        size: [N as u64, N as u64, 0, 0],
        numberBatches: 1,
        device: &mut device.as_ptr(),
        context: &mut context.as_ptr(),
        bufferSize: &mut (8 * (N * N) as u64), // 8 = sizeof(Complex<f32>)
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
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&phihat_buffer)
            .arg(&dxphi_buffer)
            .arg(&dyphi_buffer)
            .arg(N as i32)
            .arg(&L)
            .arg(&dt)
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
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&phi_buffer)
            .arg(&dxphi_buffer)
            .arg(&dyphi_buffer)
            .arg(&phi2hat_buffer)
            .arg(N as i32)
            .arg(L)
            .arg(omega)
            .arg(beta)
            .arg(dt)
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
            .global_work_size([N, N])
            .disable_arg_type_check()
            .arg(&phi_buffer)
            .arg(&phi2hat_buffer)
            .arg(N as i32)
            .arg(L)
            .build()?
    };

    // ------------------------------------------------------------------------- //

    let _sys_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    queue.finish()?;
    println!("Initialization complete. (fake)");
    let pb = ProgressBar::new(niter);

    // ------------------------------------------------------------------------- //
    let instant = Instant::now();
    unsafe {
        for _ in 0..niter {
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
    println!("End l2 norm: {}", utils::l2_norm(&get_from_gpu(&phi_buffer)?, dx));
    utils::plot_from_gpu(&phi_buffer, "plot/out.png")?;

    unsafe {
        ocl_vkfft::deleteVkFFT(&mut app);
    }

    println!("Exiting main function.");
    Ok(())
}

fn main() {
    match condensate() {
        Ok(()) => println!("Program exited successfully."),
        Err(e) => println!("Not working : {e:?}"),
    }
}

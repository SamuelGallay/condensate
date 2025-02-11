extern crate ndarray;
extern crate nix;
extern crate ocl;
extern crate ocl_vkfft;
extern crate ocl_vkfft_sys;
extern crate rand;

use crate::array::Array;
use crate::array::Stuff;
use crate::gpu::Gpu;
use crate::utils;
use crate::utils::get_from_gpu;

//use indicatif::ProgressBar;
use num::complex::Complex32;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft_sys::VkFFTConfiguration;
use std::time::Instant;

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

pub fn condensate(p: Parameters) -> () {
    ocl_vkfft_sys::say_hello();

    let name = format!(
        "{}-{}-{}-{}-{}-{}",
        p.n, p.niter, p.length, p.omega, p.beta, p.gamma
    );

    assert_eq!(usize::BITS, 64);
    println!("Final time: {}", p.final_time);
    println!("Transport CFL: {}", p.dt * p.omega * p.length / p.dx);

    let g = Gpu::new(SRC);

    let phi0 = utils::init(p.n, 1.0, p.length);
    println!("Init l2 norm: {}", utils::l2_norm(&phi0, p.dx));
    //let mut phi_back_data = Array2::<Complex32>::zeros((N, N));

    let mut phi = g.new_array(p.n);

    phi.buffer
        .write(phi0.as_slice().unwrap())
        .queue(&g.queue)
        .enq()
        .unwrap();
    let mut s = Complex32::new(0.0, 0.0);
    for i in 0..p.n {
        for j in 0..p.n {
            s += phi0[[i, j]];
        }
    }
    //println!("CPU sum: {:?}", s);
    utils::plot_from_gpu(&phi, "plot/in.png");

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

    let kernel = |name: &str| {
        let mut kernel = ocl::Kernel::builder();
        unsafe {
            kernel
                .program(&g.program)
                .queue(g.queue.clone())
                .name(name)
                .global_work_size([p.n, p.n])
                .disable_arg_type_check();
        }
        return kernel;
    };

    let scal = |a: &Array<'_, Complex32>, b: &Array<'_, Complex32>| -> f32 {
        let out = g.new_array(p.n);
        unsafe {
            kernel("scal")
                .arg(&a.buffer)
                .arg(&b.buffer)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        return out.sum().re * p.dx * p.dx;
    };

    let energy = |phi: &Array<'_, Complex32>,
                  dxphi: &Array<'_, Complex32>,
                  dyphi: &Array<'_, Complex32>|
     -> f32 {
        let out = g.new_array(p.n);
        unsafe {
            kernel("energy")
                .arg(&phi.buffer)
                .arg(&dxphi.buffer)
                .arg(&dyphi.buffer)
                .arg(&out.buffer)
                .arg(p.beta)
                .arg(p.omega)
                .arg(p.length)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        return out.sum().re * p.dx * p.dx;
    };

    let alpha = |phi: &Array<'_, Complex32>,
                 dxphi: &Array<'_, Complex32>,
                 dyphi: &Array<'_, Complex32>|
     -> f32 {
        let out = g.new_array(p.n);
        unsafe {
            kernel("alpha")
                .arg(&phi.buffer)
                .arg(&dxphi.buffer)
                .arg(&dyphi.buffer)
                .arg(&out.buffer)
                .arg(p.beta)
                .arg(p.length)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        return out.sum().re * p.dx * p.dx;
    };

    let hphif = |phi: &Array<'_, Complex32>,
                 f: &Array<'_, Complex32>,
                 dxf: &Array<'_, Complex32>,
                 dyf: &Array<'_, Complex32>,
                 lapf: &Array<'_, Complex32>|
     -> Array<'_, Complex32> {
        let out = g.new_array(p.n);
        unsafe {
            kernel("hphif")
                .arg(&phi.buffer)
                .arg(&dxf.buffer)
                .arg(&dyf.buffer)
                .arg(&lapf.buffer)
                .arg(&out.buffer)
                .arg(p.beta)
                .arg(p.omega)
                .arg(p.length)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        return out;
    };

    let hess = |phi: &Array<'_, Complex32>,
                f: &Array<'_, Complex32>,
                dxf: &Array<'_, Complex32>,
                dyf: &Array<'_, Complex32>,
                lapf: &Array<'_, Complex32>|
     -> f32 {
        return 2.0
            * (scal(f, &hphif(phi, f, dxf, dyf, lapf))
                + p.beta * scal(&(phi.clone() * phi), &(f.clone() * f)));
    };

    // ------------------------------------------------------------------------- //

    g.queue.finish().unwrap();
    println!("Initialization complete. (fake)");
    //let pb = ProgressBar::new(p.niter);

    let instant = Instant::now();
    let mut k = 0;
    let energy_decrease = -200f32;
    let precision = 10f32.powf(-7.0);
    //let theta0 = 0.1;
    //let mut theta = theta0;
    //let mut divisions = 0;

    unsafe {
        while k < p.niter && energy_decrease < -precision {
            k += 1;

            let phihat = g.new_array(p.n);
            builder.fft(&phi.buffer, &phihat.buffer, -1);

            kernel("diffusion")
                .arg(&phihat.buffer)
                .arg(p.n)
                .arg(p.length)
                .arg(p.dt)
                .build()
                .unwrap()
                .enq()
                .unwrap();

            let lapphi = g.new_array(p.n);
            let dxphi = g.new_array(p.n);
            let dyphi = g.new_array(p.n);
            kernel("differentiate")
                .arg(&phihat.buffer)
                .arg(&dxphi.buffer)
                .arg(&dyphi.buffer)
                .arg(&lapphi.buffer)
                .arg(p.n)
                .arg(p.length)
                .arg(p.dt)
                .build()
                .unwrap()
                .enq()
                .unwrap();
            builder.fft(&dxphi.buffer, &dxphi.buffer, 1);
            builder.fft(&dyphi.buffer, &dyphi.buffer, 1);
            println!("Energy : {}", energy(&phi, &dxphi, &dyphi));

            let newphi = g.new_array(p.n);
            builder.fft(&phihat.buffer, &newphi.buffer, 1);

            //let phi2hat = g.new_array(p.n);
            kernel("rotation")
                .arg(&newphi.buffer)
                .arg(&dxphi.buffer)
                .arg(&dyphi.buffer)
                .arg(p.n)
                .arg(p.length)
                .arg(p.omega)
                .arg(p.beta)
                .arg(p.dt)
                .build()
                .unwrap()
                .enq()
                .unwrap();

            let phi_squared = newphi.clone().norm2();
            let norm = phi_squared.sum().re.sqrt() * p.dx;
            phi = newphi * Complex32::new(1.0 / norm, 0.0);

            g.queue.finish().unwrap();
            //pb.inc(1);
        }
    }
    g.queue.finish().unwrap();
    println!("Loop time: {:?}", instant.elapsed());

    // ------------------------------------------------------------------------- //
    utils::plot_from_gpu(&phi, "plot/out.png");
    utils::plot_from_gpu(&phi, format!("archive/{}.png", name).as_str());
    println!(
        "End l2 norm: {}",
        utils::l2_norm(&get_from_gpu(&phi.buffer), p.dx)
    );

    app.delete();

    println!("Exiting main function.");
}

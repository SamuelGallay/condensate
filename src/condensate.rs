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
use num::complex::Complex64;
use ocl::ocl_core::ClDeviceIdPtr;
use ocl_vkfft_sys::VkFFTConfiguration;
use std::time::Instant;

const SRC: &str = include_str!("kernels.cl");

#[derive(Debug)]
pub struct Parameters {
    pub n: usize,
    pub niter: u64,
    pub length: f64,
    pub omega: f64,
    pub beta: f64,
    pub gamma: f64,
    pub dx: f64,
}

pub fn condensate(p: Parameters) -> () {
    ocl_vkfft_sys::say_hello();
    let name = format!(
        "{}-{}-{}-{}-{}-{}",
        p.n, p.niter, p.length, p.omega, p.beta, p.gamma
    );
    assert_eq!(usize::BITS, 64);

    let g = Gpu::new(SRC);

    let phi0 = utils::init(p.n, 1.0, p.length);
    println!("Init l2 norm: {}", utils::l2_norm(&phi0, p.dx));
    //let mut phi_back_data = Array2::<Complex64>::zeros((N, N));

    let mut phi = g.new_array(p.n);
    phi.buffer
        .write(phi0.as_slice().unwrap())
        .queue(&g.queue)
        .enq()
        .unwrap();

    let vector_field = g.new_array(p.n);
    vector_field
        .buffer
        .write(utils::vector_field(p.n, p.length).as_slice().unwrap())
        .queue(&g.queue)
        .enq()
        .unwrap();

    let mut s = Complex64::new(0.0, 0.0);
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
        bufferSize: &mut (16 * (p.n * p.n) as u64), // 16 = sizeof(Complex<f64>)
        normalize: 1,
        isInputFormatted: 1,
        doublePrecision: 1,
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

    let scal = |a: &Array<'_, Complex64>, b: &Array<'_, Complex64>| -> f64 {
        let out = g.new_array(p.n);
        unsafe {
            kernel("scal")
                .arg(&a.buffer)
                .arg(&b.buffer)
                .arg(&out.buffer)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        return out.sum().re * p.dx * p.dx;
    };

    let energy = |phi: &Array<'_, Complex64>,
                  dxphi: &Array<'_, Complex64>,
                  dyphi: &Array<'_, Complex64>|
     -> f64 {
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

    let alpha = |phi: &Array<'_, Complex64>,
                 dxphi: &Array<'_, Complex64>,
                 dyphi: &Array<'_, Complex64>|
     -> f64 {
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

    let hphif = |phi: &Array<'_, Complex64>,
                 f: &Array<'_, Complex64>,
                 dxf: &Array<'_, Complex64>,
                 dyf: &Array<'_, Complex64>,
                 lapf: &Array<'_, Complex64>|
     -> Array<'_, Complex64> {
        let out = g.new_array(p.n);
        unsafe {
            kernel("hphif")
                .arg(&phi.buffer)
                .arg(&f.buffer)
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

    let differentiate = |f: &Array<'_, Complex64>| -> (
        Array<'_, Complex64>,
        Array<'_, Complex64>,
        Array<'_, Complex64>,
    ) {
        let fhat = g.new_array(p.n);
        builder.fft(&f.buffer, &fhat.buffer, -1);

        let dxf = g.new_array(p.n);
        let dyf = g.new_array(p.n);
        let lapf = g.new_array(p.n);
        unsafe {
            kernel("differentiate")
                .arg(&fhat.buffer)
                .arg(&dxf.buffer)
                .arg(&dyf.buffer)
                .arg(&lapf.buffer)
                .arg(p.length)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
        builder.fft(&dxf.buffer, &dxf.buffer, 1);
        builder.fft(&dyf.buffer, &dyf.buffer, 1);
        builder.fft(&lapf.buffer, &lapf.buffer, 1);

        return (dxf, dyf, lapf);
    };

    let invamlap2 = |fhat: &Array<'_, Complex64>, a: f64| -> () {
        unsafe {
            kernel("invamlap2")
                .arg(&fhat.buffer)
                .arg(a)
                .arg(p.length)
                .arg(p.n)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        g.queue.finish().unwrap();
    };

    let hess = |phi: &Array<'_, Complex64>,
                f: &Array<'_, Complex64>,
                dxf: &Array<'_, Complex64>,
                dyf: &Array<'_, Complex64>,
                lapf: &Array<'_, Complex64>|
     -> f64 {
        return 2.0
            * (scal(f, &hphif(phi, f, dxf, dyf, lapf))
                + p.beta * scal(&(phi.clone() * phi), &(f.clone() * f)));
    };
    let norm2 = |phi: Array<'_, Complex64>| -> f64 {
        return f64::sqrt(phi.abs_squared().sum().re * p.dx * p.dx);
    };

    // ------------------------------------------------------------------------- //

    g.queue.finish().unwrap();
    println!("Initialization complete. (fake)");
    //let pb = ProgressBar::new(p.niter);

    let instant = Instant::now();
    let mut k = 0;
    let mut steps_below_precision = 0;
    let precision = 10f64.powf(-12.0);
    let theta0 = 0.1;
    let mut theta = theta0;
    let mut divisions = 0;
    let mut rnm1 = phi.clone();
    let mut p_times_rnm1 = phi.clone();
    let mut pnm1 = phi.clone();

    while k < p.niter &&  steps_below_precision < 10{
        k += 1;

        let (dxphi, dyphi, lapphi) = differentiate(&phi);
        let hphi = hphif(&phi, &phi, &dxphi, &dyphi, &lapphi);
        let e = energy(&phi, &dxphi, &dyphi);
        let a = alpha(&phi, &dxphi, &dyphi);
        let lambdan = scal(&hphi, &phi);
        let r = phi.clone() * (-lambdan) + &hphi;

        let prec = (phi.clone().abs_squared() * p.beta + &vector_field + a).inv_sqrt();
        
        let temphat = g.new_array(p.n);
        builder.fft(&(r.clone() * &prec).buffer, &temphat.buffer, -1);
        invamlap2(&temphat, a);
        let p_times_rn = g.new_array(p.n);
        builder.fft(&temphat.buffer, &p_times_rn.buffer, 1);
        let p_times_rn = p_times_rn * &prec;
        
        let mut mybeta = f64::max(
            0.0,
            scal(&(r.clone() + &(rnm1.clone() * -1.0)), &p_times_rn) / scal(&rnm1, &p_times_rnm1),
        );
        if k == 1 {
            mybeta = 0f64;
        }
        
        let descent_direction = p_times_rn.clone() * -1.0  + &(pnm1 * mybeta);
        let temp_scal = -scal(&descent_direction, &phi);

        // p := proj_descent_direction
        let p = descent_direction + &(phi.clone() * temp_scal);

        // Def previous variables
        rnm1 = r;
        p_times_rnm1 = p_times_rn;
        pnm1 = p.clone();

        let (dxp, dyp, lapp) = differentiate(&p);
        let temp_hess = hess(&phi, &p, &dxp, &dyp, &lapp);
        let normp = norm2(p.clone());
        let p_normed = p.clone() * (1.0 / normp);
        let denominator = temp_hess - lambdan * normp * normp;
        if denominator > 0.0 {
            theta = -2.0 * scal(&hphi, &p) * normp / denominator;
        } else {
            theta *= 1.5;
        }

        let mut newphi = phi.clone() * theta.cos() + &(p_normed.clone() * theta.sin());
        let (dxnewphi, dynewphi, _) = differentiate(&newphi);
        let mut energy_delta = energy(&newphi, &dxnewphi, &dynewphi) - e;

        while energy_delta > 0.0 {
            divisions += 1;
            theta /= 2.0;
            newphi = phi.clone() * theta.cos() + &(p_normed.clone() * theta.sin());
            let (dxnewphi, dynewphi, _) = differentiate(&newphi);
            energy_delta = energy(&newphi, &dxnewphi, &dynewphi) - e;
        }

        phi = newphi;
    if energy_delta > -precision {
        steps_below_precision += 1;}
    else {steps_below_precision = 0;}

        if k % 500 == 0 {
            utils::plot_from_gpu(&phi, format!("temp/{}.png", k).as_str());
            utils::plot_from_gpu(&phi, "plot/latest.png");
            println!("");
            println!("Iteration : {}", k);
            println!("Energy : {}", e);
            println!("Delta: {:?}", energy_delta);
            println!("Norm2 : {}", norm2(phi.clone()));
            println!("Divisions : {}", divisions);
            divisions = 0;
        }
        //pb.inc(1);
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

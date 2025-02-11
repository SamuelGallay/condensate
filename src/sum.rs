use ndarray::Array2;
use num::complex::Complex64;
use num::Zero;
use ocl::ocl_core::ClDeviceIdPtr;
use rand::Rng;
use std::time::Instant;

use crate::gpu;

const SRC: &str = include_str!("kernels.cl");

pub unsafe fn inplace(g: &gpu::Gpu, buffer: &ocl::Buffer<Complex64>, length: u64) -> Complex64 {
    assert!(length.is_power_of_two());
    let n = u64::ilog2(length);
    assert_eq!(2u64.pow(n), length);
    //println!("Length {}", length);
    for k in (0..n).rev() {
        ocl::Kernel::builder()
            .program(&g.program)
            .queue(g.queue.clone())
            .name("sum_inplace")
            .global_work_size([2u64.pow(k)])
            .disable_arg_type_check()
            .arg(buffer)
            .arg(2u64.pow(k))
            .build()
            .unwrap()
            .enq()
            .unwrap();
        g.queue.finish().unwrap();
    }
    let mut temp = [Complex64::zero()];
    buffer
        .read(temp.as_mut_slice())
        .queue(&g.queue)
        .enq()
        .unwrap();
    g.queue.finish().unwrap();
    return temp[0];
}

pub unsafe fn sum(g: &gpu::Gpu, buffer: &ocl::Buffer<Complex64>, length: usize) -> Complex64 {
    assert!(length.is_power_of_two());
    //let n = u64::ilog2(length);
    //assert_eq!(2u64.pow(n), length);
    //println!("Length {}", length);
    const DIVISOR:usize = 16;
    let mut current_length = length;
    while current_length % DIVISOR == 0 {
        current_length /= DIVISOR;
        ocl::Kernel::builder()
            .program(&g.program)
            .queue(g.queue.clone())
            .name("simd")
            .global_work_size([current_length])
            .disable_arg_type_check()
            .arg(buffer)
            .arg(current_length)
            .build()
            .unwrap()
            .enq()
            .unwrap();
        g.queue.finish().unwrap();
    }
    //assert_eq!(current_length, 1);
    let mut temp = [Complex64::zero(); DIVISOR]; // SIMD * divisor
    buffer
        .read(temp.as_mut_slice())
        .queue(&g.queue)
        .len(current_length)
        .enq()
        .unwrap();
    g.queue.finish().unwrap();
    let mut s = Complex64::zero();
    for i in 0..(current_length) {
        s += temp[i];
    }
    return s;
}

pub fn fourier_sum(
    g: &gpu::Gpu,
    buffer: &ocl::Buffer<Complex64>,
    fftapp: &ocl_vkfft::App,
) -> Complex64 {
    let builder = ocl_vkfft::Builder::new(&fftapp, &g.queue);
    builder.fft(&buffer, &buffer, -1);
    let mut temp = [Complex64::zero()];
    buffer
        .read(temp.as_mut_slice())
        .queue(&g.queue)
        .enq()
        .unwrap();
    g.queue.finish().unwrap();
    return temp[0];
}

pub fn benchmark() {
    let mut rng = rand::thread_rng();
    let n = 2usize.pow(12);
    let g = gpu::Gpu::new(SRC);
    let config = ocl_vkfft_sys::VkFFTConfiguration {
        FFTdim: 2,
        size: [n as u64, n as u64, 0, 0],
        numberBatches: 1,
        device: &mut g.device.as_ptr(),
        context: &mut g.context.as_ptr(),
        bufferSize: &mut (8 * (n * n) as u64), // 8 = sizeof(Complex<f64>)
        normalize: 1,
        isInputFormatted: 1,
        ..Default::default()
    };
    let fftapp = ocl_vkfft::App::new(config);

    let mut cpu_data = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cpu_data[[i, j]] = Complex64::new(rng.gen::<f64>(), rng.gen::<f64>());
        }
    }

    let data_buffer = g.new_array(n);
    let work_buffer = g.new_array(n);

    data_buffer.buffer
        .write(cpu_data.as_slice().unwrap())
        .enq()
        .unwrap();
    g.queue.finish().unwrap();

    let niter = 100;
    let mut res = Complex64::zero();

    let now = Instant::now();
    for _ in 0..niter {
        res = Complex64::zero();
        for e in cpu_data.iter() {
            res += e;
        }
    }
    println!("cpu : {}, time: {:?}", res, now.elapsed());

    let now = Instant::now();
    for _ in 0..niter {
        data_buffer.buffer.copy(&work_buffer.buffer, None, None).enq().unwrap();
        g.queue.finish().unwrap();
        res = fourier_sum(&g, &work_buffer.buffer, &fftapp);
    }
    println!("fft : {}, time: {:?}", res, now.elapsed());

    let now = Instant::now();
    for _ in 0..niter {
        data_buffer.buffer.copy(&work_buffer.buffer, None, None).enq().unwrap();
        g.queue.finish().unwrap();
        res = unsafe { sum(&g, &work_buffer.buffer, n * n) };
    }
    println!("simd: {}, time: {:?}", res, now.elapsed());

    let now = Instant::now();
    for _ in 0..niter {
        data_buffer.buffer.copy(&work_buffer.buffer, None, None).enq().unwrap();
        g.queue.finish().unwrap();
        res = unsafe { inplace(&g, &work_buffer.buffer, (n * n) as u64) };
    }
    println!("csum: {}, time: {:?}", res, now.elapsed());
}

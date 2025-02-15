extern crate noise;

use crate::array;
use colorgrad::Gradient;
use core::f64;
use image::ImageBuffer;
use ndarray::Array2;
use noise::{Fbm, NoiseFn, Perlin};
//use num::complex::{Complex, ComplexFloat};
use crate::array::Cplx;
use num::complex::Complex64;
use num::complex::ComplexFloat;
use num::integer::Roots;
//use ocl::Buffer;
use rand::Rng;
use std::f64::consts::PI;

pub fn noise2d(n: usize) -> Array2<Cplx> {
    let s = 2.0 * f64::consts::PI / (n as f64);
    let r = 10.0;
    let mut a = Array2::<Cplx>::zeros((n, n));
    let perlin: Fbm<Perlin> = Fbm::new(12);
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = Cplx::new(
                perlin.get([
                    r * (i as f64 * s).cos(),
                    r * (i as f64 * s).sin(),
                    r * (j as f64 * s).cos(),
                    r * (j as f64 * s).sin(),
                ]),
                0f64,
            );
        }
    }
    a
}

pub fn dist(a: Vec<f64>, b: Vec<f64>, dx: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0f64;
    for i in 0..a.len() {
        s += (a[i] - b[i]) * (a[i] - b[i]) * dx;
    }
    s.sqrt()
}

pub fn fftfreq(n: usize, l: f64) -> Vec<f64> {
    let mut freq = vec![0f64; n];
    let s = 2.0 * PI / l;
    for i in 0..n / 2 {
        freq[i] = i as f64 * s;
        freq[n / 2 + i] = (i as f64 - (n as f64) / 2.0) * s;
    }
    freq
}

pub fn get_from_gpu(a: &array::Array) -> Array2<Complex64> {
    let n = a.buffer.len().sqrt();
    let mut cpu_data = Array2::<Cplx>::zeros((n, n));
    a.buffer.read(cpu_data.as_slice_mut().unwrap()).enq().unwrap();
    a.gpu.queue.finish().unwrap();
    unsafe { std::mem::transmute::<Array2<Cplx>, Array2<Complex64>>(cpu_data) }
    
}

pub fn max(arr: &Array2<f64>) -> f64 {
    arr.into_iter().cloned().reduce(f64::max).unwrap()
}

pub fn printmax(a: &array::Array, name: &str) {
    let m = max(&get_from_gpu(a).mapv(Complex64::norm));
    println!("Max {} : {}", name, m);
}

pub fn image_from_array(cpu_data: &Array2<f64>) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let n = cpu_data.len_of(ndarray::Axis(0));
    let m = max(&cpu_data.mapv(f64::abs));
    let normalized = cpu_data.mapv(|x| x / m);
    let grad = colorgrad::GradientBuilder::new()
        //.html_colors(&["black", "maroon", "red", "orange", "yellow"])
        //.html_colors(&["black", "red", "orange", "yellow"])
        //.html_colors(&["black", "red", "yellow"])
        //.html_colors(&["#000004", "#170b3a", "#420a68", "#6b176e", "#932667", "#bb3654", "#dd513a", "#f3771a", "#fca50a", "#f6d644", "#fcffa4"])
 .html_colors(&["#000000", "#1e0c03", "#a91b0a", "#fc0201", "#ff1000", "#ff7800", "#ffee00", "#ffff00"])
        .build::<colorgrad::CatmullRomGradient>()
        .unwrap();
    //let grad = colorgrad::preset::inferno();
    image::ImageBuffer::from_fn(n as u32, n as u32, |i, j| {
        let v = grad
            .at(normalized[[i as usize, j as usize]] as f32)
            .to_rgba8();
        image::Rgb(v[..3].try_into().unwrap())
    })
}

pub fn plot_array(cpu_data: &Array2<f64>, name: &str) {
    let imgbuf = image_from_array(cpu_data);
    imgbuf.save(name).unwrap();
}

pub fn plot_from_gpu(a: &array::Array, name: &str) {
    let cpu_data = get_from_gpu(a).mapv(|x| x.abs()*x.abs());
    plot_array(&cpu_data, name);
}

pub fn l2_norm(a: &Array2<Complex64>, dx: f64) -> f64 {
    let mut s = 0f64;
    for e in a.iter() {
        s += e.norm_sqr() * dx * dx;
    }
    s.sqrt()
}

pub fn init(n: u64, eps0: f64, l: f64) -> Array2<Cplx> {
    let mut rng = rand::thread_rng();
    let mut a = Array2::<Complex64>::zeros((n as usize, n as usize));
    let dx = l / (n as f64);
    for i in 0..(n as usize) {
        for j in 0..(n as usize) {
            let x = (i as f64) * dx - l / 2.0;
            let y = (j as f64) * dx - l / 2.0;
            a[[i, j]].re = 1f64 / f64::sqrt(PI * eps0) * f64::exp(-(x * x + y * y) / (2.0 * eps0))
                + 1f64 / f64::sqrt(PI * eps0) * x / l * f64::exp(-(x * x + y * y) / (2.0 * eps0));
            a[[i, j]].im =
                1f64 / f64::sqrt(PI * eps0) * y / l * f64::exp(-(x * x + y * y) / (2.0 * eps0));
            a[[i, j]].re *= 1.0 + 0.1 * rng.gen::<f64>();
            a[[i, j]].im *= 1.0 + 0.1 * rng.gen::<f64>();
        }
    }
    let norm = l2_norm(&a, dx);
    let normalized = a / norm;
    unsafe { std::mem::transmute::<Array2<Complex64>, Array2<Cplx>>(normalized) }
}

pub fn vector_field(n: u64, l: f64) -> Array2<Cplx> {
    let mut a = Array2::<Complex64>::zeros((n as usize, n as usize));
    let dx = l / (n as f64);
    for i in 0..(n as usize) {
        for j in 0..(n as usize) {
            let x = (i as f64) * dx - l / 2.0;
            let y = (j as f64) * dx - l / 2.0;
            a[[i, j]].re = x * x + y * y;
            a[[i, j]].im = 0f64;
        }
    }
    unsafe { std::mem::transmute::<Array2<Complex64>, Array2<Cplx>>(a) }
}

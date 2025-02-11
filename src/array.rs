use crate::gpu::Gpu;
use crate::sum;
use num::complex::Complex32;
use ocl::Buffer;
use std::ops::{Mul, Add};

pub struct Array<'a, T: ocl::OclPrm> {
    pub buffer: Buffer<T>,
    pub gpu: &'a Gpu,
    pub size: usize,
}

pub trait Stuff {
    fn sum(self) -> Complex32;
    fn conj(self) -> Self;
    fn abs_squared(self) -> Self;
    // Only acting on the real part and makes the imaginary part zero... Yes it is hacky.
    fn inv_sqrt(self) -> Self;
}

impl<'a> Clone for Array<'a, Complex32> {
    fn clone(&self) -> Self {
        let out = self.gpu.new_array(self.size);
        self.buffer.copy(&out.buffer, None, None).enq().unwrap();
        self.gpu.queue.finish().unwrap();
        return out;
    }
}

impl<'a> Stuff for Array<'a, Complex32> {
    fn sum(self) -> Complex32 {
        return unsafe { sum::sum(self.gpu, &self.buffer, self.size * self.size) };
    }

    fn conj(self) -> Self {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("conj_vect")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
    }

    fn abs_squared(self) -> Self {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("abs_squared_vect")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
    }
    fn inv_sqrt(self) -> Self {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("inv_sqrt_vect")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
    }

}

impl<'a> Mul<Complex32> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn mul(self, scalar: Complex32) -> Self::Output {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("mul_vect_scalar")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(scalar)
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
    }
}

impl<'a> Add<f32> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn add(self, scalar: f32) -> Self::Output {
        return self + Complex32::new(scalar, 0.0);
    }
}

impl<'a> Add<Complex32> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn add(self, scalar: Complex32) -> Self::Output {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("add_vect_scalar")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(scalar)
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
    }
}

impl<'a> Mul<f32> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn mul(self, scalar: f32) -> Self::Output {
        return self * Complex32::new(scalar, 0.0);
    }
}

impl<'a> Mul<&Self> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn mul(self, other: &Self) -> Self::Output {
        //let out = self.gpu.new_array(self.size);
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("mul_vect_vect")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(&other.buffer)
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
        //return out;
    }
}

impl<'a> Add<&Self> for Array<'a, Complex32> {
    type Output = Array<'a, Complex32>;
    fn add(self, other: &Self) -> Self::Output {
        //let out = self.gpu.new_array(self.size);
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("add_vect_vect")
                .global_work_size([self.size, self.size])
                .disable_arg_type_check()
                .arg(&self.buffer)
                .arg(&other.buffer)
                .arg(&self.buffer)
                .arg(self.size)
                .build()
                .unwrap()
                .enq()
                .unwrap();
        }
        self.gpu.queue.finish().unwrap();
        return self;
        //return out;
    }
}

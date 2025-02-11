use crate::gpu::Gpu;
use crate::sum;
use num::complex::Complex32;
use ocl::Buffer;
use std::ops::Mul;

pub struct Array<'a, T: ocl::OclPrm> {
    pub buffer: Buffer<T>,
    pub gpu: &'a Gpu,
    pub size: usize,
}

pub trait Stuff {
    fn sum(self) -> Complex32;
    fn conj(self) -> Self;
    fn norm2(self) -> Self;
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

    fn norm2(self) -> Self {
        unsafe {
            ocl::Kernel::builder()
                .program(&self.gpu.program)
                .queue(self.gpu.queue.clone())
                .name("norm2_vect")
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

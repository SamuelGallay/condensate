use crate::gpu::Gpu;
use crate::sum;
use num::complex::Complex64;
//use num::complex::Cplx;
use ocl::Buffer;
use std::ops::{Add, Mul};
//use std::rc::Rc;
use crate::allocator::Allocator;
//use std::cell::RefCell;

pub type Cplx = ocl::prm::Double2;

pub struct Array<'a> {
    pub buffer: Buffer<Cplx>,
    pub gpu: &'a Gpu,
    pub size: u64,
    pub allocator: &'a Allocator<'a>,
    pub index_in_allocator: usize,
}

pub trait Stuff {
    fn sum(self) -> Complex64;
    fn conj(self) -> Self;
    fn abs_squared(self) -> Self;
    // Only acting on the real part and makes the imaginary part zero... Yes it is hacky.
    fn inv_sqrt(self) -> Self;
}

impl<'a> Clone for Array<'a> {
    fn clone(&self) -> Self {
        let out = self.allocator.new_array();
        self.buffer.copy(&out.buffer, None, None).enq().unwrap();
        self.gpu.queue.finish().unwrap();
        out
    }
}

impl<'a> Drop for Array<'a> {
    fn drop(&mut self) {
        self.allocator.set_free(self.index_in_allocator);
        //println!("Dropping an array. Free: {}", self.allocator.get_free_len());
    }
}

impl<'a> Stuff for Array<'a> {
    fn sum(self) -> Complex64 {
        unsafe { sum::sum(self.gpu, &self.buffer, self.size * self.size) }
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
        self
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
        self
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
        self
    }
}

impl<'a> Mul<Cplx> for Array<'a> {
    type Output = Array<'a>;
    fn mul(self, scalar: Cplx) -> Self::Output {
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
        self
    }
}

impl<'a> Add<f64> for Array<'a> {
    type Output = Array<'a>;
    fn add(self, scalar: f64) -> Self::Output {
        self + Cplx::new(scalar, 0.0)
    }
}

impl<'a> Add<Cplx> for Array<'a> {
    type Output = Array<'a>;
    fn add(self, scalar: Cplx) -> Self::Output {
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
        self
    }
}

impl<'a> Mul<f64> for Array<'a> {
    type Output = Array<'a>;
    fn mul(self, scalar: f64) -> Self::Output {
        self * Cplx::new(scalar, 0.0)
    }
}

impl<'a> Mul<&Self> for Array<'a> {
    type Output = Array<'a>;
    fn mul(self, other: &Self) -> Self::Output {
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
        self
    }
}

impl<'a> Add<&Self> for Array<'a> {
    type Output = Array<'a>;
    fn add(self, other: &Self) -> Self::Output {
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
        self
    }
}

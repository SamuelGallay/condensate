extern crate num;
extern crate ocl;
use core::ffi::c_void;

use num::complex::Complex64;
use ocl::{Buffer, Queue};
use ocl_vkfft_sys::{
    VkFFTApplication, VkFFTConfiguration, VkFFTLaunchParams, VkFFTResult_VKFFT_SUCCESS,
};

pub struct App {
    // This pointer must never be allowed to leave the struct (LOL)
    app: VkFFTApplication,
}
impl App {
    pub fn new(config: VkFFTConfiguration) -> Self {
        let mut vkapp = VkFFTApplication {
            ..Default::default()
        };
        let res = unsafe { ocl_vkfft_sys::initializeVkFFT(&mut vkapp, config) };
        assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);

        App { app: vkapp }
    }

    pub fn delete(&self) -> () {
        unsafe {
            ocl_vkfft_sys::deleteVkFFT(
                &self.app as *const VkFFTApplication as *mut VkFFTApplication,
            );
        }
    }
}

pub struct Params {
    queue: *mut c_void,
    inbuf: *mut c_void,
    outbuf: *mut c_void,
}
impl Params {
    pub fn new(queue: &Queue, inbuffer: &Buffer<Complex64>, outbuffer: &Buffer<Complex64>) -> Self {
        Params {
            queue: queue.as_ptr(),
            inbuf: inbuffer.as_ptr(),
            outbuf: outbuffer.as_ptr(),
        }
    }

    pub fn get(&self) -> VkFFTLaunchParams {
        return VkFFTLaunchParams {
            commandQueue: &self.queue as *const *mut c_void as *mut *mut c_void,
            inputBuffer: &self.inbuf as *const *mut c_void as *mut *mut c_void,
            buffer: &self.outbuf as *const *mut c_void as *mut *mut c_void,
            ..Default::default()
        };
    }
}

pub fn append(app: &App, dir: i32, params: &Params) -> () {
    unsafe {
        let res = ocl_vkfft_sys::VkFFTAppend(
            &app.app as *const VkFFTApplication as *mut VkFFTApplication,
            dir,
            &mut params.get(),
        );
        assert_eq!(res, VkFFTResult_VKFFT_SUCCESS);
    }
}

pub struct Builder<'a> {
    queue: &'a Queue,
    app: &'a App,
}

impl<'a> Builder<'a> {
    pub fn new(app: &'a App, queue: &'a Queue) -> Self {
        Self { app, queue }
    }
    pub fn fft(&self, inbuffer: &Buffer<Complex64>, outbuffer: &Buffer<Complex64>, dir: i32) -> () {
        let p = Params::new(self.queue, inbuffer, outbuffer);
        append(self.app, dir, &p);
        self.queue.finish().unwrap();
    }
}

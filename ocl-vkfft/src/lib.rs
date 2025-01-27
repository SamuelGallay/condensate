#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)] 

extern crate cl_sys;
use cl_sys::{cl_platform_id, cl_command_queue, cl_mem, cl_device_id, cl_context, cl_program, cl_kernel};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe extern "C" {
    pub fn almost_initializeVkFFT(
        app: *mut VkFFTApplication,
        inputLaunchConfiguration: VkFFTConfiguration,
    ) -> VkFFTResult;
}

pub fn say_hello() {
    println!("{}", concat!(env!("OUT_DIR"), "/bindings.rs"));
}


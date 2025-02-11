use crate::array;
use num::complex::Complex32;
use ocl::Buffer;

const SRC: &str = include_str!("kernels.cl");

pub struct Gpu {
    pub platform: ocl::Platform,
    pub device: ocl::Device,
    pub context: ocl::Context,
    pub program: ocl::Program,
    pub queue: ocl::Queue,
}

impl Gpu {
    pub fn new(prog: &str) -> Self {
        let platform = ocl::Platform::first().unwrap();
        let device = ocl::Device::first(&platform).unwrap();
        let context = ocl::Context::builder().build().unwrap();
        let temp = ocl::Program::builder().src(prog).build(&context);
        let program = match temp {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{}", e);
                panic!("Panicking while building program.")
            }
        };
        let queue = ocl::Queue::new(&context, device, None).unwrap();

        Self {
            platform,
            device,
            context,
            program,
            queue,
        }
    }

    pub fn delete(&self) -> () {}

    pub fn new_array(&self, n: usize) -> array::Array<Complex32> {
        let buffer = Buffer::<Complex32>::builder()
            .queue(self.queue.clone())
            .len(n * n)
            .build()
            .unwrap();
        return array::Array {
            buffer,
            gpu: &self,
            size: n,
        };
    }
}

pub fn test() {
    let gpu = Gpu::new(&SRC);
    let n = 100;
    println!("Platform: {}", gpu.platform.version().unwrap());
    println!("Device: {}", gpu.device.name().unwrap());
    println!(
        "Context: {}",
        gpu.context
            .info(ocl::enums::ContextInfo::Properties)
            .unwrap()
    );
    //let s = device.info(ocl::enums::DeviceInfo::Extensions)?;
    //println!("Infos : {:?}", s);

    let array = gpu.new_array(n);
    unsafe {
        let kernel = ocl::Kernel::builder()
            .program(&gpu.program)
            .queue(gpu.queue.clone())
            .name("conj_vect")
            .global_work_size([n, n])
            .disable_arg_type_check()
            .arg(&array.buffer)
            .arg(&array.buffer)
            .arg(n)
            .build()
            .unwrap();
        println!("Default LWS: {:?}", kernel.default_local_work_size());
        println!("Default GWS: {:?}", kernel.default_global_work_size());
        kernel.enq().unwrap();
    }
}

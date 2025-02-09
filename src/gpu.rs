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
        let program = ocl::Program::builder().src(prog).build(&context).unwrap();
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

    pub fn new_buffer(&self, n: usize) ->Buffer<Complex32> {
    let buffer = Buffer::<Complex32>::builder()
        .queue(self.queue.clone())
        .len(n * n)
        .build().unwrap();
    return buffer;
}
}

pub fn test() {
    let g = Gpu::new(&SRC);
    let n = 100;
    println!("Platform: {}", g.platform.version().unwrap());
    println!("Device: {}", g.device.name().unwrap());
    println!(
        "Context: {}",
        g.context.info(ocl::enums::ContextInfo::Properties).unwrap()
    );
    //let s = device.info(ocl::enums::DeviceInfo::Extensions)?;
    //println!("Infos : {:?}", s);


    let _buffer = Buffer::<Complex32>::builder()
        .queue(g.queue.clone())
        .len(n * n)
        .build().unwrap();
}

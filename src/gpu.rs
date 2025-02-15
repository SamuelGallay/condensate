use crate::condensate::Parameters;

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
        let device = ocl::Device::first(platform).unwrap();
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

    pub fn delete(&self) {}

    /*
    pub fn new_array(&self, n: usize) -> array::Array<Cplx> {
        let buffer = Buffer::<Cplx>::builder()
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
    */
}

pub fn test(p: Parameters) {
    let gpu = Gpu::new(SRC);
    let n = 100;
    println!("Platform: {}", gpu.platform.version().unwrap());
    println!("Device: {}", gpu.device.name().unwrap());
    println!(
        "Context: {}",
        gpu.context
            .info(ocl::enums::ContextInfo::Properties)
            .unwrap()
    );
    let alloc = crate::allocator::Allocator::new(&gpu, n);
    //let s = device.info(ocl::enums::DeviceInfo::Extensions)?;
    //println!("Infos : {:?}", s);

    let array = alloc.new_array();
    unsafe {
        let kernel = ocl::Kernel::builder()
            .program(&gpu.program)
            .queue(gpu.queue.clone())
            .name("conj_vect")
            .global_work_size([n, n])
            .arg(&array.buffer)
            .arg(&array.buffer)
            .arg(n)
            .build()
            .unwrap();
        println!("Default LWS: {:?}", kernel.default_local_work_size());
        println!("Default GWS: {:?}", kernel.default_global_work_size());
        kernel.enq().unwrap();
    }
    unsafe {
        let kernel = ocl::Kernel::builder()
            .program(&gpu.program)
            .queue(gpu.queue.clone())
            .name("read_params")
            .global_work_size([2, 2])
            .arg(p)
            .build()
            .unwrap();
        kernel.enq().unwrap();
        gpu.queue.finish().unwrap();
    }
}

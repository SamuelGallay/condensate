use crate::array::{self, Cplx};
use crate::gpu::Gpu;
use std::cell::RefCell;

pub struct Allocator<'a> {
    gpu: &'a Gpu,
    n: u64,
    memory: RefCell<Vec<ocl::Buffer<Cplx>>>,
    free: RefCell<Vec<usize>>,
}

impl<'a> Allocator<'a> {
    pub fn new(gpu: &'a Gpu, n: u64) -> Self {
        let mut allo = Self {
            gpu,
            n,
            memory: RefCell::new(Vec::new()),
            free: RefCell::new(Vec::new()),
        };

        for _ in 0..25 {
            allo.allocate_new();
        }
        allo
    }

    pub fn get_free_len(&self) -> usize {
        return self.free.borrow().len();
    }

    fn allocate_new(&mut self) {
        let i = self.memory.borrow().len();
        self.free.borrow_mut().push(i);
        let buffer = ocl::Buffer::<Cplx>::builder()
            .queue(self.gpu.queue.clone())
            .len((self.n * self.n) as usize)
            .build()
            .unwrap();
        self.memory.borrow_mut().push(buffer);
    }

    pub fn new_array(&self) -> array::Array {
        let index_in_allocator = match self.free.borrow_mut().pop() {
            Some(i) => i,
            None => panic!("No free buffer available"),
        };

        return array::Array {
            buffer: self.memory.borrow()[index_in_allocator].clone(),
            gpu: self.gpu,
            size: self.n,
            allocator: self,
            index_in_allocator,
        };
    }

    pub fn dumb_new_array(&self) -> array::Array {
        let index_in_allocator = match self.free.borrow_mut().pop() {
            Some(i) => i,
            None => panic!("No free buffer available"),
        };
        
        let buffer = ocl::Buffer::<Cplx>::builder()
            .queue(self.gpu.queue.clone())
            .len((self.n * self.n) as usize)
            .build()
            .unwrap();


        array::Array {
            buffer,
            gpu: self.gpu,
            size: self.n,
            allocator: self,
            index_in_allocator,
        }
    }

    pub fn set_free(&self, i: usize) {
        self.free.borrow_mut().push(i);
    }
}

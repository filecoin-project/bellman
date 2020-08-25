use super::*;
use crate::multicore::Worker;
use futures::Future;
use log::info;
use paired::bls12_381::Bls12;
use rust_gpu_tools::opencl as cl;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct DevicePool {
    devices: Vec<cl::Device>,
}

impl Default for DevicePool {
    fn default() -> Self {
        Self {
            devices: cl::Device::all().unwrap(),
        }
    }
}

impl DevicePool {
    pub fn new(devices: Vec<cl::Device>) -> Self {
        Self { devices }
    }
}

lazy_static::lazy_static! {
    static ref LOCK:Mutex<()>=Mutex::new(());
    static ref PROGRAMS: HashMap<cl::Device, cl::Program> = {
        let mut ret = HashMap::new();
        for d in cl::Device::all().unwrap() {
            info!("Compiling kernels on device: {} (Bus-id: {})",d.name(),d.bus_id());
            let src = sources::kernel::<Bls12>(d.brand() == cl::Brand::Nvidia);
            let program = cl::Program::from_opencl(d.clone(), &src).unwrap();
            ret.insert(d, program);
        }
        ret
    };
}

pub fn programs() -> &'static HashMap<cl::Device, cl::Program> {
    &*PROGRAMS
}

pub fn schedule<F, T>(
    worker: &Worker,
    pool: &DevicePool,
    f: F,
) -> Box<dyn Future<Item = T, Error = ()>>
where
    F: FnOnce(&cl::Program) -> T + Send + 'static,
    T: Send + 'static,
{
    let device = pool.devices[0].clone();
    Box::new(worker.compute(move || {
        let _lock = LOCK.lock().unwrap();
        let ret = f(&PROGRAMS[&device]);
        Ok(ret)
    }))
}

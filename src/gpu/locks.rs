use fs2::FileExt;
use log::{debug, info, warn};
use std::fs::File;
use std::path::PathBuf;
use std::thread;

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";
fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

fn acquire_file_lock(f: &File) {
    while f.try_lock_exclusive().is_err() {
        std::thread::yield_now();
    }
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn lock() -> GPULock {
        let gpu_lock_file = tmp_path(GPU_LOCK_NAME);
        debug!(
            "[{:?}] Acquiring GPU lock at {:?} ...",
            thread::current().id(),
            &gpu_lock_file
        );
        let f = File::create(&gpu_lock_file)
            .unwrap_or_else(|_| panic!("Cannot create GPU lock file at {:?}", &gpu_lock_file));
        acquire_file_lock(&f);
        debug!("[{:?}] GPU lock acquired!", thread::current().id());
        GPULock(f)
    }

    pub fn wait() {
        let f = File::create(tmp_path(GPU_LOCK_NAME)).unwrap();
        acquire_file_lock(&f);
    }
}

impl Drop for GPULock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("[{:?}] GPU lock released!", thread::current().id());
    }
}

/// `PriorityLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME);
        debug!(
            "[{:?}] Acquiring priority lock at {:?} ...",
            thread::current().id(),
            &priority_lock_file
        );
        let f = File::create(&priority_lock_file).unwrap_or_else(|_| {
            panic!(
                "[{:?}] Cannot create priority lock file at {:?}",
                &priority_lock_file,
                thread::current().id()
            )
        });
        acquire_file_lock(&f);
        debug!("[{:?}] Priority lock acquired!", thread::current().id());

        // Once the priority lock is acquired, wait until the GPULock
        // is released before proceeding.  This is a priority
        // inversion since we are potentially waiting for a lower
        // priority thread with the GPU lock to release it.  If we
        // don't do this, a deadlock is possible.
        GPULock::wait();

        PriorityLock(f)
    }

    pub fn wait(priority: bool) {
        debug!(
            "[{:?}] Priority lock wait called (priority: {})!",
            thread::current().id(),
            priority
        );
        if !priority {
            let f = File::create(tmp_path(PRIORITY_LOCK_NAME)).unwrap();
            acquire_file_lock(&f);

            // Once the priority lock is acquired, wait until the GPULock
            // is released before proceeding.  This is a priority
            // inversion since we are potentially waiting for a lower
            // priority thread with the GPU lock to release it.  If we
            // don't do this, a deadlock is possible.
            GPULock::wait();
        }
        debug!(
            "[{:?}] Priority lock wait returning true (priority: {})!",
            thread::current().id(),
            priority
        );
    }

    pub fn should_break(priority: bool) -> bool {
        !priority
            && File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .try_lock_exclusive()
                .is_err()
    }

    pub fn unlock(&mut self) {
        self.0.unlock().unwrap();
        debug!("[{:?}] Priority lock unlocked!", thread::current().id());
    }
}

impl Drop for PriorityLock {
    fn drop(&mut self) {
        debug!("[{:?}] Priority lock released!", thread::current().id());
    }
}

use super::error::{GPUError, GPUResult};
use super::fft::FFTKernel;
use super::multiexp::MultiexpKernel;
use crate::bls::Engine;
use crate::domain::create_fft_kernel;
use crate::multiexp::create_multiexp_kernel;

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        #[allow(clippy::upper_case_acronyms)]
        pub struct $class<E>
        where
            E: Engine,
        {
            log_d: usize,
            priority: bool,
            kernel: Option<$kern<E>>,
        }

        impl<E> $class<E>
        where
            E: Engine,
        {
            pub fn new(log_d: usize, priority: bool) -> $class<E> {
                $class::<E> {
                    log_d,
                    priority,
                    kernel: None,
                }
            }

            fn init(&mut self) -> bool {
                if self.kernel.is_none() {
                    info!("GPU is available for {}!", $name);
                    self.kernel = $func::<E>(self.log_d, self.priority);
                    return true;
                }

                true
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern<E>) -> GPUResult<R>,
            {
                if std::env::var("BELLMAN_NO_GPU").is_ok() {
                    return Err(GPUError::GPUDisabled);
                }

                if !self.init() {
                    warn!("GPU {} init failed! GPUTaken", $name);
                    return Err(GPUError::GPUTaken);
                }

                loop {
                    if let Some(ref mut k) = self.kernel {
                        match f(k) {
                            Err(GPUError::GPUTaken) => {
                                self.free();
                                warn!("GPU {} failed! GPUTaken", $name);
                                return Err(GPUError::GPUTaken);
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GPUError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(LockedFFTKernel, FFTKernel, create_fft_kernel, "FFT");
locked_kernel!(
    LockedMultiexpKernel,
    MultiexpKernel,
    create_multiexp_kernel,
    "Multiexp"
);

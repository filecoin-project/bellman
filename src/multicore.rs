//! An interface for dealing with the kinds of parallel computations involved in
//! `bellperson`. It's currently just a thin wrapper around [`CpuPool`] and
//! [`rayon`] but may be extended in the future to allow for various
//! parallelism strategies.
//!
//! [`CpuPool`]: futures_cpupool::CpuPool

use lazy_static::lazy_static;
use std::env;

lazy_static! {
    static ref NUM_CPUS: usize = if let Ok(num) = env::var("BELLMAN_NUM_CPUS") {
        if let Ok(num) = num.parse() {
            num
        } else {
            num_cpus::get()
        }
    } else {
        num_cpus::get()
    };
    pub static ref THREAD_POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
        .num_threads(*NUM_CPUS)
        .build()
        .unwrap();
    pub static ref VERIFIER_POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
        .num_threads(NUM_CPUS.max(6))
        .build()
        .unwrap();
}

#[derive(Clone)]
pub struct Worker {}

impl Worker {
    pub fn new() -> Worker {
        Worker {}
    }

    pub fn log_num_cpus(&self) -> u32 {
        log2_floor(*NUM_CPUS)
    }

    pub fn scope<'a, F, R>(&self, elements: usize, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'a>, usize) -> R + Send,
        R: Send,
    {
        let chunk_size = if elements < *NUM_CPUS {
            1
        } else {
            elements / *NUM_CPUS
        };

        THREAD_POOL.scope(|scope| f(scope, chunk_size))
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_log2_floor() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(5), 2);
        assert_eq!(log2_floor(6), 2);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
    }
}

use super::error::{GPUError, GPUResult};
use super::scheduler;
use super::utils;
use crate::multicore::Worker;
use ff::{PrimeField, ScalarEngine};
use futures::Future;
use groupy::{CurveAffine, CurveProjective};
use log::*;
use paired::Engine;
use rust_gpu_tools::*;
use std::any::TypeId;
use std::sync::Arc;

const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                Ok(0f64)
            }
        })
        .unwrap_or(0f64)
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct MultiexpKernel<E>
where
    E: Engine,
{
    _phantom: std::marker::PhantomData<E::Fr>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
    2 * core_count / num_windows
}

fn calc_window_size(n: usize, exp_bits: usize, core_count: usize) -> usize {
    // window_size = ln(n / num_groups)
    // num_windows = exp_bits / window_size
    // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
    // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
    // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
    //
    // Thus we need to solve the following equation:
    // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
    let lower_bound = (((exp_bits * n) as f64) / ((2 * core_count) as f64)).ln();
    for w in 0..MAX_WINDOW_SIZE {
        if (w as f64) + (w as f64).ln() > lower_bound {
            return w;
        }
    }

    MAX_WINDOW_SIZE
}

fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize) -> usize {
    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * 2 * core_count / exp_bits
    (((max_window_size as f64).exp() as f64)
        * (max_window_size as f64)
        * 2f64
        * (core_count as f64)
        / (exp_bits as f64))
        .ceil() as usize
}

fn calc_chunk_size<E>(mem: u64, core_count: usize) -> usize
where
    E: Engine,
{
    let aff_size = std::mem::size_of::<E::G1Affine>() + std::mem::size_of::<E::G2Affine>();
    let exp_size = std::mem::size_of::<E::Fr>();
    let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();
    ((((mem as f64) * (1f64 - MEMORY_PADDING)) as usize)
        - (2 * core_count * ((1 << MAX_WINDOW_SIZE) + 1) * proj_size))
        / (aff_size + exp_size)
}

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    fn ensure_curve() -> GPUResult<()> {
        if TypeId::of::<E>() == TypeId::of::<paired::bls12_381::Bls12>() {
            Ok(())
        } else {
            Err(GPUError::CurveNotSupported)
        }
    }

    fn chunk_size_of(program: &opencl::Program) -> usize {
        let core_count = utils::get_core_count(&program.device());
        let exp_bits = std::mem::size_of::<E::Fr>() * 8;
        let max_n = calc_chunk_size::<E>(program.device().memory(), core_count);
        let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        std::cmp::min(max_n, best_n)
    }

    fn multiexp_on<G>(
        program: &opencl::Program,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        MultiexpKernel::<E>::ensure_curve()?;

        info!(
            "Running Multiexp of {} elements on {}...",
            n,
            program.device().name()
        );

        let core_count = utils::get_core_count(&program.device());
        let exp_bits = std::mem::size_of::<E::Fr>() * 8;

        let window_size = calc_window_size(n as usize, exp_bits, core_count);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(core_count, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let mut base_buffer = program.create_buffer::<G>(n)?;
        base_buffer.write_from(0, bases)?;
        let mut exp_buffer =
            program.create_buffer::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>(n)?;
        exp_buffer.write_from(0, exps)?;

        let bucket_buffer =
            program.create_buffer::<<G as CurveAffine>::Projective>(2 * core_count * bucket_len)?;
        let result_buffer =
            program.create_buffer::<<G as CurveAffine>::Projective>(2 * core_count)?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut global_work_size = num_windows * num_groups;
        global_work_size +=
            (LOCAL_WORK_SIZE - (global_work_size % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let kernel = program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            },
            global_work_size,
            None,
        );

        call_kernel!(
            kernel,
            &base_buffer,
            &bucket_buffer,
            &result_buffer,
            &exp_buffer,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        )?;

        let mut results = vec![<G as CurveAffine>::Projective::zero(); num_groups * num_windows];
        result_buffer.read_into(0, &mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }

    pub fn multiexp<G>(
        devices: &scheduler::DevicePool,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
        <G as groupy::CurveAffine>::Engine: paired::Engine,
    {
        MultiexpKernel::<E>::ensure_curve()?;

        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + n)];
        let exps = &exps[..n];

        let chunk_size = scheduler::programs()
            .values()
            .map(|p| MultiexpKernel::<E>::chunk_size_of(p))
            .min()
            .unwrap();

        let result = bases
            .chunks(chunk_size)
            .zip(exps.chunks(chunk_size))
            .map(|(bases, exps)| {
                let bases = bases.to_vec();
                let exps = exps.to_vec();
                scheduler::schedule(
                    pool,
                    devices,
                    move |prog| -> GPUResult<<G as CurveAffine>::Projective> {
                        MultiexpKernel::<E>::multiexp_on(prog, &bases, &exps, bases.len())
                    },
                )
            })
            .map(|future| future.wait().unwrap())
            .collect::<GPUResult<Vec<<G as CurveAffine>::Projective>>>()?
            .iter()
            .fold(<G as CurveAffine>::Projective::zero(), |mut a, b| {
                a.add_assign(b);
                a
            });

        Ok(result)
    }
}

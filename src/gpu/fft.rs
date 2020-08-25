use crate::gpu::{
    error::{GPUError, GPUResult},
    scheduler,
};
use crate::multicore::Worker;
use ff::Field;
use futures::future::Future;
use log::*;
use paired::Engine;
use rust_gpu_tools::*;
use std::any::TypeId;
use std::cmp;

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

pub struct FFTKernel<E>
where
    E: Engine,
{
    _phantom: std::marker::PhantomData<E::Fr>,
}

impl<E> FFTKernel<E>
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

    /// Peforms a FFT round
    /// * `log_n` - Specifies log2 of number of elements
    /// * `log_p` - Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    /// * `deg` - 1=>radix2, 2=>radix4, 3=>radix8, ...
    /// * `max_deg` - The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    fn radix_fft_round(
        program: &opencl::Program,
        src_buffer: &opencl::Buffer<E::Fr>,
        dst_buffer: &opencl::Buffer<E::Fr>,
        pq_buffer: &opencl::Buffer<E::Fr>,
        omegas_buffer: &opencl::Buffer<E::Fr>,
        log_n: u32,
        log_p: u32,
        deg: u32,
        max_deg: u32,
    ) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = (n >> deg) * local_work_size;
        let kernel = program.create_kernel(
            "radix_fft",
            global_work_size as usize,
            Some(local_work_size as usize),
        );
        call_kernel!(
            kernel,
            src_buffer,
            dst_buffer,
            pq_buffer,
            omegas_buffer,
            opencl::LocalBuffer::<E::Fr>::new(1 << deg),
            n,
            log_p,
            deg,
            max_deg
        )?;
        Ok(())
    }

    /// Share some precalculated values between threads to boost the performance
    fn setup_pq_omegas(
        program: &opencl::Program,
        omega: &E::Fr,
        n: usize,
        max_deg: u32,
    ) -> GPUResult<(opencl::Buffer<E::Fr>, opencl::Buffer<E::Fr>)> {
        FFTKernel::<E>::ensure_curve()?;

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] = E::Fr::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }
        let mut pq_buffer = program.create_buffer::<E::Fr>(1 << MAX_LOG2_RADIX >> 1)?;
        pq_buffer.write_from(0, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![E::Fr::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer::<E::Fr>(LOG2_MAX_ELEMENTS)?;
        omegas_buffer.write_from(0, &omegas)?;

        Ok((pq_buffer, omegas_buffer))
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(
        devices: &scheduler::DevicePool,
        a: &mut [E::Fr],
        omega: &E::Fr,
        log_n: u32,
    ) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let mut elems = a.to_vec();
        let omega = *omega;
        let worker = Worker::new();
        let result =
            scheduler::schedule(&worker, devices, move |program| -> GPUResult<Vec<E::Fr>> {
                let n = 1 << log_n;
                info!(
                    "Running FFT of {} elements on {}...",
                    n,
                    program.device().name()
                );
                let mut src_buffer = program.create_buffer::<E::Fr>(n)?;
                let mut dst_buffer = program.create_buffer::<E::Fr>(n)?;

                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
                let (pq_buffer, omegas_buffer) =
                    FFTKernel::<E>::setup_pq_omegas(program, &omega, n, max_deg)?;

                src_buffer.write_from(0, &elems)?;
                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    FFTKernel::<E>::radix_fft_round(
                        program,
                        &src_buffer,
                        &dst_buffer,
                        &pq_buffer,
                        &omegas_buffer,
                        log_n,
                        log_p,
                        deg,
                        max_deg,
                    )?;
                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                src_buffer.read_into(0, &mut elems)?;

                Ok(elems)
            })
            .wait()
            .unwrap()?;
        a.copy_from_slice(&result);
        Ok(())
    }
}

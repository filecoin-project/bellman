use super::error::{GPUError, GPUResult};
use crate::multicore::Worker;
use ff::{PrimeField, ScalarEngine};
use groupy::CurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

// This module is compiled instead of `fft.rs` and `multiexp.rs` if `gpu` feature is disabled.

pub struct FFTKernel<E>(PhantomData<E>)
where
    E: ScalarEngine;

impl<E> FFTKernel<E>
where
    E: ScalarEngine,
{
    pub fn radix_fft(_: &DevicePool, _: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }
}

pub struct MultiexpKernel<E>(PhantomData<E>)
where
    E: ScalarEngine;

impl<E> MultiexpKernel<E>
where
    E: ScalarEngine,
{
    pub fn multiexp<G>(
        _: &DevicePool,
        _: &Worker,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        return Err(GPUError::GPUDisabled);
    }
}

pub struct DevicePool;

impl Default for DevicePool {
    fn default() -> Self {
        Self
    }
}

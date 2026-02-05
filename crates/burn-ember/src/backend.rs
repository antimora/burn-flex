use alloc::string::String;
use core::sync::atomic::{AtomicU64, Ordering};

use burn_backend::{Backend, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};
use burn_std::device::Device;
use rand::{SeedableRng, rngs::StdRng};

use crate::qtensor::EmberQTensor;
use crate::tensor::EmberTensor;

/// Global seed for random number generation.
/// Uses AtomicU64 for thread-safe seed storage.
static SEED: AtomicU64 = AtomicU64::new(0);

/// Flag indicating if a seed has been explicitly set.
static SEED_SET: AtomicU64 = AtomicU64::new(0);

/// Get a random number generator, either seeded or with OS entropy.
pub(crate) fn get_rng() -> StdRng {
    if SEED_SET.load(Ordering::SeqCst) != 0 {
        // Use the stored seed, then increment it for next call
        let seed = SEED.fetch_add(1, Ordering::SeqCst);
        StdRng::seed_from_u64(seed)
    } else {
        // Use OS entropy
        StdRng::from_os_rng()
    }
}

/// CPU device for the Ember backend.
///
/// Unit struct since there's only one CPU device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmberDevice;

impl Device for EmberDevice {
    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, 0)
    }

    fn from_id(_id: DeviceId) -> Self {
        Self
    }

    fn device_count(_kind: u16) -> usize {
        1
    }
}

impl DeviceOps for EmberDevice {}

impl core::fmt::Display for EmberDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ember")
    }
}

/// The Ember backend - a fast, portable CPU backend for Burn.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ember;

impl Backend for Ember {
    type Device = EmberDevice;

    type FloatTensorPrimitive = EmberTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = EmberTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = EmberTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = EmberQTensor;

    fn name(_device: &Self::Device) -> String {
        "ember".into()
    }

    fn seed(_device: &Self::Device, seed: u64) {
        SEED.store(seed, Ordering::SeqCst);
        SEED_SET.store(1, Ordering::SeqCst);
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            // Full support for standard types
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::Bool => DTypeUsage::Storage | DTypeUsage::Arithmetic,
            // Quantized types: storage only for now
            DType::QFloat(_) => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
}

// Ops traits are implemented in the ops module

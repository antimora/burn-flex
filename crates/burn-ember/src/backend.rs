use alloc::string::String;

use burn_backend::{Backend, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};
use burn_std::device::Device;
use burn_std::rand::{SeedableRng, StdRng};
use burn_std::stub::Mutex;

use crate::qtensor::EmberQTensor;
use crate::tensor::EmberTensor;

/// Type alias for the RNG used by Ember.
pub type EmberRng = StdRng;

/// Global seed storage for reproducible random number generation.
/// Uses Mutex for thread-safe RNG state management.
pub(crate) static SEED: Mutex<Option<EmberRng>> = Mutex::new(None);

/// Get a random number generator.
/// If a seed was set, clones and returns the seeded RNG.
/// Otherwise, creates a new RNG with OS entropy (std) or constant seed (no_std).
pub(crate) fn get_seeded_rng() -> EmberRng {
    burn_std::rand::get_seeded_rng()
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
        let rng = EmberRng::seed_from_u64(seed);
        let mut seed_lock = SEED.lock().unwrap();
        *seed_lock = Some(rng);
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

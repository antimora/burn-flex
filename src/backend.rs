use alloc::string::String;

use burn_backend::{Backend, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};
use burn_std::device::Device;

use crate::qtensor::EmberQTensor;
use crate::tensor::EmberTensor;

/// CPU device for the Ember backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmberDevice {
    index: usize,
}

impl EmberDevice {
    /// Create a new CPU device.
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Get the device index.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl Device for EmberDevice {
    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, self.index as u32)
    }

    fn from_id(id: DeviceId) -> Self {
        Self {
            index: id.index_id as usize,
        }
    }

    fn device_count(_kind: u16) -> usize {
        1 // Single CPU device
    }
}

impl DeviceOps for EmberDevice {}

impl core::fmt::Display for EmberDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ember({})", self.index)
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

    fn seed(_device: &Self::Device, _seed: u64) {
        // TODO: Implement seeding for random number generation
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

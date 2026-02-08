use alloc::vec::Vec;

use burn_backend::{DType, QTensorPrimitive, TensorMetadata, quantization::QuantStore};
use burn_std::{QuantScheme, Shape};

use crate::tensor::FlexTensor;

/// Quantized tensor for the Flex backend.
///
/// Stores quantized i8 values in the tensor and keeps scales separately
/// for efficient dequantization without reparsing bytes.
#[derive(Clone, Debug)]
pub struct FlexQTensor {
    /// The underlying quantized data (stored as i8).
    pub tensor: FlexTensor,
    /// Quantization scheme.
    pub scheme: QuantScheme,
    /// Per-tensor or per-block scale factors.
    pub scales: Vec<f32>,
}

impl QTensorPrimitive for FlexQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        QuantScheme::default().with_store(QuantStore::Native)
    }
}

impl TensorMetadata for FlexQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    fn rank(&self) -> usize {
        self.tensor.rank()
    }
}

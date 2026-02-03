use burn_backend::{DType, QTensorPrimitive, TensorMetadata};
use burn_std::{QuantScheme, Shape};

use crate::tensor::EmberTensor;

/// Quantized tensor for the Ember backend.
#[derive(Clone, Debug)]
pub struct EmberQTensor {
    /// The underlying quantized data.
    pub tensor: EmberTensor,
    /// Quantization scheme.
    pub scheme: QuantScheme,
}

impl QTensorPrimitive for EmberQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
}

impl TensorMetadata for EmberQTensor {
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

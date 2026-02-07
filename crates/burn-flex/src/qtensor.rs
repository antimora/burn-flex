use burn_backend::{DType, QTensorPrimitive, TensorMetadata};
use burn_std::{QuantScheme, Shape};

use crate::tensor::FlexTensor;

/// Quantized tensor for the Flex backend.
#[derive(Clone, Debug)]
pub struct FlexQTensor {
    /// The underlying quantized data.
    pub tensor: FlexTensor,
    /// Quantization scheme.
    pub scheme: QuantScheme,
}

impl QTensorPrimitive for FlexQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
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

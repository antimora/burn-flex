//! Quantized tensor operations for the Flex backend.

use burn_backend::{
    DType, ExecutionError, TensorData,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{Device, FloatTensor, QuantizedTensor},
};
use burn_std::Shape;

use crate::{Flex, FlexQTensor, FlexTensor};

impl QTensorOps<Flex> for Flex {
    fn q_from_data(data: TensorData, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected quantized dtype, got {:?}", data.dtype),
        };
        FlexQTensor {
            tensor: FlexTensor::from_data(data),
            scheme,
        }
    }

    fn quantize(
        _tensor: FloatTensor<Flex>,
        _scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Flex>,
    ) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor quantize not yet supported")
    }

    fn dequantize(_tensor: QuantizedTensor<Flex>) -> FloatTensor<Flex> {
        unimplemented!("Flex: quantized tensor dequantize not yet supported")
    }

    fn q_device(_tensor: &QuantizedTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn q_to_device(tensor: QuantizedTensor<Flex>, _device: &Device<Flex>) -> QuantizedTensor<Flex> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Flex>, shape: Shape) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: tensor.tensor.reshape(shape),
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Flex>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.tensor.into_data())
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Flex>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Flex> {
        FlexQTensor {
            tensor: tensor.tensor.transpose(dim1, dim2),
            scheme: tensor.scheme,
        }
    }

    fn q_permute(_tensor: QuantizedTensor<Flex>, _axes: &[usize]) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor permute not yet supported")
    }

    fn q_flip(_tensor: QuantizedTensor<Flex>, _axes: &[usize]) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor flip not yet supported")
    }

    fn q_expand(_tensor: QuantizedTensor<Flex>, _shape: Shape) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor expand not yet supported")
    }

    fn q_select(
        _tensor: QuantizedTensor<Flex>,
        _dim: usize,
        _indices: burn_backend::tensor::IntTensor<Flex>,
    ) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor select not yet supported")
    }

    fn q_slice(
        _tensor: QuantizedTensor<Flex>,
        _slices: &[burn_std::Slice],
    ) -> QuantizedTensor<Flex> {
        unimplemented!("Flex: quantized tensor slice not yet supported")
    }
}

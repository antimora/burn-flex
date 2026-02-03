//! Quantized tensor operations for the Ember backend.

use core::future::Future;

use burn_backend::{
    DType, ExecutionError, TensorData,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{Device, FloatTensor, QuantizedTensor},
};
use burn_std::Shape;

use crate::{Ember, EmberQTensor, EmberTensor};

impl QTensorOps<Ember> for Ember {
    fn q_from_data(data: TensorData, _device: &Device<Ember>) -> QuantizedTensor<Ember> {
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            _ => panic!("Expected quantized dtype, got {:?}", data.dtype),
        };
        EmberQTensor {
            tensor: EmberTensor::from_data(data),
            scheme,
        }
    }

    fn quantize(
        _tensor: FloatTensor<Ember>,
        _scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Ember>,
    ) -> QuantizedTensor<Ember> {
        todo!("quantize")
    }

    fn dequantize(_tensor: QuantizedTensor<Ember>) -> FloatTensor<Ember> {
        todo!("dequantize")
    }

    fn q_device(_tensor: &QuantizedTensor<Ember>) -> Device<Ember> {
        Default::default()
    }

    fn q_to_device(
        tensor: QuantizedTensor<Ember>,
        _device: &Device<Ember>,
    ) -> QuantizedTensor<Ember> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Ember>, shape: Shape) -> QuantizedTensor<Ember> {
        EmberQTensor {
            tensor: tensor.tensor.reshape(shape),
            scheme: tensor.scheme,
        }
    }

    fn q_into_data(
        tensor: QuantizedTensor<Ember>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { Ok(tensor.tensor.into_data()) }
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Ember>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Ember> {
        EmberQTensor {
            tensor: tensor.tensor.transpose(dim1, dim2),
            scheme: tensor.scheme,
        }
    }

    fn q_permute(_tensor: QuantizedTensor<Ember>, _axes: &[usize]) -> QuantizedTensor<Ember> {
        todo!("q_permute")
    }

    fn q_flip(_tensor: QuantizedTensor<Ember>, _axes: &[usize]) -> QuantizedTensor<Ember> {
        todo!("q_flip")
    }

    fn q_expand(_tensor: QuantizedTensor<Ember>, _shape: Shape) -> QuantizedTensor<Ember> {
        todo!("q_expand")
    }

    fn q_select(
        _tensor: QuantizedTensor<Ember>,
        _dim: usize,
        _indices: burn_backend::tensor::IntTensor<Ember>,
    ) -> QuantizedTensor<Ember> {
        todo!("q_select")
    }

    fn q_slice(
        _tensor: QuantizedTensor<Ember>,
        _slices: &[burn_std::Slice],
    ) -> QuantizedTensor<Ember> {
        todo!("q_slice")
    }
}

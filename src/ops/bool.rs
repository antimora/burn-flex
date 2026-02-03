//! Bool tensor operations for the Ember backend.

use burn_backend::{
    DType, ExecutionError, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::{Ember, EmberTensor};

impl BoolTensorOps<Ember> for Ember {
    fn bool_from_data(data: TensorData, _device: &Device<Ember>) -> BoolTensor<Ember> {
        EmberTensor::from_data(data)
    }

    async fn bool_into_data(tensor: BoolTensor<Ember>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn bool_device(_tensor: &BoolTensor<Ember>) -> Device<Ember> {
        Default::default()
    }

    fn bool_to_device(tensor: BoolTensor<Ember>, _device: &Device<Ember>) -> BoolTensor<Ember> {
        tensor
    }

    fn bool_reshape(tensor: BoolTensor<Ember>, shape: Shape) -> BoolTensor<Ember> {
        tensor.reshape(shape)
    }

    fn bool_slice(_tensor: BoolTensor<Ember>, _slices: &[Slice]) -> BoolTensor<Ember> {
        todo!("bool_slice")
    }

    fn bool_empty(shape: Shape, _device: &Device<Ember>) -> BoolTensor<Ember> {
        EmberTensor::empty(shape, DType::Bool)
    }

    fn bool_slice_assign(
        _tensor: BoolTensor<Ember>,
        _slices: &[Slice],
        _value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_slice_assign")
    }

    fn bool_into_int(_tensor: BoolTensor<Ember>) -> IntTensor<Ember> {
        todo!("bool_into_int")
    }

    fn bool_into_float(_tensor: BoolTensor<Ember>) -> FloatTensor<Ember> {
        todo!("bool_into_float")
    }

    fn bool_swap_dims(tensor: BoolTensor<Ember>, dim1: usize, dim2: usize) -> BoolTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn bool_permute(_tensor: BoolTensor<Ember>, _axes: &[usize]) -> BoolTensor<Ember> {
        todo!("bool_permute")
    }

    fn bool_flip(_tensor: BoolTensor<Ember>, _axes: &[usize]) -> BoolTensor<Ember> {
        todo!("bool_flip")
    }

    fn bool_equal(_lhs: BoolTensor<Ember>, _rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        todo!("bool_equal")
    }

    fn bool_not(_tensor: BoolTensor<Ember>) -> BoolTensor<Ember> {
        todo!("bool_not")
    }

    fn bool_and(_lhs: BoolTensor<Ember>, _rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        todo!("bool_and")
    }

    fn bool_or(_lhs: BoolTensor<Ember>, _rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        todo!("bool_or")
    }

    fn bool_xor(_lhs: BoolTensor<Ember>, _rhs: BoolTensor<Ember>) -> BoolTensor<Ember> {
        todo!("bool_xor")
    }

    fn bool_expand(_tensor: BoolTensor<Ember>, _shape: Shape) -> BoolTensor<Ember> {
        todo!("bool_expand")
    }

    // Missing methods
    fn bool_zeros(shape: Shape, device: &Device<Ember>) -> BoolTensor<Ember> {
        Self::bool_empty(shape, device)
    }

    fn bool_ones(_shape: Shape, _device: &Device<Ember>) -> BoolTensor<Ember> {
        todo!("bool_ones")
    }

    fn bool_mask_where(
        _tensor: BoolTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_mask_where")
    }

    fn bool_mask_fill(
        _tensor: BoolTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: bool,
    ) -> BoolTensor<Ember> {
        todo!("bool_mask_fill")
    }

    fn bool_gather(
        _dim: usize,
        _tensor: BoolTensor<Ember>,
        _indices: IntTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_gather")
    }

    fn bool_scatter_or(
        _dim: usize,
        _tensor: BoolTensor<Ember>,
        _indices: IntTensor<Ember>,
        _value: BoolTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("bool_scatter_or")
    }

    fn bool_equal_elem(_lhs: BoolTensor<Ember>, _rhs: bool) -> BoolTensor<Ember> {
        todo!("bool_equal_elem")
    }

    fn bool_unfold(
        _tensor: BoolTensor<Ember>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> BoolTensor<Ember> {
        todo!("bool_unfold")
    }
}

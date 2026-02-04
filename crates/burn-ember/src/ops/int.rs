//! Int tensor operations for the Ember backend.

use burn_backend::{
    Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{IntDType, Shape, Slice};
use num_traits::ToPrimitive;

use crate::{Ember, EmberTensor, ops::matmul};

impl IntTensorOps<Ember> for Ember {
    fn int_from_data(data: TensorData, _device: &Device<Ember>) -> IntTensor<Ember> {
        EmberTensor::from_data(data)
    }

    async fn int_into_data(tensor: IntTensor<Ember>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_device(_tensor: &IntTensor<Ember>) -> Device<Ember> {
        Default::default()
    }

    fn int_to_device(tensor: IntTensor<Ember>, _device: &Device<Ember>) -> IntTensor<Ember> {
        tensor
    }

    fn int_reshape(tensor: IntTensor<Ember>, shape: Shape) -> IntTensor<Ember> {
        tensor.reshape(shape)
    }

    fn int_slice(tensor: IntTensor<Ember>, slices: &[Slice]) -> IntTensor<Ember> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn int_empty(shape: Shape, _device: &Device<Ember>, dtype: IntDType) -> IntTensor<Ember> {
        EmberTensor::empty(shape, dtype.into())
    }

    fn int_mask_where(
        _tensor: IntTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_mask_where")
    }

    fn int_mask_fill(
        _tensor: IntTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: Scalar,
    ) -> IntTensor<Ember> {
        todo!("int_mask_fill")
    }

    fn int_slice_assign(
        tensor: IntTensor<Ember>,
        slices: &[Slice],
        value: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn int_gather(
        _dim: usize,
        _tensor: IntTensor<Ember>,
        _indices: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_gather")
    }

    fn int_scatter_add(
        _dim: usize,
        _tensor: IntTensor<Ember>,
        _indices: IntTensor<Ember>,
        _value: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_scatter_add")
    }

    fn int_select(
        _tensor: IntTensor<Ember>,
        _dim: usize,
        _indices: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_select")
    }

    fn int_select_add(
        _tensor: IntTensor<Ember>,
        _dim: usize,
        _indices: IntTensor<Ember>,
        _value: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_select_add")
    }

    fn int_equal(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::int_equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: IntTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::int_equal_elem(lhs, rhs.to_f64().unwrap() as i64)
    }

    fn int_greater(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::int_greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: IntTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::int_greater_elem(lhs, rhs.to_f64().unwrap() as i64)
    }

    fn int_greater_equal(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::int_greater_equal_elem(lhs, rhs.to_f64().unwrap() as i64)
    }

    fn int_lower(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::int_lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: IntTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::int_lower_elem(lhs, rhs.to_f64().unwrap() as i64)
    }

    fn int_lower_equal(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::int_lower_equal_elem(lhs, rhs.to_f64().unwrap() as i64)
    }

    fn int_add(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_add")
    }

    fn int_add_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_add_scalar")
    }

    fn int_sub(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_sub")
    }

    fn int_sub_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_sub_scalar")
    }

    fn int_mul(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_mul")
    }

    fn int_mul_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_mul_scalar")
    }

    fn int_div(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_div")
    }

    fn int_div_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_div_scalar")
    }

    fn int_remainder(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_remainder")
    }

    fn int_remainder_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_remainder_scalar")
    }

    fn int_into_float(_tensor: IntTensor<Ember>) -> FloatTensor<Ember> {
        todo!("int_into_float")
    }

    fn int_swap_dims(tensor: IntTensor<Ember>, dim1: usize, dim2: usize) -> IntTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn int_permute(_tensor: IntTensor<Ember>, _axes: &[usize]) -> IntTensor<Ember> {
        todo!("int_permute")
    }

    fn int_flip(_tensor: IntTensor<Ember>, _axes: &[usize]) -> IntTensor<Ember> {
        todo!("int_flip")
    }

    fn int_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &Device<Ember>,
    ) -> IntTensor<Ember> {
        todo!("int_random")
    }

    fn int_expand(tensor: IntTensor<Ember>, shape: Shape) -> IntTensor<Ember> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn int_matmul(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        matmul::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<Ember>) -> IntTensor<Ember> {
        crate::ops::reduce::sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<Ember>) -> IntTensor<Ember> {
        crate::ops::reduce::prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn int_cumsum(_tensor: IntTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("int_cumsum")
    }

    fn int_cumprod(_tensor: IntTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("int_cumprod")
    }

    fn int_cummin(_tensor: IntTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("int_cummin")
    }

    fn int_cummax(_tensor: IntTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("int_cummax")
    }

    fn int_argmax(tensor: IntTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::argmax(tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::argmin(tensor, dim)
    }

    fn int_abs(_tensor: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_abs")
    }

    fn bitwise_and(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_and")
    }

    fn bitwise_and_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("bitwise_and_scalar")
    }

    fn bitwise_or(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_or")
    }

    fn bitwise_or_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("bitwise_or_scalar")
    }

    fn bitwise_xor(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_xor")
    }

    fn bitwise_xor_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("bitwise_xor_scalar")
    }

    fn bitwise_not(_tensor: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_not")
    }

    fn bitwise_left_shift(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_left_shift")
    }

    fn bitwise_left_shift_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("bitwise_left_shift_scalar")
    }

    fn bitwise_right_shift(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("bitwise_right_shift")
    }

    fn bitwise_right_shift_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("bitwise_right_shift_scalar")
    }

    fn int_cast(_tensor: IntTensor<Ember>, _dtype: IntDType) -> IntTensor<Ember> {
        todo!("int_cast")
    }

    fn int_unfold(
        _tensor: IntTensor<Ember>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> IntTensor<Ember> {
        todo!("int_unfold")
    }
}

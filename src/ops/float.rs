//! Float tensor operations for the Ember backend.

use burn_backend::{
    Distribution, ExecutionError, FloatDType, Scalar, TensorData,
    ops::FloatTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::{Ember, EmberTensor};

impl FloatTensorOps<Ember> for Ember {
    fn float_from_data(data: TensorData, _device: &Device<Ember>) -> FloatTensor<Ember> {
        EmberTensor::from_data(data)
    }

    fn float_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &Device<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_random")
    }

    async fn float_into_data(tensor: FloatTensor<Ember>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn float_device(_tensor: &FloatTensor<Ember>) -> Device<Ember> {
        // CPU backend: all tensors are on the default device
        Default::default()
    }

    fn float_to_device(tensor: FloatTensor<Ember>, _device: &Device<Ember>) -> FloatTensor<Ember> {
        // CPU backend: no-op, tensors are always on CPU
        tensor
    }

    fn float_into_int(_tensor: FloatTensor<Ember>) -> IntTensor<Ember> {
        todo!("float_into_int")
    }

    fn float_empty(shape: Shape, _device: &Device<Ember>, dtype: FloatDType) -> FloatTensor<Ember> {
        EmberTensor::empty(shape, dtype.into())
    }

    fn float_add(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_add")
    }

    fn float_add_scalar(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> FloatTensor<Ember> {
        todo!("float_add_scalar")
    }

    fn float_sub(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_sub")
    }

    fn float_sub_scalar(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> FloatTensor<Ember> {
        todo!("float_sub_scalar")
    }

    fn float_mul(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_mul")
    }

    fn float_mul_scalar(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> FloatTensor<Ember> {
        todo!("float_mul_scalar")
    }

    fn float_div(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_div")
    }

    fn float_div_scalar(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> FloatTensor<Ember> {
        todo!("float_div_scalar")
    }

    fn float_remainder(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_remainder")
    }

    fn float_remainder_scalar(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> FloatTensor<Ember> {
        todo!("float_remainder_scalar")
    }

    fn float_matmul(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_matmul")
    }

    fn float_cross(
        _lhs: FloatTensor<Ember>,
        _rhs: FloatTensor<Ember>,
        _dim: usize,
    ) -> FloatTensor<Ember> {
        todo!("float_cross")
    }

    fn float_recip(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_recip")
    }

    fn float_swap_dims(tensor: FloatTensor<Ember>, dim1: usize, dim2: usize) -> FloatTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn float_permute(_tensor: FloatTensor<Ember>, _axes: &[usize]) -> FloatTensor<Ember> {
        todo!("float_permute")
    }

    fn float_flip(_tensor: FloatTensor<Ember>, _axes: &[usize]) -> FloatTensor<Ember> {
        todo!("float_flip")
    }

    fn float_reshape(tensor: FloatTensor<Ember>, shape: Shape) -> FloatTensor<Ember> {
        tensor.reshape(shape)
    }

    fn float_gather(
        _dim: usize,
        _tensor: FloatTensor<Ember>,
        _indices: IntTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_gather")
    }

    fn float_scatter_add(
        _dim: usize,
        _tensor: FloatTensor<Ember>,
        _indices: IntTensor<Ember>,
        _value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_scatter_add")
    }

    fn float_select(
        _tensor: FloatTensor<Ember>,
        _dim: usize,
        _indices: IntTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_select")
    }

    fn float_select_add(
        _tensor: FloatTensor<Ember>,
        _dim: usize,
        _indices: IntTensor<Ember>,
        _value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_select_add")
    }

    fn float_slice(_tensor: FloatTensor<Ember>, _slices: &[Slice]) -> FloatTensor<Ember> {
        todo!("float_slice")
    }

    fn float_slice_assign(
        _tensor: FloatTensor<Ember>,
        _slices: &[Slice],
        _value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_slice_assign")
    }

    fn float_mask_where(
        _tensor: FloatTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        todo!("float_mask_where")
    }

    fn float_mask_fill(
        _tensor: FloatTensor<Ember>,
        _mask: BoolTensor<Ember>,
        _value: Scalar,
    ) -> FloatTensor<Ember> {
        todo!("float_mask_fill")
    }

    fn float_equal(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        todo!("float_equal")
    }

    fn float_equal_elem(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> BoolTensor<Ember> {
        todo!("float_equal_elem")
    }

    fn float_greater(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        todo!("float_greater")
    }

    fn float_greater_elem(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> BoolTensor<Ember> {
        todo!("float_greater_elem")
    }

    fn float_greater_equal(
        _lhs: FloatTensor<Ember>,
        _rhs: FloatTensor<Ember>,
    ) -> BoolTensor<Ember> {
        todo!("float_greater_equal")
    }

    fn float_greater_equal_elem(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> BoolTensor<Ember> {
        todo!("float_greater_equal_elem")
    }

    fn float_lower(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        todo!("float_lower")
    }

    fn float_lower_elem(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> BoolTensor<Ember> {
        todo!("float_lower_elem")
    }

    fn float_lower_equal(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        todo!("float_lower_equal")
    }

    fn float_lower_equal_elem(_lhs: FloatTensor<Ember>, _rhs: Scalar) -> BoolTensor<Ember> {
        todo!("float_lower_equal_elem")
    }

    fn float_sum(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_sum")
    }

    fn float_sum_dim(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_sum_dim")
    }

    fn float_mean_dim(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_mean_dim")
    }

    fn float_cumsum(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_cumsum")
    }

    fn float_cumprod(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_cumprod")
    }

    fn float_cummin(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_cummin")
    }

    fn float_cummax(_tensor: FloatTensor<Ember>, _dim: usize) -> FloatTensor<Ember> {
        todo!("float_cummax")
    }

    fn float_cast(_tensor: FloatTensor<Ember>, _dtype: FloatDType) -> FloatTensor<Ember> {
        todo!("float_cast")
    }

    fn float_exp(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_exp")
    }

    fn float_log(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_log")
    }

    fn float_log1p(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_log1p")
    }

    fn float_powf(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_powf")
    }

    fn float_powf_scalar_impl(_tensor: FloatTensor<Ember>, _value: Scalar) -> FloatTensor<Ember> {
        todo!("float_powf_scalar_impl")
    }

    fn float_sqrt(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_sqrt")
    }

    fn float_abs(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_abs")
    }

    fn float_cos(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_cos")
    }

    fn float_sin(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_sin")
    }

    fn float_tan(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_tan")
    }

    fn float_cosh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_cosh")
    }

    fn float_sinh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_sinh")
    }

    fn float_tanh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_tanh")
    }

    fn float_acos(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_acos")
    }

    fn float_acosh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_acosh")
    }

    fn float_asin(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_asin")
    }

    fn float_asinh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_asinh")
    }

    fn float_atan(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_atan")
    }

    fn float_atanh(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_atanh")
    }

    fn float_atan2(_lhs: FloatTensor<Ember>, _rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_atan2")
    }

    fn float_round(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_round")
    }

    fn float_floor(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_floor")
    }

    fn float_ceil(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_ceil")
    }

    fn float_trunc(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_trunc")
    }

    fn float_erf(_tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        todo!("float_erf")
    }

    fn float_argmax(_tensor: FloatTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("float_argmax")
    }

    fn float_argmin(_tensor: FloatTensor<Ember>, _dim: usize) -> IntTensor<Ember> {
        todo!("float_argmin")
    }

    fn float_expand(_tensor: FloatTensor<Ember>, _shape: Shape) -> FloatTensor<Ember> {
        todo!("float_expand")
    }

    fn float_unfold(
        _tensor: FloatTensor<Ember>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> FloatTensor<Ember> {
        todo!("float_unfold")
    }
}

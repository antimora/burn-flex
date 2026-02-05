//! Int tensor operations for the Ember backend.

use burn_backend::{
    DType, Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, IntDType, Shape, Slice};
use num_traits::ToPrimitive;

use crate::Layout;
use crate::ops::binary::{int_binary_op, int_scalar_op};
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
        tensor: IntTensor<Ember>,
        mask: BoolTensor<Ember>,
        value: IntTensor<Ember>,
    ) -> IntTensor<Ember> {
        crate::ops::mask::mask_where_i64(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Ember>,
        mask: BoolTensor<Ember>,
        value: Scalar,
    ) -> IntTensor<Ember> {
        crate::ops::mask::mask_fill_i64(tensor, mask, value.to_i64().unwrap())
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

    fn int_add(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        int_binary_op(lhs, rhs, |a, b| a + b)
    }

    fn int_add_scalar(lhs: IntTensor<Ember>, rhs: Scalar) -> IntTensor<Ember> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a + b)
    }

    fn int_sub(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        int_binary_op(lhs, rhs, |a, b| a - b)
    }

    fn int_sub_scalar(lhs: IntTensor<Ember>, rhs: Scalar) -> IntTensor<Ember> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a - b)
    }

    fn int_mul(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        int_binary_op(lhs, rhs, |a, b| a * b)
    }

    fn int_mul_scalar(lhs: IntTensor<Ember>, rhs: Scalar) -> IntTensor<Ember> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a * b)
    }

    fn int_div(lhs: IntTensor<Ember>, rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        int_binary_op(lhs, rhs, |a, b| a / b)
    }

    fn int_div_scalar(lhs: IntTensor<Ember>, rhs: Scalar) -> IntTensor<Ember> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a / b)
    }

    fn int_remainder(_lhs: IntTensor<Ember>, _rhs: IntTensor<Ember>) -> IntTensor<Ember> {
        todo!("int_remainder")
    }

    fn int_remainder_scalar(_lhs: IntTensor<Ember>, _rhs: Scalar) -> IntTensor<Ember> {
        todo!("int_remainder_scalar")
    }

    fn int_into_float(tensor: IntTensor<Ember>) -> FloatTensor<Ember> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let dtype = tensor.dtype();

        let float_data: Vec<f32> = match dtype {
            DType::I64 => tensor.storage::<i64>().iter().map(|x| *x as f32).collect(),
            DType::I32 => tensor.storage::<i32>().iter().map(|x| *x as f32).collect(),
            DType::I16 => tensor.storage::<i16>().iter().map(|x| *x as f32).collect(),
            DType::I8 => tensor.storage::<i8>().iter().map(|x| *x as f32).collect(),
            DType::U64 => tensor.storage::<u64>().iter().map(|x| *x as f32).collect(),
            DType::U32 => tensor.storage::<u32>().iter().map(|x| *x as f32).collect(),
            DType::U16 => tensor.storage::<u16>().iter().map(|x| *x as f32).collect(),
            DType::U8 => tensor.storage::<u8>().iter().map(|x| *x as f32).collect(),
            _ => panic!("int_into_float: unsupported dtype {:?}", dtype),
        };

        EmberTensor::new(
            Bytes::from_elems(float_data),
            Layout::contiguous(shape),
            DType::F32,
        )
    }

    fn int_swap_dims(tensor: IntTensor<Ember>, dim1: usize, dim2: usize) -> IntTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<Ember>, axes: &[usize]) -> IntTensor<Ember> {
        tensor.permute(axes)
    }

    fn int_flip(tensor: IntTensor<Ember>, axes: &[usize]) -> IntTensor<Ember> {
        crate::ops::flip::flip(tensor, axes)
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

#[cfg(test)]
mod tests {
    use burn_tensor::{Int, Tensor, TensorData};

    use crate::Ember;

    #[test]
    fn test_int_add() {
        let a: Tensor<Ember, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Ember, 2, Int> = Tensor::from_data([[5i64, 6], [7, 8]], &Default::default());

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([[6i64, 8], [10, 12]]));
    }

    #[test]
    fn test_int_sub() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let b: Tensor<Ember, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());

        let result = a - b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([9i64, 18, 27]));
    }

    #[test]
    fn test_int_mul() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([2i64, 3, 4], &Default::default());
        let b: Tensor<Ember, 1, Int> = Tensor::from_data([5i64, 6, 7], &Default::default());

        let result = a * b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([10i64, 18, 28]));
    }

    #[test]
    fn test_int_div() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([10i64, 21, 35], &Default::default());
        let b: Tensor<Ember, 1, Int> = Tensor::from_data([2i64, 7, 5], &Default::default());

        let result = a / b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([5i64, 3, 7]));
    }

    #[test]
    fn test_int_add_scalar() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());
        let result = a + 10;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([11i64, 12, 13]));
    }

    #[test]
    fn test_int_sub_scalar() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let result = a - 5;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([5i64, 15, 25]));
    }

    #[test]
    fn test_int_mul_scalar() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());
        let result = a * 3;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([3i64, 6, 9]));
    }

    #[test]
    fn test_int_div_scalar() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let result = a / 5;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([2i64, 4, 6]));
    }

    #[test]
    fn test_int_add_transposed() {
        let a: Tensor<Ember, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Ember, 2, Int> =
            Tensor::from_data([[10i64, 20], [30, 40]], &Default::default());

        let a_t = a.transpose();
        let b_t = b.transpose();

        let result = a_t + b_t;
        let data = result.into_data();

        // a_t = [[1, 3], [2, 4]], b_t = [[10, 30], [20, 40]]
        // result = [[11, 33], [22, 44]]
        assert_eq!(data, TensorData::from([[11i64, 33], [22, 44]]));
    }

    #[test]
    fn test_int_negative_values() {
        let a: Tensor<Ember, 1, Int> = Tensor::from_data([-5i64, 10, -15], &Default::default());
        let b: Tensor<Ember, 1, Int> = Tensor::from_data([5i64, -10, 15], &Default::default());

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([0i64, 0, 0]));
    }

    #[test]
    fn test_int_into_float() {
        let t: Tensor<Ember, 1, Int> = Tensor::from_data([1i64, 2, -3, 0], &Default::default());
        let float_t: Tensor<Ember, 1> = t.float();
        let data: Vec<f32> = float_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![1.0f32, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn test_int_into_float_2d() {
        let t: Tensor<Ember, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let float_t: Tensor<Ember, 2> = t.float();
        let data: Vec<f32> = float_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}

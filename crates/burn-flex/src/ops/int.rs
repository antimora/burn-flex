//! Int tensor operations for the Flex backend.

use alloc::vec::Vec;
use burn_backend::{
    DType, Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, IntDType, Shape, Slice};
use num_traits::ToPrimitive;

use crate::Layout;
use crate::ops::binary::{binary_op_typed, int_binary_op, int_scalar_op, scalar_op_typed};
use crate::{Flex, FlexTensor, ops::matmul};

impl IntTensorOps<Flex> for Flex {
    fn int_from_data(data: TensorData, _device: &Device<Flex>) -> IntTensor<Flex> {
        FlexTensor::from_data(data)
    }

    async fn int_into_data(tensor: IntTensor<Flex>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_device(_tensor: &IntTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn int_to_device(tensor: IntTensor<Flex>, _device: &Device<Flex>) -> IntTensor<Flex> {
        tensor
    }

    fn int_reshape(tensor: IntTensor<Flex>, shape: Shape) -> IntTensor<Flex> {
        tensor.reshape(shape)
    }

    fn int_slice(tensor: IntTensor<Flex>, slices: &[Slice]) -> IntTensor<Flex> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn int_empty(shape: Shape, _device: &Device<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        FlexTensor::empty(shape, dtype.into())
    }

    fn int_mask_where(
        tensor: IntTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::mask::mask_where_i64(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: Scalar,
    ) -> IntTensor<Flex> {
        crate::ops::mask::mask_fill_i64(tensor, mask, value.to_i64().unwrap())
    }

    fn int_slice_assign(
        tensor: IntTensor<Flex>,
        slices: &[Slice],
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::gather_scatter::gather_i64(tensor, dim, indices)
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Flex>,
        indices: IntTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::gather_scatter::scatter_add_i64(tensor, dim, indices, value)
    }

    fn int_select(
        tensor: IntTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::gather_scatter::select_i64(tensor, dim, indices)
    }

    fn int_select_add(
        tensor: IntTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
        value: IntTensor<Flex>,
    ) -> IntTensor<Flex> {
        crate::ops::gather_scatter::select_add_i64(tensor, dim, indices, value)
    }

    fn int_equal(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::int_equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: IntTensor<Flex>, rhs: Scalar) -> BoolTensor<Flex> {
        crate::ops::comparison::int_equal_elem(lhs, rhs.to_i64().unwrap())
    }

    fn int_greater(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: IntTensor<Flex>, rhs: Scalar) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater_elem(lhs, rhs.to_i64().unwrap())
    }

    fn int_greater_equal(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Flex>, rhs: Scalar) -> BoolTensor<Flex> {
        crate::ops::comparison::int_greater_equal_elem(lhs, rhs.to_i64().unwrap())
    }

    fn int_lower(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: IntTensor<Flex>, rhs: Scalar) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower_elem(lhs, rhs.to_i64().unwrap())
    }

    fn int_lower_equal(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Flex>, rhs: Scalar) -> BoolTensor<Flex> {
        crate::ops::comparison::int_lower_equal_elem(lhs, rhs.to_i64().unwrap())
    }

    fn int_add(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a + b)
    }

    fn int_add_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a + b)
    }

    fn int_sub(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a - b)
    }

    fn int_sub_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a - b)
    }

    fn int_mul(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a * b)
    }

    fn int_mul_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a * b)
    }

    fn int_div(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        // U64 values > i64::MAX produce wrong results through i64 cast
        if lhs.dtype() == DType::U64 {
            let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);
            return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a / b);
        }
        int_binary_op(lhs, rhs, |a, b| a / b)
    }

    fn int_div_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a / b);
        }
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a / b)
    }

    fn int_remainder(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        // U64 values > i64::MAX produce wrong results through i64 cast
        if lhs.dtype() == DType::U64 {
            let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);
            return binary_op_typed(lhs, &rhs, |a: u64, b: u64| a % b);
        }
        // Python/PyTorch-style remainder: result has same sign as divisor
        int_binary_op(lhs, rhs, |a, b| ((a % b) + b) % b)
    }

    fn int_remainder_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        if lhs.dtype() == DType::U64 {
            return scalar_op_typed(lhs, rhs.to_u64().unwrap(), |a: u64, b: u64| a % b);
        }
        // Python/PyTorch-style remainder: result has same sign as divisor
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| ((a % b) + b) % b)
    }

    // i64/u64 > 2^24 lose precision when converted to f32 (matches PyTorch).
    fn int_into_float(tensor: IntTensor<Flex>) -> FloatTensor<Flex> {
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

        FlexTensor::new(
            Bytes::from_elems(float_data),
            Layout::contiguous(shape),
            DType::F32,
        )
    }

    fn int_swap_dims(tensor: IntTensor<Flex>, dim1: usize, dim2: usize) -> IntTensor<Flex> {
        tensor.transpose(dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<Flex>, axes: &[usize]) -> IntTensor<Flex> {
        tensor.permute(axes)
    }

    fn int_flip(tensor: IntTensor<Flex>, axes: &[usize]) -> IntTensor<Flex> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Flex>,
    ) -> IntTensor<Flex> {
        let mut seed = crate::backend::SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            crate::backend::get_seeded_rng()
        };
        let data = TensorData::random::<i64, _, _>(shape, distribution, &mut rng);
        *seed = Some(rng);
        FlexTensor::from_data(data)
    }

    fn int_expand(tensor: IntTensor<Flex>, shape: Shape) -> IntTensor<Flex> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn int_matmul(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        matmul::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::reduce::prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::cumulative::cumsum_i64(tensor, dim)
    }

    fn int_cumprod(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::cumulative::cumprod_i64(tensor, dim)
    }

    fn int_cummin(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::cumulative::cummin_i64(tensor, dim)
    }

    fn int_cummax(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::cumulative::cummax_i64(tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::argmax(tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        crate::ops::unary::int_abs(tensor)
    }

    fn bitwise_and(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a & b)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a & b)
    }

    fn bitwise_or(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a | b)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a | b)
    }

    fn bitwise_xor(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a ^ b)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a ^ b)
    }

    fn bitwise_not(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        // Use scalar op with dummy value, only applying NOT to lhs
        int_scalar_op(tensor, 0, |a, _| !a)
    }

    // Shift amounts masked to type width via wrapping_shl/wrapping_shr.
    fn bitwise_left_shift(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a.wrapping_shl(b as u32))
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shl(b as u32))
    }

    fn bitwise_right_shift(lhs: IntTensor<Flex>, rhs: IntTensor<Flex>) -> IntTensor<Flex> {
        int_binary_op(lhs, rhs, |a, b| a.wrapping_shr(b as u32))
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Flex>, rhs: Scalar) -> IntTensor<Flex> {
        int_scalar_op(lhs, rhs.to_i64().unwrap(), |a, b| a.wrapping_shr(b as u32))
    }

    fn int_cast(tensor: IntTensor<Flex>, dtype: IntDType) -> IntTensor<Flex> {
        let target_dtype: DType = dtype.into();

        // If already the target dtype, return as-is
        if tensor.dtype() == target_dtype {
            return tensor;
        }

        // Make contiguous for easier iteration
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();

        // Helper macro to convert between types
        macro_rules! cast_impl {
            ($src_type:ty, $dst_type:ty, $dst_dtype:expr) => {{
                let src: &[$src_type] = tensor.storage();
                let dst: Vec<$dst_type> = src.iter().map(|&x| x as $dst_type).collect();
                FlexTensor::new(
                    Bytes::from_elems(dst),
                    Layout::contiguous(shape),
                    $dst_dtype,
                )
            }};
        }

        // Match source dtype to target dtype
        match (tensor.dtype(), target_dtype) {
            // From I64
            (DType::I64, DType::I32) => cast_impl!(i64, i32, DType::I32),
            (DType::I64, DType::I16) => cast_impl!(i64, i16, DType::I16),
            (DType::I64, DType::I8) => cast_impl!(i64, i8, DType::I8),
            (DType::I64, DType::U64) => cast_impl!(i64, u64, DType::U64),
            (DType::I64, DType::U32) => cast_impl!(i64, u32, DType::U32),
            (DType::I64, DType::U16) => cast_impl!(i64, u16, DType::U16),
            (DType::I64, DType::U8) => cast_impl!(i64, u8, DType::U8),

            // From I32
            (DType::I32, DType::I64) => cast_impl!(i32, i64, DType::I64),
            (DType::I32, DType::I16) => cast_impl!(i32, i16, DType::I16),
            (DType::I32, DType::I8) => cast_impl!(i32, i8, DType::I8),
            (DType::I32, DType::U64) => cast_impl!(i32, u64, DType::U64),
            (DType::I32, DType::U32) => cast_impl!(i32, u32, DType::U32),
            (DType::I32, DType::U16) => cast_impl!(i32, u16, DType::U16),
            (DType::I32, DType::U8) => cast_impl!(i32, u8, DType::U8),

            // From I16
            (DType::I16, DType::I64) => cast_impl!(i16, i64, DType::I64),
            (DType::I16, DType::I32) => cast_impl!(i16, i32, DType::I32),
            (DType::I16, DType::I8) => cast_impl!(i16, i8, DType::I8),
            (DType::I16, DType::U64) => cast_impl!(i16, u64, DType::U64),
            (DType::I16, DType::U32) => cast_impl!(i16, u32, DType::U32),
            (DType::I16, DType::U16) => cast_impl!(i16, u16, DType::U16),
            (DType::I16, DType::U8) => cast_impl!(i16, u8, DType::U8),

            // From I8
            (DType::I8, DType::I64) => cast_impl!(i8, i64, DType::I64),
            (DType::I8, DType::I32) => cast_impl!(i8, i32, DType::I32),
            (DType::I8, DType::I16) => cast_impl!(i8, i16, DType::I16),
            (DType::I8, DType::U64) => cast_impl!(i8, u64, DType::U64),
            (DType::I8, DType::U32) => cast_impl!(i8, u32, DType::U32),
            (DType::I8, DType::U16) => cast_impl!(i8, u16, DType::U16),
            (DType::I8, DType::U8) => cast_impl!(i8, u8, DType::U8),

            // From U64
            (DType::U64, DType::I64) => cast_impl!(u64, i64, DType::I64),
            (DType::U64, DType::I32) => cast_impl!(u64, i32, DType::I32),
            (DType::U64, DType::I16) => cast_impl!(u64, i16, DType::I16),
            (DType::U64, DType::I8) => cast_impl!(u64, i8, DType::I8),
            (DType::U64, DType::U32) => cast_impl!(u64, u32, DType::U32),
            (DType::U64, DType::U16) => cast_impl!(u64, u16, DType::U16),
            (DType::U64, DType::U8) => cast_impl!(u64, u8, DType::U8),

            // From U32
            (DType::U32, DType::I64) => cast_impl!(u32, i64, DType::I64),
            (DType::U32, DType::I32) => cast_impl!(u32, i32, DType::I32),
            (DType::U32, DType::I16) => cast_impl!(u32, i16, DType::I16),
            (DType::U32, DType::I8) => cast_impl!(u32, i8, DType::I8),
            (DType::U32, DType::U64) => cast_impl!(u32, u64, DType::U64),
            (DType::U32, DType::U16) => cast_impl!(u32, u16, DType::U16),
            (DType::U32, DType::U8) => cast_impl!(u32, u8, DType::U8),

            // From U16
            (DType::U16, DType::I64) => cast_impl!(u16, i64, DType::I64),
            (DType::U16, DType::I32) => cast_impl!(u16, i32, DType::I32),
            (DType::U16, DType::I16) => cast_impl!(u16, i16, DType::I16),
            (DType::U16, DType::I8) => cast_impl!(u16, i8, DType::I8),
            (DType::U16, DType::U64) => cast_impl!(u16, u64, DType::U64),
            (DType::U16, DType::U32) => cast_impl!(u16, u32, DType::U32),
            (DType::U16, DType::U8) => cast_impl!(u16, u8, DType::U8),

            // From U8
            (DType::U8, DType::I64) => cast_impl!(u8, i64, DType::I64),
            (DType::U8, DType::I32) => cast_impl!(u8, i32, DType::I32),
            (DType::U8, DType::I16) => cast_impl!(u8, i16, DType::I16),
            (DType::U8, DType::I8) => cast_impl!(u8, i8, DType::I8),
            (DType::U8, DType::U64) => cast_impl!(u8, u64, DType::U64),
            (DType::U8, DType::U32) => cast_impl!(u8, u32, DType::U32),
            (DType::U8, DType::U16) => cast_impl!(u8, u16, DType::U16),

            _ => panic!(
                "int_cast: unsupported conversion from {:?} to {:?}",
                tensor.dtype(),
                target_dtype
            ),
        }
    }

    fn int_unfold(
        tensor: IntTensor<Flex>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Flex> {
        crate::ops::unfold::unfold_int(tensor, dim, size, step)
    }

    fn int_neg(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        int_scalar_op(tensor, 0i64, |a, _| a.wrapping_neg())
    }

    fn int_clamp(tensor: IntTensor<Flex>, min: Scalar, max: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let min_val = min.to_u64().unwrap();
            let max_val = max.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.clamp(min_val, max_val));
        }
        let min_val = min.to_i64().unwrap();
        let max_val = max.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.clamp(min_val, max_val))
    }

    fn int_clamp_min(tensor: IntTensor<Flex>, min: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let min_val = min.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.max(min_val));
        }
        let min_val = min.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.max(min_val))
    }

    fn int_clamp_max(tensor: IntTensor<Flex>, max: Scalar) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            let max_val = max.to_u64().unwrap();
            return scalar_op_typed(tensor, 0u64, move |x: u64, _| x.min(max_val));
        }
        let max_val = max.to_i64().unwrap();
        int_scalar_op(tensor, 0i64, move |x, _| x.min(max_val))
    }

    fn int_sign(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        if tensor.dtype() == DType::U64 {
            return scalar_op_typed(tensor, 0u64, |x: u64, _| if x > 0 { 1 } else { 0 });
        }
        int_scalar_op(tensor, 0i64, |x, _| {
            if x > 0 {
                1
            } else if x < 0 {
                -1
            } else {
                0
            }
        })
    }

    fn int_mean(tensor: IntTensor<Flex>) -> IntTensor<Flex> {
        let n = tensor.layout().num_elements();
        assert!(n > 0, "int_mean: cannot take mean of empty tensor");
        let sum_result = crate::ops::reduce::sum(tensor);
        let data: &[i64] = sum_result.storage();
        let mean_val = data[0] / n as i64;
        FlexTensor::new(
            Bytes::from_elems(alloc::vec![mean_val]),
            Layout::contiguous(Shape::from(alloc::vec![1])),
            DType::I64,
        )
    }

    fn int_max_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::max_dim(tensor, dim)
    }

    fn int_min_dim(tensor: IntTensor<Flex>, dim: usize) -> IntTensor<Flex> {
        crate::ops::reduce::min_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Flex>,
        dim: usize,
    ) -> (IntTensor<Flex>, IntTensor<Flex>) {
        crate::ops::reduce::max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Flex>,
        dim: usize,
    ) -> (IntTensor<Flex>, IntTensor<Flex>) {
        crate::ops::reduce::min_dim_with_indices(tensor, dim)
    }

    fn int_any(tensor: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::any_int(tensor)
    }

    fn int_any_dim(tensor: IntTensor<Flex>, dim: usize) -> BoolTensor<Flex> {
        crate::ops::comparison::any_int_dim(tensor, dim)
    }

    fn int_all(tensor: IntTensor<Flex>) -> BoolTensor<Flex> {
        crate::ops::comparison::all_int(tensor)
    }

    fn int_all_dim(tensor: IntTensor<Flex>, dim: usize) -> BoolTensor<Flex> {
        crate::ops::comparison::all_int_dim(tensor, dim)
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::{Int, Tensor, TensorData};

    use crate::Flex;
    use crate::FlexTensor;
    use burn_backend::ops::IntTensorOps;

    #[test]
    fn test_int_add() {
        let a: Tensor<Flex, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Flex, 2, Int> = Tensor::from_data([[5i64, 6], [7, 8]], &Default::default());

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([[6i64, 8], [10, 12]]));
    }

    #[test]
    fn test_int_sub() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());

        let result = a - b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([9i64, 18, 27]));
    }

    #[test]
    fn test_int_mul() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([2i64, 3, 4], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([5i64, 6, 7], &Default::default());

        let result = a * b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([10i64, 18, 28]));
    }

    #[test]
    fn test_int_div() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 21, 35], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([2i64, 7, 5], &Default::default());

        let result = a / b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([5i64, 3, 7]));
    }

    #[test]
    fn test_int_add_scalar() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());
        let result = a + 10;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([11i64, 12, 13]));
    }

    #[test]
    fn test_int_sub_scalar() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let result = a - 5;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([5i64, 15, 25]));
    }

    #[test]
    fn test_int_mul_scalar() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3], &Default::default());
        let result = a * 3;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([3i64, 6, 9]));
    }

    #[test]
    fn test_int_div_scalar() {
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 20, 30], &Default::default());
        let result = a / 5;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([2i64, 4, 6]));
    }

    #[test]
    fn test_int_add_transposed() {
        let a: Tensor<Flex, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Flex, 2, Int> =
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
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([-5i64, 10, -15], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([5i64, -10, 15], &Default::default());

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([0i64, 0, 0]));
    }

    #[test]
    fn test_int_into_float() {
        let t: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, -3, 0], &Default::default());
        let float_t: Tensor<Flex, 1> = t.float();
        let data: Vec<f32> = float_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![1.0f32, 2.0, -3.0, 0.0]);
    }

    #[test]
    fn test_int_into_float_2d() {
        let t: Tensor<Flex, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let float_t: Tensor<Flex, 2> = t.float();
        let data: Vec<f32> = float_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    // === Non-contiguous (negative stride) tests ===

    #[test]
    fn test_int_add_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // [10, 20, 30, 40] flipped -> [40, 30, 20, 10]
        // [4, 3, 2, 1] + [40, 30, 20, 10] = [44, 33, 22, 11]
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3, 4], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 20, 30, 40], &Default::default());

        let a = a.flip([0]);
        let b = b.flip([0]);

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([44i64, 33, 22, 11]));
    }

    #[test]
    fn test_int_sub_flipped() {
        // [10, 20, 30, 40] flipped -> [40, 30, 20, 10]
        // [1, 2, 3, 4] (contiguous)
        // [40, 30, 20, 10] - [1, 2, 3, 4] = [39, 28, 17, 6]
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([10i64, 20, 30, 40], &Default::default());
        let b: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3, 4], &Default::default());

        let a = a.flip([0]);

        let result = a - b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([39i64, 28, 17, 6]));
    }

    #[test]
    fn test_int_mul_flipped_2d() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // [[10, 20], [30, 40]]
        // [[3, 4], [1, 2]] * [[10, 20], [30, 40]] = [[30, 80], [30, 80]]
        let a: Tensor<Flex, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Flex, 2, Int> =
            Tensor::from_data([[10i64, 20], [30, 40]], &Default::default());

        let a = a.flip([0]);

        let result = a * b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([[30i64, 80], [30, 80]]));
    }

    #[test]
    fn test_int_add_scalar_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // [4, 3, 2, 1] + 10 = [14, 13, 12, 11]
        let a: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3, 4], &Default::default());
        let a = a.flip([0]);

        let result = a + 10;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([14i64, 13, 12, 11]));
    }

    #[test]
    fn test_int_into_float_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Convert to float: [4.0, 3.0, 2.0, 1.0]
        let t: Tensor<Flex, 1, Int> = Tensor::from_data([1i64, 2, 3, 4], &Default::default());
        let t = t.flip([0]);
        let float_t: Tensor<Flex, 1> = t.float();
        let data: Vec<f32> = float_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![4.0f32, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_int_mul_flipped_both_axes() {
        // [[1, 2], [3, 4]] flipped on both axes -> [[4, 3], [2, 1]]
        // [[5, 5], [5, 5]]
        // [[4, 3], [2, 1]] * [[5, 5], [5, 5]] = [[20, 15], [10, 5]]
        let a: Tensor<Flex, 2, Int> = Tensor::from_data([[1i64, 2], [3, 4]], &Default::default());
        let b: Tensor<Flex, 2, Int> = Tensor::from_data([[5i64, 5], [5, 5]], &Default::default());

        let a = a.flip([0, 1]);

        let result = a * b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([[20i64, 15], [10, 5]]));
    }

    #[test]
    fn test_u64_div_large_values() {
        let a = FlexTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Flex::int_div(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX / 2);
    }

    #[test]
    fn test_u64_remainder_large_values() {
        let a = FlexTensor::from_data(TensorData::new(vec![u64::MAX], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![2u64], [1]));
        let result = Flex::int_remainder(a, b);
        let values: Vec<u64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], u64::MAX % 2);
    }

    #[test]
    fn test_int_abs_min_value() {
        // i64::MIN.abs() panics in debug; wrapping_abs returns MIN (matches PyTorch)
        let a = FlexTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Flex::int_abs(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_abs());
    }

    #[test]
    fn test_int_neg_min_value() {
        // i64::MIN negation panics in debug; wrapping_neg returns MIN (matches PyTorch)
        let a = FlexTensor::from_data(TensorData::new(vec![i64::MIN], [1]));
        let result = Flex::int_neg(a);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], i64::MIN.wrapping_neg());
    }

    #[test]
    fn test_int_shift_large_amount() {
        // Shift by >= bit width panics without wrapping; should not crash
        let a = FlexTensor::from_data(TensorData::new(vec![1i64], [1]));
        let b = FlexTensor::from_data(TensorData::new(vec![64i64], [1]));
        let _left = Flex::bitwise_left_shift(a.clone(), b.clone());
        let _right = Flex::bitwise_right_shift(a, b);
    }
}

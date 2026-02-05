//! Float tensor operations for the Ember backend.

use burn_backend::{
    DType, Distribution, ExecutionError, FloatDType, Scalar, TensorData,
    ops::FloatTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, Shape, Slice, bf16, f16};

use crate::Layout;
use num_traits::ToPrimitive;

use crate::ops::binary::{binary_op, scalar_op};
use crate::ops::matmul;
use crate::ops::unary;
use crate::{Ember, EmberTensor};

impl FloatTensorOps<Ember> for Ember {
    fn float_from_data(data: TensorData, _device: &Device<Ember>) -> FloatTensor<Ember> {
        EmberTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Ember>,
    ) -> FloatTensor<Ember> {
        let mut seed = crate::backend::SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            crate::backend::get_seeded_rng()
        };
        let data = TensorData::random::<f32, _, _>(shape, distribution, &mut rng);
        *seed = Some(rng);
        EmberTensor::from_data(data)
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

    fn float_into_int(tensor: FloatTensor<Ember>) -> IntTensor<Ember> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let dtype = tensor.dtype();

        let int_data: Vec<i64> = match dtype {
            DType::F32 => tensor.storage::<f32>().iter().map(|x| *x as i64).collect(),
            DType::F64 => tensor.storage::<f64>().iter().map(|x| *x as i64).collect(),
            DType::F16 => tensor
                .storage::<f16>()
                .iter()
                .map(|x| f32::from(*x) as i64)
                .collect(),
            DType::BF16 => tensor
                .storage::<bf16>()
                .iter()
                .map(|x| f32::from(*x) as i64)
                .collect(),
            _ => panic!("float_into_int: unsupported dtype {:?}", dtype),
        };

        EmberTensor::new(
            Bytes::from_elems(int_data),
            Layout::contiguous(shape),
            DType::I64,
        )
    }

    fn float_empty(shape: Shape, _device: &Device<Ember>, dtype: FloatDType) -> FloatTensor<Ember> {
        EmberTensor::empty(shape, dtype.into())
    }

    fn float_add(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a, b| a + b, |a, b| a + b)
    }

    fn float_add_scalar(lhs: FloatTensor<Ember>, rhs: Scalar) -> FloatTensor<Ember> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a + b, |a, b| a + b)
    }

    fn float_sub(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a, b| a - b, |a, b| a - b)
    }

    fn float_sub_scalar(lhs: FloatTensor<Ember>, rhs: Scalar) -> FloatTensor<Ember> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a - b, |a, b| a - b)
    }

    fn float_mul(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a, b| a * b, |a, b| a * b)
    }

    fn float_mul_scalar(lhs: FloatTensor<Ember>, rhs: Scalar) -> FloatTensor<Ember> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a * b, |a, b| a * b)
    }

    fn float_div(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a, b| a / b, |a, b| a / b)
    }

    fn float_div_scalar(lhs: FloatTensor<Ember>, rhs: Scalar) -> FloatTensor<Ember> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a / b, |a, b| a / b)
    }

    fn float_remainder(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        // Python/PyTorch-style remainder: result has same sign as divisor
        binary_op(lhs, rhs, |a, b| ((a % b) + b) % b, |a, b| ((a % b) + b) % b)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Ember>, rhs: Scalar) -> FloatTensor<Ember> {
        let rhs_val = rhs.to_f64().unwrap();
        // Python/PyTorch-style remainder: result has same sign as divisor
        scalar_op(
            lhs,
            rhs_val,
            |a, b| ((a % b) + b) % b,
            |a, b| ((a % b) + b) % b,
        )
    }

    fn float_matmul(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        matmul::matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<Ember>,
        rhs: FloatTensor<Ember>,
        dim: usize,
    ) -> FloatTensor<Ember> {
        let shape = lhs.layout().shape();
        let ndims = shape.num_dims();
        assert_eq!(
            shape.dims[dim], 3,
            "cross product requires dimension {} to have size 3, got {}",
            dim, shape.dims[dim]
        );

        // Helper to create slices that select index `idx` along `dim`
        let make_slices = |idx: usize| -> alloc::vec::Vec<Slice> {
            (0..ndims)
                .map(|d| {
                    if d == dim {
                        Slice::new(idx as isize, Some((idx + 1) as isize), 1)
                    } else {
                        Slice::new(0, None, 1)
                    }
                })
                .collect()
        };

        // Extract components along the dimension
        // a = [a0, a1, a2], b = [b0, b1, b2]
        let a0 = Self::float_slice(lhs.clone(), &make_slices(0));
        let a1 = Self::float_slice(lhs.clone(), &make_slices(1));
        let a2 = Self::float_slice(lhs, &make_slices(2));

        let b0 = Self::float_slice(rhs.clone(), &make_slices(0));
        let b1 = Self::float_slice(rhs.clone(), &make_slices(1));
        let b2 = Self::float_slice(rhs, &make_slices(2));

        // Cross product: c = a × b
        // c0 = a1*b2 - a2*b1
        // c1 = a2*b0 - a0*b2
        // c2 = a0*b1 - a1*b0
        let c0 = Self::float_sub(
            Self::float_mul(a1.clone(), b2.clone()),
            Self::float_mul(a2.clone(), b1.clone()),
        );
        let c1 = Self::float_sub(
            Self::float_mul(a2, b0.clone()),
            Self::float_mul(a0.clone(), b2),
        );
        let c2 = Self::float_sub(Self::float_mul(a0, b1), Self::float_mul(a1, b0));

        // Concatenate along the dimension
        Self::float_cat(vec![c0, c1, c2], dim)
    }

    fn float_recip(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::recip(tensor)
    }

    fn float_swap_dims(tensor: FloatTensor<Ember>, dim1: usize, dim2: usize) -> FloatTensor<Ember> {
        tensor.transpose(dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Ember>, axes: &[usize]) -> FloatTensor<Ember> {
        tensor.permute(axes)
    }

    fn float_flip(tensor: FloatTensor<Ember>, axes: &[usize]) -> FloatTensor<Ember> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn float_reshape(tensor: FloatTensor<Ember>, shape: Shape) -> FloatTensor<Ember> {
        tensor.reshape(shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Ember>,
        indices: IntTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::gather_f32(tensor, dim, indices),
            DType::F64 => crate::ops::gather_scatter::gather_f64(tensor, dim, indices),
            _ => panic!("float_gather: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Ember>,
        indices: IntTensor<Ember>,
        value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::scatter_add_f32(tensor, dim, indices, value),
            DType::F64 => crate::ops::gather_scatter::scatter_add_f64(tensor, dim, indices, value),
            _ => panic!("float_scatter_add: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_select(
        tensor: FloatTensor<Ember>,
        dim: usize,
        indices: IntTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::select_f32(tensor, dim, indices),
            DType::F64 => crate::ops::gather_scatter::select_f64(tensor, dim, indices),
            _ => panic!("float_select: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_select_add(
        tensor: FloatTensor<Ember>,
        dim: usize,
        indices: IntTensor<Ember>,
        value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::select_add_f32(tensor, dim, indices, value),
            DType::F64 => crate::ops::gather_scatter::select_add_f64(tensor, dim, indices, value),
            _ => panic!("float_select_add: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_slice(tensor: FloatTensor<Ember>, slices: &[Slice]) -> FloatTensor<Ember> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Ember>,
        slices: &[Slice],
        value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Ember>,
        mask: BoolTensor<Ember>,
        value: FloatTensor<Ember>,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::mask::mask_where_f32(tensor, mask, value),
            DType::F64 => crate::ops::mask::mask_where_f64(tensor, mask, value),
            DType::F16 => crate::ops::mask::mask_where_f16(tensor, mask, value),
            DType::BF16 => crate::ops::mask::mask_where_bf16(tensor, mask, value),
            dtype => panic!("float_mask_where: unsupported dtype {:?}", dtype),
        }
    }

    fn float_mask_fill(
        tensor: FloatTensor<Ember>,
        mask: BoolTensor<Ember>,
        value: Scalar,
    ) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::mask::mask_fill_f32(tensor, mask, value.to_f32().unwrap()),
            DType::F64 => crate::ops::mask::mask_fill_f64(tensor, mask, value.to_f64().unwrap()),
            DType::F16 => crate::ops::mask::mask_fill_f16(
                tensor,
                mask,
                f16::from_f64(value.to_f64().unwrap()),
            ),
            DType::BF16 => crate::ops::mask::mask_fill_bf16(
                tensor,
                mask,
                bf16::from_f64(value.to_f64().unwrap()),
            ),
            dtype => panic!("float_mask_fill: unsupported dtype {:?}", dtype),
        }
    }

    fn float_equal(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::equal(lhs, rhs)
    }

    fn float_equal_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::equal_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_greater(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::greater(lhs, rhs)
    }

    fn float_greater_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::greater_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_greater_equal(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::greater_equal_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_lower(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::lower(lhs, rhs)
    }

    fn float_lower_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::lower_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_lower_equal(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::lower_equal_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_not_equal(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> BoolTensor<Ember> {
        crate::ops::comparison::not_equal(lhs, rhs)
    }

    fn float_not_equal_elem(lhs: FloatTensor<Ember>, rhs: Scalar) -> BoolTensor<Ember> {
        crate::ops::comparison::not_equal_elem(lhs, rhs.to_f64().unwrap())
    }

    fn float_sum(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        crate::ops::reduce::sum(tensor)
    }

    fn float_sum_dim(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn float_prod(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        crate::ops::reduce::prod(tensor)
    }

    fn float_prod_dim(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn float_cumsum(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cumsum_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cumsum_f64(tensor, dim),
            _ => panic!("float_cumsum: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cumprod(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cumprod_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cumprod_f64(tensor, dim),
            _ => panic!("float_cumprod: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cummin(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cummin_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cummin_f64(tensor, dim),
            _ => panic!("float_cummin: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cummax(tensor: FloatTensor<Ember>, dim: usize) -> FloatTensor<Ember> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cummax_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cummax_f64(tensor, dim),
            _ => panic!("float_cummax: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cast(tensor: FloatTensor<Ember>, dtype: FloatDType) -> FloatTensor<Ember> {
        use crate::Layout;
        use burn_std::{Bytes, bf16, f16};

        let src_dtype = tensor.dtype();
        let target_dtype = DType::from(dtype);

        // No-op if already the same dtype
        if src_dtype == target_dtype {
            return tensor;
        }

        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();

        // Convert to f64 intermediate, then to target
        let f64_values: Vec<f64> = match src_dtype {
            DType::F32 => {
                let src: &[f32] = tensor.storage();
                src.iter().map(|&v| v as f64).collect()
            }
            DType::F64 => {
                let src: &[f64] = tensor.storage();
                src.to_vec()
            }
            DType::F16 => {
                let src: &[f16] = tensor.storage();
                src.iter().map(|&v| v.to_f32() as f64).collect()
            }
            DType::BF16 => {
                let src: &[bf16] = tensor.storage();
                src.iter().map(|&v| v.to_f32() as f64).collect()
            }
            _ => panic!("float_cast: unsupported source dtype {:?}", src_dtype),
        };

        // Convert from f64 to target dtype
        match target_dtype {
            DType::F32 => {
                let result: Vec<f32> = f64_values.iter().map(|&v| v as f32).collect();
                let bytes = Bytes::from_elems(result);
                EmberTensor::new(bytes, Layout::contiguous(shape), DType::F32)
            }
            DType::F64 => {
                let bytes = Bytes::from_elems(f64_values);
                EmberTensor::new(bytes, Layout::contiguous(shape), DType::F64)
            }
            DType::F16 => {
                let result: Vec<f16> = f64_values.iter().map(|&v| f16::from_f64(v)).collect();
                let bytes = Bytes::from_elems(result);
                EmberTensor::new(bytes, Layout::contiguous(shape), DType::F16)
            }
            DType::BF16 => {
                let result: Vec<bf16> = f64_values.iter().map(|&v| bf16::from_f64(v)).collect();
                let bytes = Bytes::from_elems(result);
                EmberTensor::new(bytes, Layout::contiguous(shape), DType::BF16)
            }
            _ => panic!("float_cast: unsupported target dtype {:?}", target_dtype),
        }
    }

    fn float_exp(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::exp(tensor)
    }

    fn float_log(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::log1p(tensor)
    }

    fn float_powf(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b))
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Ember>, value: Scalar) -> FloatTensor<Ember> {
        let exp = value.to_f64().unwrap();
        scalar_op(tensor, exp, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b))
    }

    fn float_sqrt(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::sin(tensor)
    }

    fn float_tan(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::tan(tensor)
    }

    fn float_cosh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::cosh(tensor)
    }

    fn float_sinh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::sinh(tensor)
    }

    fn float_tanh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::tanh(tensor)
    }

    fn float_acos(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::acos(tensor)
    }

    fn float_acosh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::acosh(tensor)
    }

    fn float_asin(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::asin(tensor)
    }

    fn float_asinh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::asinh(tensor)
    }

    fn float_atan(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::atan(tensor)
    }

    fn float_atanh(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::atanh(tensor)
    }

    fn float_atan2(lhs: FloatTensor<Ember>, rhs: FloatTensor<Ember>) -> FloatTensor<Ember> {
        binary_op(lhs, rhs, |a: f32, b| a.atan2(b), |a: f64, b| a.atan2(b))
    }

    fn float_round(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::round(tensor)
    }

    fn float_floor(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::ceil(tensor)
    }

    fn float_trunc(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::trunc(tensor)
    }

    fn float_erf(tensor: FloatTensor<Ember>) -> FloatTensor<Ember> {
        unary::erf(tensor)
    }

    fn float_argmax(tensor: FloatTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::argmax(tensor, dim)
    }

    fn float_argmin(tensor: FloatTensor<Ember>, dim: usize) -> IntTensor<Ember> {
        crate::ops::reduce::argmin(tensor, dim)
    }

    fn float_expand(tensor: FloatTensor<Ember>, shape: Shape) -> FloatTensor<Ember> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<Ember>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Ember> {
        // unfold is now type-agnostic (zero-copy strided view)
        crate::ops::unfold::unfold(tensor, dim, size, step)
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::{Tensor, TensorData};

    use crate::Ember;

    #[test]
    fn test_add_tensors() {
        let a: Tensor<Ember, 2> =
            Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &Default::default());
        let b: Tensor<Ember, 2> =
            Tensor::from_data([[5.0f32, 6.0], [7.0, 8.0]], &Default::default());

        let result = a + b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([[6.0f32, 8.0], [10.0, 12.0]]));
    }

    #[test]
    fn test_sub_tensors() {
        let a: Tensor<Ember, 1> = Tensor::from_data([10.0f32, 20.0, 30.0], &Default::default());
        let b: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());

        let result = a - b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([9.0f32, 18.0, 27.0]));
    }

    #[test]
    fn test_mul_tensors() {
        let a: Tensor<Ember, 1> = Tensor::from_data([2.0f32, 3.0, 4.0], &Default::default());
        let b: Tensor<Ember, 1> = Tensor::from_data([5.0f32, 6.0, 7.0], &Default::default());

        let result = a * b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([10.0f32, 18.0, 28.0]));
    }

    #[test]
    fn test_div_tensors() {
        let a: Tensor<Ember, 1> = Tensor::from_data([10.0f32, 20.0, 30.0], &Default::default());
        let b: Tensor<Ember, 1> = Tensor::from_data([2.0f32, 4.0, 5.0], &Default::default());

        let result = a / b;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([5.0f32, 5.0, 6.0]));
    }

    #[test]
    fn test_add_scalar() {
        let a: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());
        let result = a + 10.0;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([11.0f32, 12.0, 13.0]));
    }

    #[test]
    fn test_mul_scalar() {
        let a: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());
        let result = a * 3.0;
        let data = result.into_data();

        assert_eq!(data, TensorData::from([3.0f32, 6.0, 9.0]));
    }

    fn assert_approx(result: TensorData, expected: &[f32], tol: f32) {
        let actual: Vec<f32> = result.to_vec().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < tol, "got {}, expected {}", a, e);
        }
    }

    #[test]
    fn test_exp_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([0.0f32, 1.0, 2.0], &Default::default());
        let result = t.exp().into_data();
        assert_approx(
            result,
            &[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            1e-5,
        );
    }

    #[test]
    fn test_log_tensor_api() {
        let t: Tensor<Ember, 1> =
            Tensor::from_data([1.0f32, std::f32::consts::E], &Default::default());
        let result = t.log().into_data();
        assert_approx(result, &[0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_sqrt_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 4.0, 9.0], &Default::default());
        let result = t.sqrt().into_data();
        assert_approx(result, &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_abs_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([-3.0f32, -1.0, 0.0, 1.0], &Default::default());
        let result = t.abs().into_data();
        assert_approx(result, &[3.0, 1.0, 0.0, 1.0], 1e-5);
    }

    #[test]
    fn test_sin_cos_tensor_api() {
        let t: Tensor<Ember, 1> =
            Tensor::from_data([0.0f32, std::f32::consts::FRAC_PI_2], &Default::default());
        let sin_result = t.clone().sin().into_data();
        let cos_result = t.cos().into_data();
        assert_approx(sin_result, &[0.0, 1.0], 1e-5);
        assert_approx(cos_result, &[1.0, 0.0], 1e-5);
    }

    #[test]
    fn test_tanh_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([0.0f32, 1.0], &Default::default());
        let result = t.tanh().into_data();
        assert_approx(result, &[0.0, 1.0f32.tanh()], 1e-5);
    }

    #[test]
    fn test_recip_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 2.0, 4.0], &Default::default());
        let result = t.recip().into_data();
        assert_approx(result, &[1.0, 0.5, 0.25], 1e-5);
    }

    #[test]
    fn test_floor_ceil_round_tensor_api() {
        let t: Tensor<Ember, 1> = Tensor::from_data([1.3f32, 1.7, -1.3, -1.7], &Default::default());
        let floor_result = t.clone().floor().into_data();
        let ceil_result = t.clone().ceil().into_data();
        let round_result = t.round().into_data();
        assert_approx(floor_result, &[1.0, 1.0, -2.0, -2.0], 1e-5);
        assert_approx(ceil_result, &[2.0, 2.0, -1.0, -1.0], 1e-5);
        assert_approx(round_result, &[1.0, 2.0, -1.0, -2.0], 1e-5);
    }

    #[test]
    fn test_random_uniform() {
        use burn_tensor::Distribution;

        let device = Default::default();
        let t: Tensor<Ember, 2> =
            Tensor::random([10, 10], Distribution::Uniform(0.0, 1.0), &device);
        let data = t.into_data();
        let values: Vec<f32> = data.to_vec().unwrap();

        // All values should be in [0, 1]
        for v in &values {
            assert!(*v >= 0.0 && *v <= 1.0, "value {} out of range [0, 1]", v);
        }
    }

    #[test]
    fn test_float_into_int() {
        use burn_tensor::Int;

        let t: Tensor<Ember, 1> = Tensor::from_data([1.5f32, 2.7, -3.9, 0.0], &Default::default());
        let int_t: Tensor<Ember, 1, Int> = t.int();
        let data: Vec<i64> = int_t.into_data().to_vec().unwrap();

        // Truncation towards zero
        assert_eq!(data, vec![1i64, 2, -3, 0]);
    }

    #[test]
    fn test_float_into_int_2d() {
        use burn_tensor::Int;

        let t: Tensor<Ember, 2> =
            Tensor::from_data([[1.1f32, 2.9], [3.5, 4.0]], &Default::default());
        let int_t: Tensor<Ember, 2, Int> = t.int();
        let data: Vec<i64> = int_t.into_data().to_vec().unwrap();

        assert_eq!(data, vec![1i64, 2, 3, 4]);
    }

    #[test]
    fn test_cross_product() {
        // Cross product of [1, 0, 0] x [0, 1, 0] = [0, 0, 1]
        let a: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 0.0, 0.0], &Default::default());
        let b: Tensor<Ember, 1> = Tensor::from_data([0.0f32, 1.0, 0.0], &Default::default());
        let result = a.cross(b, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0]);

        // Cross product of [1, 2, 3] x [4, 5, 6] = [-3, 6, -3]
        let a: Tensor<Ember, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());
        let b: Tensor<Ember, 1> = Tensor::from_data([4.0f32, 5.0, 6.0], &Default::default());
        let result = a.cross(b, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![-3.0, 6.0, -3.0]);
    }

    #[test]
    fn test_cross_product_2d() {
        // Batched cross product: [[1,0,0], [0,1,0]] x [[0,1,0], [0,0,1]]
        // = [[0,0,1], [1,0,0]]
        let a: Tensor<Ember, 2> =
            Tensor::from_data([[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], &Default::default());
        let b: Tensor<Ember, 2> =
            Tensor::from_data([[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0]], &Default::default());
        let result = a.cross(b, 1); // dim=1 is the 3-element dimension
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unfold() {
        // Input: [1, 2, 3, 4, 5] shape [5]
        // Unfold dim=0, size=3, step=1
        // Output shape: [3, 3]
        let t: Tensor<Ember, 1> =
            Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &Default::default());
        let result: Tensor<Ember, 2> = t.unfold(0, 3, 1);
        assert_eq!(result.shape().dims, [3, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }
}

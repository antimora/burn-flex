//! Reduction operations for EmberTensor.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{EmberTensor, Layout};

/// Sum all elements in a tensor, returning a scalar tensor.
pub fn sum(tensor: EmberTensor) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => sum_impl::<f32>(&tensor),
        DType::F64 => sum_impl::<f64>(&tensor),
        DType::F16 => sum_f16(&tensor),
        DType::BF16 => sum_bf16(&tensor),
        DType::I8 => sum_impl::<i8>(&tensor),
        DType::I16 => sum_impl::<i16>(&tensor),
        DType::I32 => sum_impl::<i32>(&tensor),
        DType::I64 => sum_impl::<i64>(&tensor),
        _ => panic!("sum: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn sum_impl<E: Element + bytemuck::Pod + Default + core::iter::Sum>(
    tensor: &EmberTensor,
) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[E] = tensor.storage();
    let result: E = data.iter().copied().sum();

    let bytes = Bytes::from_elems(vec![result]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), tensor.dtype())
}

fn sum_f16(tensor: &EmberTensor) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[f16] = tensor.storage();
    // Accumulate in f32 for precision
    let result: f32 = data.iter().map(|x| x.to_f32()).sum();

    let bytes = Bytes::from_elems(vec![f16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F16)
}

fn sum_bf16(tensor: &EmberTensor) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    // Accumulate in f32 for precision
    let result: f32 = data.iter().map(|x| x.to_f32()).sum();

    let bytes = Bytes::from_elems(vec![bf16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::BF16)
}

/// Sum along a dimension, keeping the dimension with size 1.
pub fn sum_dim(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_impl::<f32, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F16 => reduce_dim_f16(&tensor, dim, |acc, x| acc + x),
        DType::BF16 => reduce_dim_bf16(&tensor, dim, |acc, x| acc + x),
        DType::I8 => reduce_dim_impl::<i8, _>(&tensor, dim, 0, |acc, x| acc + x),
        DType::I16 => reduce_dim_impl::<i16, _>(&tensor, dim, 0, |acc, x| acc + x),
        DType::I32 => reduce_dim_impl::<i32, _>(&tensor, dim, 0, |acc, x| acc + x),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 0, |acc, x| acc + x),
        _ => panic!("sum_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Mean along a dimension, keeping the dimension with size 1.
pub fn mean_dim(tensor: EmberTensor, dim: usize) -> EmberTensor {
    let dim_size = tensor.layout().shape().dims[dim];
    let dtype = tensor.dtype();
    let sum_result = sum_dim(tensor, dim);

    // Divide by dimension size
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, dim_size as f32),
        DType::F64 => scalar_div::<f64>(sum_result, dim_size as f64),
        DType::F16 => scalar_div_f16(sum_result, dim_size as f32),
        DType::BF16 => scalar_div_bf16(sum_result, dim_size as f32),
        DType::I8 => scalar_div_int::<i8>(sum_result, dim_size as i8),
        DType::I16 => scalar_div_int::<i16>(sum_result, dim_size as i16),
        DType::I32 => scalar_div_int::<i32>(sum_result, dim_size as i32),
        DType::I64 => scalar_div_int::<i64>(sum_result, dim_size as i64),
        _ => panic!("mean_dim: unsupported dtype {:?}", dtype),
    }
}

/// Product of all elements in a tensor, returning a scalar tensor.
pub fn prod(tensor: EmberTensor) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => prod_impl::<f32>(&tensor),
        DType::F64 => prod_impl::<f64>(&tensor),
        DType::F16 => prod_f16(&tensor),
        DType::BF16 => prod_bf16(&tensor),
        DType::I8 => prod_impl::<i8>(&tensor),
        DType::I16 => prod_impl::<i16>(&tensor),
        DType::I32 => prod_impl::<i32>(&tensor),
        DType::I64 => prod_impl::<i64>(&tensor),
        _ => panic!("prod: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn prod_impl<E: Element + bytemuck::Pod + Default + core::iter::Product>(
    tensor: &EmberTensor,
) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[E] = tensor.storage();
    let result: E = data.iter().copied().product();

    let bytes = Bytes::from_elems(vec![result]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), tensor.dtype())
}

fn prod_f16(tensor: &EmberTensor) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[f16] = tensor.storage();
    let result: f32 = data.iter().map(|x| x.to_f32()).product();

    let bytes = Bytes::from_elems(vec![f16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F16)
}

fn prod_bf16(tensor: &EmberTensor) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let result: f32 = data.iter().map(|x| x.to_f32()).product();

    let bytes = Bytes::from_elems(vec![bf16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::BF16)
}

/// Product along a dimension, keeping the dimension with size 1.
pub fn prod_dim(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_impl::<f32, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F16 => reduce_dim_f16(&tensor, dim, |acc, x| acc * x),
        DType::BF16 => reduce_dim_bf16(&tensor, dim, |acc, x| acc * x),
        DType::I8 => reduce_dim_impl::<i8, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I16 => reduce_dim_impl::<i16, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I32 => reduce_dim_impl::<i32, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 1, |acc, x| acc * x),
        _ => panic!("prod_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Argmax along a dimension, returning indices as i64.
pub fn argmax(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => argmax_impl::<f32>(&tensor, dim),
        DType::F64 => argmax_impl::<f64>(&tensor, dim),
        DType::F16 => argmax_f16(&tensor, dim),
        DType::BF16 => argmax_bf16(&tensor, dim),
        DType::I8 => argmax_impl::<i8>(&tensor, dim),
        DType::I16 => argmax_impl::<i16>(&tensor, dim),
        DType::I32 => argmax_impl::<i32>(&tensor, dim),
        DType::I64 => argmax_impl::<i64>(&tensor, dim),
        _ => panic!("argmax: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Argmin along a dimension, returning indices as i64.
pub fn argmin(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => argmin_impl::<f32>(&tensor, dim),
        DType::F64 => argmin_impl::<f64>(&tensor, dim),
        DType::F16 => argmin_f16(&tensor, dim),
        DType::BF16 => argmin_bf16(&tensor, dim),
        DType::I8 => argmin_impl::<i8>(&tensor, dim),
        DType::I16 => argmin_impl::<i16>(&tensor, dim),
        DType::I32 => argmin_impl::<i32>(&tensor, dim),
        DType::I64 => argmin_impl::<i64>(&tensor, dim),
        _ => panic!("argmin: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic dimension reduction implementation.
fn reduce_dim_impl<E, F>(tensor: &EmberTensor, dim: usize, init: E, reduce_fn: F) -> EmberTensor
where
    E: Element + bytemuck::Pod + Copy,
    F: Fn(E, E) -> E,
{
    let tensor = tensor.to_contiguous();
    let src: &[E] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);

    let dim_size = shape.dims[dim];

    // Calculate output shape (same as input but with dim reduced to 1)
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // For each output position, reduce along the dimension
    let mut result: Vec<E> = Vec::with_capacity(out_size);

    // Compute outer and inner sizes
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = init;
            for d in 0..dim_size {
                // Use flat indexing for contiguous tensor
                let base = if dim == 0 {
                    0
                } else {
                    outer * (dim_size * inner_size)
                };
                let idx = base + d * inner_size + inner;

                if idx < src.len() {
                    acc = reduce_fn(acc, src[idx]);
                }
            }
            result.push(acc);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), tensor.dtype())
}

/// F16 dimension reduction with f32 accumulation.
fn reduce_dim_f16<F>(tensor: &EmberTensor, dim: usize, reduce_fn: F) -> EmberTensor
where
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let src: &[f16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<f16> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = if reduce_fn(1.0, 1.0) == 2.0 { 0.0f32 } else { 1.0f32 }; // sum vs prod
            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    acc = reduce_fn(acc, src[idx].to_f32());
                }
            }
            result.push(f16::from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::F16)
}

/// BF16 dimension reduction with f32 accumulation.
fn reduce_dim_bf16<F>(tensor: &EmberTensor, dim: usize, reduce_fn: F) -> EmberTensor
where
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let src: &[bf16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<bf16> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = if reduce_fn(1.0, 1.0) == 2.0 { 0.0f32 } else { 1.0f32 };
            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    acc = reduce_fn(acc, src[idx].to_f32());
                }
            }
            result.push(bf16::from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::BF16)
}

/// Scalar division helper for mean.
fn scalar_div<E: Element + bytemuck::Pod + core::ops::Div<Output = E> + Copy>(
    mut tensor: EmberTensor,
    divisor: E,
) -> EmberTensor {
    let data: &mut [E] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = *x / divisor;
    }
    tensor
}

fn scalar_div_f16(mut tensor: EmberTensor, divisor: f32) -> EmberTensor {
    let data: &mut [f16] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = f16::from_f32(x.to_f32() / divisor);
    }
    tensor
}

fn scalar_div_bf16(mut tensor: EmberTensor, divisor: f32) -> EmberTensor {
    let data: &mut [bf16] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = bf16::from_f32(x.to_f32() / divisor);
    }
    tensor
}

fn scalar_div_int<E: Element + bytemuck::Pod + core::ops::Div<Output = E> + Copy>(
    mut tensor: EmberTensor,
    divisor: E,
) -> EmberTensor {
    let data: &mut [E] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = *x / divisor;
    }
    tensor
}

/// Argmax implementation.
fn argmax_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[E] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_idx: i64 = 0;
            let mut max_val: Option<E> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx];
                    if max_val.is_none() || val > max_val.unwrap() {
                        max_val = Some(val);
                        max_idx = d as i64;
                    }
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

fn argmax_f16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[f16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_idx: i64 = 0;
            let mut max_val: Option<f32> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx].to_f32();
                    if max_val.is_none() || val > max_val.unwrap() {
                        max_val = Some(val);
                        max_idx = d as i64;
                    }
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

fn argmax_bf16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[bf16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_idx: i64 = 0;
            let mut max_val: Option<f32> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx].to_f32();
                    if max_val.is_none() || val > max_val.unwrap() {
                        max_val = Some(val);
                        max_idx = d as i64;
                    }
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

/// Argmin implementation.
fn argmin_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[E] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims, "dim {} out of bounds for {} dimensions", dim, ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut min_idx: i64 = 0;
            let mut min_val: Option<E> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx];
                    if min_val.is_none() || val < min_val.unwrap() {
                        min_val = Some(val);
                        min_idx = d as i64;
                    }
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

fn argmin_f16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[f16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut min_idx: i64 = 0;
            let mut min_val: Option<f32> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx].to_f32();
                    if min_val.is_none() || val < min_val.unwrap() {
                        min_val = Some(val);
                        min_idx = d as i64;
                    }
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

fn argmin_bf16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let src: &[bf16] = tensor.storage();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut min_idx: i64 = 0;
            let mut min_val: Option<f32> = None;

            for d in 0..dim_size {
                let base = if dim == 0 { 0 } else { outer * (dim_size * inner_size) };
                let idx = base + d * inner_size + inner;
                if idx < src.len() {
                    let val = src[idx].to_f32();
                    if min_val.is_none() || val < min_val.unwrap() {
                        min_val = Some(val);
                        min_idx = d as i64;
                    }
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(out_shape)), DType::I64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_sum_1d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let result = sum(tensor);

        assert_eq!(result.layout().shape().dims, vec![1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn test_sum_2d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum(tensor);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![21.0]);
    }

    #[test]
    fn test_sum_dim_2d_dim0() {
        // [[1, 2, 3], [4, 5, 6]] -> sum along dim 0 -> [[5, 7, 9]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum_dim(tensor, 0);

        assert_eq!(result.layout().shape().dims, vec![1, 3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_dim_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 -> [[6], [15]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_dim() {
        // [[1, 2, 3], [4, 5, 6]] -> mean along dim 1 -> [[2], [5]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = mean_dim(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.0, 5.0]);
    }

    #[test]
    fn test_argmax_1d() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let result = argmax(tensor, 0);

        assert_eq!(result.layout().shape().dims, vec![1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1]); // index of 5.0
    }

    #[test]
    fn test_argmax_2d_dim1() {
        // [[1, 5, 3], [6, 2, 4]] -> argmax along dim 1 -> [[1], [0]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 6.0, 2.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = argmax(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1, 0]); // indices of max in each row
    }

    #[test]
    fn test_argmin_2d_dim1() {
        // [[1, 5, 3], [6, 2, 4]] -> argmin along dim 1 -> [[0], [1]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 6.0, 2.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = argmin(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0, 1]); // indices of min in each row
    }

    #[test]
    fn test_prod() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [4]));
        let result = prod(tensor);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![24.0]);
    }

    #[test]
    fn test_sum_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let result = sum(tensor);

        assert_eq!(result.layout().shape().dims, vec![1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15]);
    }

    #[test]
    fn test_sum_dim_i32() {
        // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 -> [[6], [15]]
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6, 15]);
    }

    #[test]
    fn test_argmax_i32() {
        let data: Vec<i32> = vec![1, 5, 3, 2, 4];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let result = argmax(tensor, 0);

        assert_eq!(result.layout().shape().dims, vec![1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1]); // index of 5
    }
}

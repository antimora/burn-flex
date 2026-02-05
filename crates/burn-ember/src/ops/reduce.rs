//! Reduction operations for EmberTensor.
//!
//! Optimized with:
//! - Strided iteration (no copy for non-contiguous tensors)
//! - Portable SIMD via pulp (NEON, AVX2, SIMD128, scalar fallback)
//! - Rayon parallelism for large tensors

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::strided_index::StridedIter;
use crate::{EmberTensor, Layout};

#[cfg(feature = "simd")]
use crate::simd::kernels;

#[cfg(feature = "simd")]
use crate::simd::aligned;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Threshold for parallel execution (elements).
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 256 * 1024; // 256K elements

// ============================================================================
// Sum (all elements)
// ============================================================================

/// Sum all elements in a tensor, returning a scalar tensor.
pub fn sum(tensor: EmberTensor) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => sum_f32(&tensor),
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

/// Optimized f32 sum with SIMD and parallelism.
fn sum_f32(tensor: &EmberTensor) -> EmberTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            let slice = &data[start..end];
            sum_f32_contiguous(slice)
        }
        None => {
            // Non-contiguous: check if we can sum the buffer directly.
            // For transposed tensors that use all elements (no slicing),
            // the sum is the same regardless of element order.
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();

            if data.len() == elem_count {
                // Tensor uses entire buffer - sum directly (order doesn't matter for sum)
                sum_f32_contiguous(data)
            } else {
                // Sliced or partial view - must use strided iteration
                StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

/// SIMD + parallel sum for contiguous f32 slice.
#[inline]
fn sum_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "rayon")]
    if data.len() >= PARALLEL_THRESHOLD {
        return sum_f32_parallel(data);
    }

    #[cfg(feature = "simd")]
    {
        kernels::sum_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter().copied().sum()
    }
}

/// Parallel sum using rayon with SIMD per chunk.
#[cfg(feature = "rayon")]
#[inline]
fn sum_f32_parallel(data: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 64 * 1024; // 64K elements per chunk

    data.par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            #[cfg(feature = "simd")]
            {
                kernels::sum_f32(chunk)
            }
            #[cfg(not(feature = "simd"))]
            {
                chunk.iter().copied().sum::<f32>()
            }
        })
        .sum()
}

fn sum_impl<E: Element + bytemuck::Pod + Default + core::iter::Sum>(
    tensor: &EmberTensor,
) -> EmberTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().sum()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

fn sum_f16(tensor: &EmberTensor) -> EmberTensor {
    // Accumulate in f32 for precision
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f16] = tensor.storage();
            data[start..end].iter().map(|x| x.to_f32()).sum()
        }
        None => {
            let data: &[f16] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx].to_f32())
                .sum()
        }
    };

    let bytes = Bytes::from_elems(vec![f16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F16)
}

fn sum_bf16(tensor: &EmberTensor) -> EmberTensor {
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[bf16] = tensor.storage();
            data[start..end].iter().map(|x| x.to_f32()).sum()
        }
        None => {
            let data: &[bf16] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx].to_f32())
                .sum()
        }
    };

    let bytes = Bytes::from_elems(vec![bf16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::BF16)
}

// ============================================================================
// Sum along dimension
// ============================================================================

/// Sum along a dimension, keeping the dimension with size 1.
pub fn sum_dim(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Sum),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F16 => reduce_dim_f16(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::BF16 => reduce_dim_bf16(&tensor, dim, 0.0, |acc, x| acc + x),
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
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().product()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .product()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

fn prod_f16(tensor: &EmberTensor) -> EmberTensor {
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f16] = tensor.storage();
            data[start..end].iter().map(|x| x.to_f32()).product()
        }
        None => {
            let data: &[f16] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx].to_f32())
                .product()
        }
    };

    let bytes = Bytes::from_elems(vec![f16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F16)
}

fn prod_bf16(tensor: &EmberTensor) -> EmberTensor {
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[bf16] = tensor.storage();
            data[start..end].iter().map(|x| x.to_f32()).product()
        }
        None => {
            let data: &[bf16] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx].to_f32())
                .product()
        }
    };

    let bytes = Bytes::from_elems(vec![bf16::from_f32(result)]);
    EmberTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::BF16)
}

/// Product along a dimension, keeping the dimension with size 1.
pub fn prod_dim(tensor: EmberTensor, dim: usize) -> EmberTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Prod),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F16 => reduce_dim_f16(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::BF16 => reduce_dim_bf16(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::I8 => reduce_dim_impl::<i8, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I16 => reduce_dim_impl::<i16, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I32 => reduce_dim_impl::<i32, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 1, |acc, x| acc * x),
        _ => panic!("prod_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

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

// ============================================================================
// Dimension reduction helpers
// ============================================================================

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Prod,
}

/// Optimized f32 dimension reduction with SIMD.
fn reduce_dim_f32(tensor: &EmberTensor, dim: usize, op: ReduceOp) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[f32] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let dim_stride = strides[dim];

    let (init, reduce_fn): (f32, fn(f32, f32) -> f32) = match op {
        ReduceOp::Sum => (0.0, |a, b| a + b),
        ReduceOp::Prod => (1.0, |a, b| a * b),
    };

    // Check for negative strides (from flip operations) - fall back to general case
    let has_negative_strides = strides.iter().any(|&s| s < 0);

    // Check if inner dimension is contiguous (stride = 1) and no negative strides
    let inner_contiguous = !has_negative_strides && (dim + 1 >= ndims || strides[ndims - 1] == 1);

    let result: Vec<f32> = if inner_contiguous && dim == ndims - 1 {
        // Reducing last dimension with contiguous data: use SIMD
        reduce_last_dim_f32(data, start_offset, outer_size, dim_size, strides, dim, op)
    } else if dim == 0 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // First-dim reduction with contiguous inner: use cache-friendly accumulation
        reduce_first_dim_f32(data, start_offset, dim_size, inner_size, dim_stride)
    } else if dim > 0 && dim < ndims - 1 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // Middle-dim reduction (e.g., [B, M, K] reducing dim=1): cache-friendly accumulation
        let outer_stride = strides[dim - 1];
        reduce_middle_dim_f32(
            data,
            start_offset,
            outer_size,
            dim_size,
            inner_size,
            outer_stride,
            dim_stride,
        )
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) && outer_size == 1 {
        // Reduction dimension is contiguous, no outer batch (e.g., transposed 2D reducing dim=0)
        // Storage is [inner_size rows of dim_size elements each] - use sum_rows_f32
        #[cfg(feature = "simd")]
        {
            let mut result = vec![0.0f32; inner_size];
            kernels::sum_rows_f32(
                &data[start_offset..],
                &mut result,
                inner_size, // number of rows (output positions)
                dim_size,   // elements per row (to sum)
            );
            result
        }
        #[cfg(not(feature = "simd"))]
        {
            let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };
            let mut result = Vec::with_capacity(out_size);
            for inner in 0..inner_size {
                let base = (start_offset as isize + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                result.push(slice.iter().copied().sum());
            }
            result
        }
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) {
        // Reduction dimension is contiguous but with outer batches
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = (start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                #[cfg(feature = "simd")]
                let acc = kernels::sum_f32(slice);
                #[cfg(not(feature = "simd"))]
                let acc = slice.iter().copied().sum();
                result.push(acc);
            }
        }
        result
    } else {
        // General strided case
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let base = start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride;
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = (base + d as isize * dim_stride) as usize;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    };

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// Reduce middle dimension (e.g., [B, M, K] reducing dim=1) with cache-friendly iteration.
/// For each batch, iterate over rows (dim to reduce) sequentially and accumulate into columns.
#[inline]
fn reduce_middle_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize, // batch size
    dim_size: usize,   // rows to sum
    inner_size: usize, // columns (output per batch)
    outer_stride: isize,
    dim_stride: isize,
) -> Vec<f32> {
    let out_size = outer_size * inner_size;

    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(out_size);
        kernels::scatter_add_batched_f32(
            &data[start_offset..],
            &mut result,
            outer_size,
            dim_size,
            inner_size,
            outer_stride as usize,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; out_size];
        let start = start_offset as isize;
        for batch in 0..outer_size {
            let batch_start = (start + batch as isize * outer_stride) as usize;
            let out_batch_start = batch * inner_size;

            for row in 0..dim_size {
                let row_start = (batch_start as isize + row as isize * dim_stride) as usize;
                for c in 0..inner_size {
                    result[out_batch_start + c] += data[row_start + c];
                }
            }
        }
        result
    }
}

/// Reduce first dimension with cache-friendly row iteration.
/// Instead of iterating per-output (col) and gathering from rows (cache-unfriendly),
/// iterate over rows (sequential access) and scatter-accumulate into outputs.
#[inline]
fn reduce_first_dim_f32(
    data: &[f32],
    start_offset: usize,
    dim_size: usize,   // number of rows to sum
    inner_size: usize, // number of columns (output positions)
    dim_stride: isize, // stride between rows
) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(inner_size);
        kernels::scatter_add_f32(
            &data[start_offset..],
            &mut result,
            dim_size,
            inner_size,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; inner_size];
        let start = start_offset as isize;
        for row in 0..dim_size {
            let row_start = (start + row as isize * dim_stride) as usize;
            for c in 0..inner_size {
                result[c] += data[row_start + c];
            }
        }
        result
    }
}

/// Reduce last dimension with SIMD (most common case).
#[inline]
fn reduce_last_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize,
    dim_size: usize,
    strides: &[isize],
    dim: usize,
    op: ReduceOp,
) -> Vec<f32> {
    let outer_stride: isize = if dim > 0 {
        strides[dim - 1]
    } else {
        dim_size as isize
    };
    let mut result = Vec::with_capacity(outer_size.max(1));

    for outer in 0..outer_size.max(1) {
        let row_start = (start_offset as isize + outer as isize * outer_stride) as usize;
        let row = &data[row_start..row_start + dim_size];

        let val = match op {
            ReduceOp::Sum => {
                #[cfg(feature = "simd")]
                {
                    kernels::sum_f32(row)
                }
                #[cfg(not(feature = "simd"))]
                {
                    row.iter().copied().sum()
                }
            }
            ReduceOp::Prod => row.iter().copied().product(),
        };
        result.push(val);
    }

    result
}

/// Generic dimension reduction implementation.
fn reduce_dim_impl<E, F>(tensor: &EmberTensor, dim: usize, init: E, reduce_fn: F) -> EmberTensor
where
    E: Element + bytemuck::Pod + Copy,
    F: Fn(E, E) -> E,
{
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut acc = init;
            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                acc = reduce_fn(acc, data[idx]);
            }
            result.push(acc);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// F16 dimension reduction with f32 accumulation.
fn reduce_dim_f16<F>(tensor: &EmberTensor, dim: usize, init: f32, reduce_fn: F) -> EmberTensor
where
    F: Fn(f32, f32) -> f32,
{
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[f16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<f16> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut acc = init;
            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                acc = reduce_fn(acc, data[idx].to_f32());
            }
            result.push(f16::from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::F16,
    )
}

/// BF16 dimension reduction with f32 accumulation.
fn reduce_dim_bf16<F>(tensor: &EmberTensor, dim: usize, init: f32, reduce_fn: F) -> EmberTensor
where
    F: Fn(f32, f32) -> f32,
{
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[bf16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<bf16> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut acc = init;
            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                acc = reduce_fn(acc, data[idx].to_f32());
            }
            result.push(bf16::from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::BF16,
    )
}

// ============================================================================
// Scalar division helpers
// ============================================================================

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

// ============================================================================
// Argmax / Argmin implementations
// ============================================================================

fn argmax_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &EmberTensor,
    dim: usize,
) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut max_idx: i64 = 0;
            let mut max_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx];
                if max_val.is_none() || val > max_val.unwrap() {
                    max_val = Some(val);
                    max_idx = d as i64;
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmax_f16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[f16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut max_idx: i64 = 0;
            let mut max_val: Option<f32> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx].to_f32();
                if max_val.is_none() || val > max_val.unwrap() {
                    max_val = Some(val);
                    max_idx = d as i64;
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmax_bf16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[bf16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut max_idx: i64 = 0;
            let mut max_val: Option<f32> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx].to_f32();
                if max_val.is_none() || val > max_val.unwrap() {
                    max_val = Some(val);
                    max_idx = d as i64;
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmin_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &EmberTensor,
    dim: usize,
) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut min_idx: i64 = 0;
            let mut min_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx];
                if min_val.is_none() || val < min_val.unwrap() {
                    min_val = Some(val);
                    min_idx = d as i64;
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmin_f16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[f16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut min_idx: i64 = 0;
            let mut min_val: Option<f32> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx].to_f32();
                if min_val.is_none() || val < min_val.unwrap() {
                    min_val = Some(val);
                    min_idx = d as i64;
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmin_bf16(tensor: &EmberTensor, dim: usize) -> EmberTensor {
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[bf16] = tensor.storage();
    let start_offset = tensor.layout().start_offset() as isize;
    let dim_stride = strides[dim];
    let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
    let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let base = start_offset + outer as isize * outer_stride + inner as isize * inner_stride;
            let mut min_idx: i64 = 0;
            let mut min_val: Option<f32> = None;

            for d in 0..dim_size {
                let idx = (base + d as isize * dim_stride) as usize;
                let val = data[idx].to_f32();
                if min_val.is_none() || val < min_val.unwrap() {
                    min_val = Some(val);
                    min_idx = d as i64;
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

// ============================================================================
// Tests
// ============================================================================

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
    fn test_sum_transposed() {
        // Test that sum works on non-contiguous (transposed) tensor without copying
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let transposed = tensor.transpose(0, 1);

        // Transposed tensor should NOT be contiguous
        assert!(!transposed.is_contiguous());

        let result = sum(transposed);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![21.0]); // Same sum regardless of layout
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

    // === Non-contiguous / negative stride tests ===

    #[test]
    fn test_sum_flipped() {
        // [1, 2, 3, 4, 5] flipped -> [5, 4, 3, 2, 1]
        // Sum is order-independent, should still be 15
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let result = sum(flipped);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn test_sum_dim_flipped() {
        // [[1, 2, 3], [4, 5, 6]] with axis 0 flipped -> [[4, 5, 6], [1, 2, 3]]
        // sum along dim 0 -> [[5, 7, 9]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let result = sum_dim(flipped, 0);
        assert_eq!(result.layout().shape().dims, vec![1, 3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Same sum regardless of row order
        assert_eq!(values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_dim_flipped_axis1() {
        // [[1, 2, 3], [4, 5, 6]] with axis 1 flipped -> [[3, 2, 1], [6, 5, 4]]
        // sum along dim 1 -> [[6], [15]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let flipped = crate::ops::flip::flip(tensor, &[1]);
        assert!(flipped.layout().strides()[1] < 0);

        let result = sum_dim(flipped, 1);
        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6.0, 15.0]);
    }

    #[test]
    fn test_argmax_flipped() {
        // [1, 5, 3, 2, 4] flipped -> [4, 2, 3, 5, 1]
        // argmax of flipped tensor -> index 3 (where 5 is in the flipped view)
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let result = argmax(flipped, 0);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // In flipped view [4, 2, 3, 5, 1], max is 5 at index 3
        assert_eq!(values, vec![3]);
    }

    #[test]
    fn test_argmax_2d_flipped() {
        // [[1, 5, 3], [6, 2, 4]] with axis 1 flipped -> [[3, 5, 1], [4, 2, 6]]
        // argmax along dim 1 -> [[1], [2]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 6.0, 2.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let flipped = crate::ops::flip::flip(tensor, &[1]);
        assert!(flipped.layout().strides()[1] < 0);

        let result = argmax(flipped, 1);
        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Row 0: [3, 5, 1] -> max at index 1
        // Row 1: [4, 2, 6] -> max at index 2
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_argmin_flipped() {
        // [5, 1, 4, 2, 3] flipped -> [3, 2, 4, 1, 5]
        // argmin of flipped tensor -> index 3 (where 1 is)
        let data: Vec<f32> = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let result = argmin(flipped, 0);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // In flipped view [3, 2, 4, 1, 5], min is 1 at index 3
        assert_eq!(values, vec![3]);
    }

    #[test]
    fn test_mean_dim_flipped() {
        // [[1, 2, 3], [4, 5, 6]] with axis 0 flipped -> [[4, 5, 6], [1, 2, 3]]
        // mean along dim 0 -> [[2.5, 3.5, 4.5]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 3]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);

        let result = mean_dim(flipped, 0);
        assert_eq!(result.layout().shape().dims, vec![1, 3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_prod_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Product is order-independent, should be 24
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [4]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let result = prod(flipped);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![24.0]);
    }

    #[test]
    fn test_sum_narrowed() {
        // [0, 1, 2, 3, 4] narrowed to [1, 2, 3] (indices 1..4)
        // sum = 6
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [5]));
        let narrowed = tensor.narrow(0, 1, 3);

        let result = sum(narrowed);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6.0]);
    }

    #[test]
    fn test_sum_flipped_both_axes() {
        // [[1, 2], [3, 4]] flipped on both axes -> [[4, 3], [2, 1]]
        // Sum is still 10
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = EmberTensor::from_data(TensorData::new(data, [2, 2]));
        let flipped = crate::ops::flip::flip(tensor, &[0, 1]);
        assert!(flipped.layout().strides()[0] < 0);
        assert!(flipped.layout().strides()[1] < 0);

        let result = sum(flipped);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![10.0]);
    }
}

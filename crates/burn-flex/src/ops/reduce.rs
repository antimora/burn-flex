//! Reduction operations for FlexTensor.
//!
//! Optimized with:
//! - Strided iteration (no copy for non-contiguous tensors)
//! - Portable SIMD via macerator (NEON, AVX2, SIMD128, scalar fallback)
//! - Rayon parallelism for large tensors

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::strided_index::StridedIter;
use crate::{FlexTensor, Layout};

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
pub fn sum(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => sum_f32(&tensor),
        DType::F64 => sum_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => sum_impl_widening::<i8>(&tensor),
        DType::I16 => sum_impl_widening::<i16>(&tensor),
        DType::I32 => sum_impl_widening::<i32>(&tensor),
        DType::I64 => sum_impl::<i64>(&tensor),
        _ => panic!("sum: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Optimized f32 sum with SIMD and parallelism.
fn sum_f32(tensor: &FlexTensor) -> FlexTensor {
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
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
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
    tensor: &FlexTensor,
) -> FlexTensor {
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
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Widening scalar reduction for small integer types: accumulate in i64 to avoid overflow.
macro_rules! widening_scalar_reduce {
    ($name:ident, $fold:expr, $init:expr) => {
        fn $name<E>(tensor: &FlexTensor) -> FlexTensor
        where
            E: Element + bytemuck::Pod + Default,
            i64: From<E>,
        {
            let total: i64 = match tensor.layout().contiguous_offsets() {
                Some((start, end)) => {
                    let data: &[E] = tensor.storage();
                    data[start..end]
                        .iter()
                        .fold($init, |acc, x| ($fold)(acc, i64::from(*x)))
                }
                None => {
                    let data: &[E] = tensor.storage();
                    StridedIter::new(tensor.layout())
                        .fold($init, |acc, idx| ($fold)(acc, i64::from(data[idx])))
                }
            };
            // Truncate back to target type (wrapping, matches PyTorch)
            let data: &[E] = tensor.storage();
            let _ = data; // just to bind E
            let result_bytes = total.to_ne_bytes();
            // Extract lowest bytes for the target type
            let result: E =
                bytemuck::cast_slice::<u8, E>(&result_bytes[..core::mem::size_of::<E>()])[0];
            let bytes = Bytes::from_elems(vec![result]);
            FlexTensor::new(
                bytes,
                Layout::contiguous(Shape::from(vec![1])),
                tensor.dtype(),
            )
        }
    };
}

widening_scalar_reduce!(
    sum_impl_widening,
    |acc: i64, x: i64| acc.wrapping_add(x),
    0i64
);
widening_scalar_reduce!(
    prod_impl_widening,
    |acc: i64, x: i64| acc.wrapping_mul(x),
    1i64
);

/// Scalar reduction for half-precision types, accumulating in f32.
fn reduce_scalar_half<E>(
    tensor: &FlexTensor,
    fold: fn(f32, f32) -> f32,
    init: f32,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    E: Element + bytemuck::Pod,
{
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .fold(init, |acc, x| fold(acc, to_f32(*x)))
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).fold(init, |acc, idx| fold(acc, to_f32(data[idx])))
        }
    };

    let bytes = Bytes::from_elems(vec![from_f32(result)]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), E::dtype())
}

// ============================================================================
// Sum along dimension
// ============================================================================

/// Sum along a dimension, keeping the dimension with size 1.
pub fn sum_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Sum),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 0, |acc, x| acc + x),
        _ => panic!("sum_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Mean along a dimension, keeping the dimension with size 1.
pub fn mean_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    let dim_size = tensor.layout().shape().dims[dim];
    assert!(
        dim_size > 0,
        "mean_dim: cannot take mean of empty dimension"
    );
    let dtype = tensor.dtype();
    let sum_result = sum_dim(tensor, dim);

    // Divide by dimension size
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, dim_size as f32),
        DType::F64 => scalar_div::<f64>(sum_result, dim_size as f64),
        DType::F16 => scalar_div_f16(sum_result, dim_size as f32),
        DType::BF16 => scalar_div_bf16(sum_result, dim_size as f32),
        DType::I8 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i8] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i8;
            }
            tensor
        }
        DType::I16 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i16] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i16;
            }
            tensor
        }
        DType::I32 => scalar_div::<i32>(sum_result, dim_size as i32),
        DType::I64 => scalar_div::<i64>(sum_result, dim_size as i64),
        _ => panic!("mean_dim: unsupported dtype {:?}", dtype),
    }
}

/// Product of all elements in a tensor, returning a scalar tensor.
pub fn prod(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => prod_impl::<f32>(&tensor),
        DType::F64 => prod_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => prod_impl_widening::<i8>(&tensor),
        DType::I16 => prod_impl_widening::<i16>(&tensor),
        DType::I32 => prod_impl_widening::<i32>(&tensor),
        DType::I64 => prod_impl::<i64>(&tensor),
        _ => panic!("prod: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn prod_impl<E: Element + bytemuck::Pod + Default + core::iter::Product>(
    tensor: &FlexTensor,
) -> FlexTensor {
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
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Product along a dimension, keeping the dimension with size 1.
pub fn prod_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Prod),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 1, |acc, x| acc * x),
        _ => panic!("prod_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

/// Argmax along a dimension, returning indices as i64.
pub fn argmax(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => argmax_float_impl::<f32>(&tensor, dim),
        DType::F64 => argmax_float_impl::<f64>(&tensor, dim),
        DType::F16 => argext_half::<f16>(&tensor, dim, |a, b| a.is_nan() || a > b, f16::to_f32),
        DType::BF16 => argext_half::<bf16>(&tensor, dim, |a, b| a.is_nan() || a > b, bf16::to_f32),
        DType::I8 => argmax_impl::<i8>(&tensor, dim),
        DType::I16 => argmax_impl::<i16>(&tensor, dim),
        DType::I32 => argmax_impl::<i32>(&tensor, dim),
        DType::I64 => argmax_impl::<i64>(&tensor, dim),
        _ => panic!("argmax: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Argmin along a dimension, returning indices as i64.
pub fn argmin(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => argmin_float_impl::<f32>(&tensor, dim),
        DType::F64 => argmin_float_impl::<f64>(&tensor, dim),
        DType::F16 => argext_half::<f16>(&tensor, dim, |a, b| a.is_nan() || a < b, f16::to_f32),
        DType::BF16 => argext_half::<bf16>(&tensor, dim, |a, b| a.is_nan() || a < b, bf16::to_f32),
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
fn reduce_dim_f32(tensor: &FlexTensor, dim: usize, op: ReduceOp) -> FlexTensor {
    let ndims = tensor.layout().shape().num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

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
    } else if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims)
        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim (e.g., flipped 2D)
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
    FlexTensor::new(
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
fn reduce_dim_impl<E, F>(tensor: &FlexTensor, dim: usize, init: E, reduce_fn: F) -> FlexTensor
where
    E: Element + bytemuck::Pod + Copy,
    F: Fn(E, E) -> E,
{
    let ndims = tensor.layout().shape().num_dims();
    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims)
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim
        let dim_stride = strides[dim];
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

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
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Widening dimension reduction for small integer types: accumulate in i64 to avoid overflow.
fn reduce_dim_widening<E, F>(tensor: &FlexTensor, dim: usize, init: i64, reduce_fn: F) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    i64: From<E>,
    F: Fn(i64, i64) -> i64,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, i64::from(data[idx]));
            }
            // Truncate back to target type (wrapping, matches PyTorch)
            let acc_bytes = acc.to_ne_bytes();
            let val: E = bytemuck::cast_slice::<u8, E>(&acc_bytes[..core::mem::size_of::<E>()])[0];
            result.push(val);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Half-precision dimension reduction with f32 accumulation.
///
/// Works for both f16 and bf16 via the `to_f32`/`from_f32` closures.
fn reduce_dim_half<E, F>(
    tensor: &FlexTensor,
    dim: usize,
    init: f32,
    reduce_fn: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, to_f32(data[idx]));
            }
            result.push(from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

// ============================================================================
// Mean (all elements)
// ============================================================================

/// Mean of all elements, returning a scalar tensor.
pub fn mean(tensor: FlexTensor) -> FlexTensor {
    let n = tensor.layout().num_elements();
    let sum_result = sum(tensor);
    let dtype = sum_result.dtype();
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, n as f32),
        DType::F64 => scalar_div::<f64>(sum_result, n as f64),
        DType::F16 => scalar_div_f16(sum_result, n as f32),
        DType::BF16 => scalar_div_bf16(sum_result, n as f32),
        _ => panic!("mean: unsupported dtype {:?}", dtype),
    }
}

// ============================================================================
// Max/Min along dimension (value + optional indices in a single pass)
// ============================================================================

/// Max along a dimension, returning only values.
pub fn max_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert!(
        tensor.layout().shape().dims[dim] > 0,
        "max_dim: dimension {dim} has size 0"
    );
    match tensor.dtype() {
        DType::F32 => max_dim_float_impl::<f32>(&tensor, dim),
        DType::F64 => max_dim_float_impl::<f64>(&tensor, dim),
        DType::F16 => max_dim_f16(&tensor, dim, true),
        DType::BF16 => max_dim_bf16(&tensor, dim, true),
        DType::I64 => max_dim_impl::<i64>(&tensor, dim),
        DType::I32 => max_dim_impl::<i32>(&tensor, dim),
        _ => panic!("max_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Min along a dimension, returning only values.
pub fn min_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert!(
        tensor.layout().shape().dims[dim] > 0,
        "min_dim: dimension {dim} has size 0"
    );
    match tensor.dtype() {
        DType::F32 => min_dim_float_impl::<f32>(&tensor, dim),
        DType::F64 => min_dim_float_impl::<f64>(&tensor, dim),
        DType::F16 => min_dim_f16(&tensor, dim, true),
        DType::BF16 => min_dim_bf16(&tensor, dim, true),
        DType::I64 => min_dim_impl::<i64>(&tensor, dim),
        DType::I32 => min_dim_impl::<i32>(&tensor, dim),
        _ => panic!("min_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Max along a dimension with indices, returning (values, indices) in a single pass.
pub fn max_dim_with_indices(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    assert!(
        tensor.layout().shape().dims[dim] > 0,
        "max_dim_with_indices: dimension {dim} has size 0"
    );
    match tensor.dtype() {
        DType::F32 => max_dim_with_indices_float_impl::<f32>(&tensor, dim),
        DType::F64 => max_dim_with_indices_float_impl::<f64>(&tensor, dim),
        DType::F16 => {
            let values = max_dim_f16(&tensor, dim, true);
            let indices = argext_half::<f16>(&tensor, dim, |a, b| a.is_nan() || a > b, f16::to_f32);
            (values, indices)
        }
        DType::BF16 => {
            let values = max_dim_bf16(&tensor, dim, true);
            let indices =
                argext_half::<bf16>(&tensor, dim, |a, b| a.is_nan() || a > b, bf16::to_f32);
            (values, indices)
        }
        DType::I64 => max_dim_with_indices_impl::<i64>(&tensor, dim),
        DType::I32 => max_dim_with_indices_impl::<i32>(&tensor, dim),
        _ => panic!(
            "max_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

/// Min along a dimension with indices, returning (values, indices) in a single pass.
pub fn min_dim_with_indices(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    assert!(
        tensor.layout().shape().dims[dim] > 0,
        "min_dim_with_indices: dimension {dim} has size 0"
    );
    match tensor.dtype() {
        DType::F32 => min_dim_with_indices_float_impl::<f32>(&tensor, dim),
        DType::F64 => min_dim_with_indices_float_impl::<f64>(&tensor, dim),
        DType::F16 => {
            let values = min_dim_f16(&tensor, dim, true);
            let indices = argext_half::<f16>(&tensor, dim, |a, b| a.is_nan() || a < b, f16::to_f32);
            (values, indices)
        }
        DType::BF16 => {
            let values = min_dim_bf16(&tensor, dim, true);
            let indices =
                argext_half::<bf16>(&tensor, dim, |a, b| a.is_nan() || a < b, bf16::to_f32);
            (values, indices)
        }
        DType::I64 => min_dim_with_indices_impl::<i64>(&tensor, dim),
        DType::I32 => min_dim_with_indices_impl::<i32>(&tensor, dim),
        _ => panic!(
            "min_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

fn max_dim_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<E> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx];
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                }
            }
            values.push(max_val);
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Float-specific max_dim that propagates NaN.
fn max_dim_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<E> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx];
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val.is_nan() || val > max_val {
                    max_val = val;
                }
            }
            values.push(max_val);
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

fn min_dim_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<E> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx];
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val < min_val {
                    min_val = val;
                }
            }
            values.push(min_val);
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Float-specific min_dim that propagates NaN.
fn min_dim_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<E> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx];
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val.is_nan() || val < min_val {
                    min_val = val;
                }
            }
            values.push(min_val);
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Float-specific max_dim_with_indices that propagates NaN.
fn max_dim_with_indices_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let cap = outer_size.max(1) * inner_size.max(1);

    let mut values: Vec<E> = Vec::with_capacity(cap);
    let mut indices: Vec<i64> = Vec::with_capacity(cap);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx];
            let mut max_idx: i64 = 0;
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val.is_nan() || val > max_val {
                    max_val = val;
                    max_idx = d as i64;
                }
            }
            values.push(max_val);
            indices.push(max_idx);
        }
    }

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    );
    (val_tensor, idx_tensor)
}

/// Float-specific min_dim_with_indices that propagates NaN.
fn min_dim_with_indices_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let cap = outer_size.max(1) * inner_size.max(1);

    let mut values: Vec<E> = Vec::with_capacity(cap);
    let mut indices: Vec<i64> = Vec::with_capacity(cap);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx];
            let mut min_idx: i64 = 0;
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val.is_nan() || val < min_val {
                    min_val = val;
                    min_idx = d as i64;
                }
            }
            values.push(min_val);
            indices.push(min_idx);
        }
    }

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    );
    (val_tensor, idx_tensor)
}

fn max_dim_with_indices_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let cap = outer_size.max(1) * inner_size.max(1);

    let mut values: Vec<E> = Vec::with_capacity(cap);
    let mut indices: Vec<i64> = Vec::with_capacity(cap);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx];
            let mut max_idx: i64 = 0;
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val > max_val {
                    max_val = val;
                    max_idx = d as i64;
                }
            }
            values.push(max_val);
            indices.push(max_idx);
        }
    }

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    );
    (val_tensor, idx_tensor)
}

fn min_dim_with_indices_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let cap = outer_size.max(1) * inner_size.max(1);

    let mut values: Vec<E> = Vec::with_capacity(cap);
    let mut indices: Vec<i64> = Vec::with_capacity(cap);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx];
            let mut min_idx: i64 = 0;
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if val < min_val {
                    min_val = val;
                    min_idx = d as i64;
                }
            }
            values.push(min_val);
            indices.push(min_idx);
        }
    }

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    );
    (val_tensor, idx_tensor)
}

fn max_dim_f16(tensor: &FlexTensor, dim: usize, _values_only: bool) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[f16] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<f16> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx].to_f32();
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx].to_f32();
                if val.is_nan() || val > max_val {
                    max_val = val;
                }
            }
            values.push(f16::from_f32(max_val));
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F16,
    )
}

fn min_dim_f16(tensor: &FlexTensor, dim: usize, _values_only: bool) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[f16] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<f16> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx].to_f32();
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx].to_f32();
                if val.is_nan() || val < min_val {
                    min_val = val;
                }
            }
            values.push(f16::from_f32(min_val));
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F16,
    )
}

fn max_dim_bf16(tensor: &FlexTensor, dim: usize, _values_only: bool) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[bf16] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<bf16> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut max_val = data[first_idx].to_f32();
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx].to_f32();
                if val.is_nan() || val > max_val {
                    max_val = val;
                }
            }
            values.push(bf16::from_f32(max_val));
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::BF16,
    )
}

fn min_dim_bf16(tensor: &FlexTensor, dim: usize, _values_only: bool) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();
    let data: &[bf16] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut values: Vec<bf16> = Vec::with_capacity(outer_size.max(1) * inner_size.max(1));
    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let first_idx = start_offset + outer * dim_size * inner_size + inner;
            let mut min_val = data[first_idx].to_f32();
            for d in 1..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx].to_f32();
                if val.is_nan() || val < min_val {
                    min_val = val;
                }
            }
            values.push(bf16::from_f32(min_val));
        }
    }

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::BF16,
    )
}

// ============================================================================
// Scalar division helpers
// ============================================================================

fn scalar_div<E: Element + bytemuck::Pod + core::ops::Div<Output = E> + Copy>(
    mut tensor: FlexTensor,
    divisor: E,
) -> FlexTensor {
    let data: &mut [E] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = *x / divisor;
    }
    tensor
}

fn scalar_div_f16(mut tensor: FlexTensor, divisor: f32) -> FlexTensor {
    let data: &mut [f16] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = f16::from_f32(x.to_f32() / divisor);
    }
    tensor
}

fn scalar_div_bf16(mut tensor: FlexTensor, divisor: f32) -> FlexTensor {
    let data: &mut [bf16] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = bf16::from_f32(x.to_f32() / divisor);
    }
    tensor
}

// ============================================================================
// Argmax / Argmin implementations
// ============================================================================

fn argmax_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_idx: i64 = 0;
            let mut max_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
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
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

/// Arg-extremum for half-precision types, comparing via f32 conversion.
///
/// `is_better` returns true when `new` should replace `current` (e.g., `>` for argmax, `<` for argmin).
fn argext_half<E: Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
    is_better: fn(f32, f32) -> bool,
    to_f32: fn(E) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(dim < ndims);

    let dim_size = shape.dims[dim];
    let mut out_shape: Vec<usize> = shape.dims.clone();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    let outer_size: usize = shape.dims[..dim].iter().product();
    let inner_size: usize = shape.dims[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut best_idx: i64 = 0;
            let mut best_val: Option<f32> = None;

            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = to_f32(data[idx]);
                if best_val.is_none() || is_better(val, best_val.unwrap()) {
                    best_val = Some(val);
                    best_idx = d as i64;
                }
            }
            result.push(best_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

fn argmin_impl<E: Element + bytemuck::Pod + PartialOrd>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut min_idx: i64 = 0;
            let mut min_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
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
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

/// Float-specific argmax that propagates NaN.
fn argmax_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_idx: i64 = 0;
            let mut max_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if max_val.is_none() || val.is_nan() || val > max_val.unwrap() {
                    max_val = Some(val);
                    max_idx = d as i64;
                }
            }
            result.push(max_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::I64,
    )
}

/// Float-specific argmin that propagates NaN.
fn argmin_float_impl<E: num_traits::Float + Element + bytemuck::Pod>(
    tensor: &FlexTensor,
    dim: usize,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
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
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<i64> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut min_idx: i64 = 0;
            let mut min_val: Option<E> = None;

            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                let val = data[idx];
                if min_val.is_none() || val.is_nan() || val < min_val.unwrap() {
                    min_val = Some(val);
                    min_idx = d as i64;
                }
            }
            result.push(min_idx);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
        let result = sum(tensor);

        assert_eq!(result.layout().shape().dims, vec![1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15.0]);
    }

    #[test]
    fn test_sum_2d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum(tensor);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![21.0]);
    }

    #[test]
    fn test_sum_transposed() {
        // Test that sum works on non-contiguous (transposed) tensor without copying
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
        let result = mean_dim(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.0, 5.0]);
    }

    #[test]
    fn test_mean_dim_i8_large_dimension() {
        // dim_size=200 exceeds i8::MAX (127). Before the fix, 200 as i8 = -56,
        // causing wrong results (or 256 as i8 = 0 causing div-by-zero).
        // Use shape [1, 200] with mostly zeros so sum doesn't overflow i8.
        let mut data: Vec<i8> = vec![0i8; 200];
        // Put one non-zero value to test: sum = 100, mean = 100 / 200 = 0
        data[0] = 100;
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 200]));
        let result = mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i8> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // integer division: 100 / 200 = 0
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_mean_dim_i16_large_dimension() {
        // dim_size=40000 exceeds i16::MAX (32767).
        // sum = 32000, mean = 32000 / 40000 = 0 (integer division)
        let mut data: Vec<i16> = vec![0i16; 40000];
        data[0] = 32000;
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 40000]));
        let result = mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i16> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_argmax_1d() {
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
        let result = argmin(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0, 1]); // indices of min in each row
    }

    #[test]
    fn test_prod() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [4]));
        let result = prod(tensor);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![24.0]);
    }

    #[test]
    fn test_sum_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
        let result = sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().dims, vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6, 15]);
    }

    #[test]
    fn test_argmax_i32() {
        let data: Vec<i32> = vec![1, 5, 3, 2, 4];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [4]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
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
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 2]));
        let flipped = crate::ops::flip::flip(tensor, &[0, 1]);
        assert!(flipped.layout().strides()[0] < 0);
        assert!(flipped.layout().strides()[1] < 0);

        let result = sum(flipped);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![10.0]);
    }

    // Regression tests: argmax/argmin on permuted 4D tensors (was index OOB)

    #[test]
    fn test_argmax_permuted_4d() {
        // Shape [2,3,4,5] permuted to [2,4,3,5] via permute [0,2,1,3]
        // argmax on dim 3 should work without panicking
        let n = 2 * 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3, 4, 5]));
        let permuted = tensor.permute(&[0, 2, 1, 3]);

        assert!(!permuted.is_contiguous());
        assert_eq!(permuted.layout().shape().dims, vec![2, 4, 3, 5]);

        let result = argmax(permuted.clone(), 3);
        assert_eq!(result.layout().shape().dims, vec![2, 4, 3, 1]);

        // Verify values are valid indices (0..5)
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        for &v in &values {
            assert!(v >= 0 && v < 5, "argmax index out of range: {v}");
        }

        // Also test argmax on dim 2 of permuted tensor
        let result = argmax(permuted, 2);
        assert_eq!(result.layout().shape().dims, vec![2, 4, 1, 5]);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        for &v in &values {
            assert!(v >= 0 && v < 3, "argmax index out of range: {v}");
        }
    }

    #[test]
    fn test_argmin_permuted_4d() {
        let n = 2 * 3 * 4 * 5;
        let data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3, 4, 5]));
        let permuted = tensor.permute(&[0, 2, 1, 3]);

        assert!(!permuted.is_contiguous());

        let result = argmin(permuted, 3);
        assert_eq!(result.layout().shape().dims, vec![2, 4, 3, 1]);

        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        for &v in &values {
            assert!(v >= 0 && v < 5, "argmin index out of range: {v}");
        }
    }

    // Regression: 4D tensor reducing a middle dim (YOLOv8n crash)
    #[test]
    fn test_argmax_4d_middle_dim() {
        // Shape [1, 84, 80, 80], argmax dim=1
        // inner_size = 80*80 = 6400, old code used strides[2]=80 causing OOB
        let n = 1 * 84 * 80 * 80;
        let data: Vec<f32> = (0..n).map(|i| (i % 84) as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 84, 80, 80]));

        let result = argmax(tensor, 1);
        assert_eq!(result.layout().shape().dims, vec![1, 1, 80, 80]);

        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values.len(), 6400);
        for &v in &values {
            assert!(v >= 0 && v < 84, "argmax index out of range: {v}");
        }
    }

    #[test]
    fn test_sum_dim_4d_middle_dim() {
        // Shape [1, 84, 80, 80], sum_dim dim=1
        // Fill with 1.0 so every output position should sum to 84.0
        let shape = [1, 84, 80, 80];
        let n: usize = shape.iter().product();
        let data: Vec<f32> = vec![1.0; n];
        let tensor = FlexTensor::from_data(TensorData::new(data, shape));

        let result = sum_dim(tensor, 1);
        assert_eq!(result.layout().shape().dims, vec![1, 1, 80, 80]);

        let values: Vec<f32> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values.len(), 6400);
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - 84.0).abs() < 1e-4,
                "sum_dim mismatch at position {i}: got {v}, expected 84.0"
            );
        }
    }

    #[test]
    fn test_argmax_permuted_correctness() {
        // Verify actual correctness, not just no-panic
        // Data: [2,2,3] = [[[ 1, 2, 3], [ 4, 5, 6]],
        //                   [[ 7, 8, 9], [10,11,12]]]
        // Permute [0,2,1] -> [2,2,3] but with transposed middle dims:
        //   [[[ 1, 2, 3], [ 7, 8, 9]],
        //    [[ 4, 5, 6], [10,11,12]]]
        // argmax on dim 2 of permuted:
        //   row [1,2,3] -> idx 2, row [7,8,9] -> idx 2
        //   row [4,5,6] -> idx 2, row [10,11,12] -> idx 2
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 2, 3]));
        let permuted = tensor.permute(&[0, 2, 1]);
        // Shape is now [2, 3, 2]

        let result = argmax(permuted, 2);
        assert_eq!(result.layout().shape().dims, vec![2, 3, 1]);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        // For each row along dim 2, the second element (from the original dim 0 of 2x3 blocks)
        // is always larger: [1 vs 4] -> idx 1, [2 vs 5] -> idx 1, etc.
        assert_eq!(values, vec![1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_max_dim_nan_propagation() {
        let data: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 3]));
        let result = max_dim(tensor, 1);
        let values: Vec<f32> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert!(values[0].is_nan());
    }

    #[test]
    fn test_min_dim_nan_propagation() {
        let data: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 3]));
        let result = min_dim(tensor, 1);
        let values: Vec<f32> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert!(values[0].is_nan());
    }

    #[test]
    fn test_max_dim_with_indices_nan_propagation() {
        let data: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 3]));
        let (values, indices) = max_dim_with_indices(tensor, 1);
        let vals: Vec<f32> = bytemuck::cast_slice(&values.into_data().bytes).to_vec();
        let idxs: Vec<i64> = bytemuck::cast_slice(&indices.into_data().bytes).to_vec();
        assert!(vals[0].is_nan());
        assert_eq!(idxs[0], 1); // NaN is at index 1
    }

    #[test]
    fn test_argmax_nan_propagation() {
        let data: Vec<f32> = vec![1.0, f32::NAN, 3.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 3]));
        let result = argmax(tensor, 1);
        let values: Vec<i64> = bytemuck::cast_slice(&result.into_data().bytes).to_vec();
        assert_eq!(values[0], 1); // NaN is at index 1
    }

    #[test]
    #[should_panic(expected = "dimension 0 has size 0")]
    fn test_max_dim_zero_size_panics() {
        let tensor = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]));
        max_dim(tensor, 0);
    }

    #[test]
    #[should_panic(expected = "dimension 1 has size 0")]
    fn test_min_dim_zero_size_panics() {
        let tensor = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0]));
        min_dim(tensor, 1);
    }
}

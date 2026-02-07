//! Matrix multiplication via gemm crate.
//!
//! Optimizations:
//! - Uses strided gemm to avoid copying transposed tensors
//! - Enables parallelism for large matrices (with rayon feature)
//! - Batched matmul parallelized across batch dimension

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{FlexTensor, Layout};

/// Checked multiplication for matrix sizes, panics on overflow.
#[inline]
fn checked_size(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("matmul: matrix size overflow")
}

/// Threshold for enabling parallelism (M*N*K operations).
/// 192^3 = ~7M ops - balance between 128x128 (no parallel) and 256x256 (parallel)
const PARALLEL_THRESHOLD: usize = 192 * 192 * 192;

/// Threshold for batch-level parallelism (total ops across all batches).
/// Use batch parallelism when individual matrices are small but total work is large.
#[cfg(feature = "rayon")]
const BATCH_PARALLEL_THRESHOLD: usize = 128 * 128 * 128; // ~2M ops total

/// Get parallelism setting based on matrix size.
fn get_parallelism(m: usize, n: usize, k: usize) -> gemm::Parallelism {
    let ops = m.saturating_mul(n).saturating_mul(k);
    if ops >= PARALLEL_THRESHOLD {
        #[cfg(feature = "rayon")]
        {
            gemm::Parallelism::Rayon(0) // 0 = use all available threads
        }
        #[cfg(not(feature = "rayon"))]
        {
            gemm::Parallelism::None
        }
    } else {
        gemm::Parallelism::None
    }
}

/// Dispatch matrix multiplication based on dtype.
pub fn matmul(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "matmul: dtype mismatch");

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    debug_assert!(lhs_rank >= 2, "matmul requires at least 2D tensors");
    debug_assert!(rhs_rank >= 2, "matmul requires at least 2D tensors");

    // Check inner dimensions match: lhs[..., M, K] x rhs[..., K, N]
    let k_lhs = lhs_shape.dims[lhs_rank - 1];
    let k_rhs = rhs_shape.dims[rhs_rank - 2];
    debug_assert_eq!(k_lhs, k_rhs, "matmul: inner dimensions must match");

    match lhs.dtype() {
        DType::F32 => matmul_f32(lhs, rhs),
        DType::F64 => matmul_f64(lhs, rhs),
        DType::F16 => matmul_f16(lhs, rhs),
        DType::BF16 => matmul_bf16(lhs, rhs),
        _ => panic!("matmul: unsupported dtype {:?}", lhs.dtype()),
    }
}

/// Extract 2D matrix strides from a tensor layout.
/// Returns (row_stride, col_stride) for the last two dimensions.
fn get_2d_strides(layout: &Layout) -> (isize, isize) {
    let strides = layout.strides();
    let ndim = strides.len();
    let row_stride = strides[ndim - 2];
    let col_stride = strides[ndim - 1];
    (row_stride, col_stride)
}

/// Compute broadcast batch dimensions for batched matmul.
/// Returns (broadcast_shape, lhs_strides, rhs_strides) where strides map
/// output batch index to input batch offset (in matrices).
fn broadcast_batch_dims(
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // Pad shorter batch dims with 1s on the left
    let max_len = lhs_batch.len().max(rhs_batch.len());
    let lhs_padded: Vec<usize> = (0..max_len)
        .map(|i| {
            if i < max_len - lhs_batch.len() {
                1
            } else {
                lhs_batch[i - (max_len - lhs_batch.len())]
            }
        })
        .collect();
    let rhs_padded: Vec<usize> = (0..max_len)
        .map(|i| {
            if i < max_len - rhs_batch.len() {
                1
            } else {
                rhs_batch[i - (max_len - rhs_batch.len())]
            }
        })
        .collect();

    // Compute broadcast shape and strides
    let mut broadcast_shape = Vec::with_capacity(max_len);
    let mut lhs_strides = Vec::with_capacity(max_len);
    let mut rhs_strides = Vec::with_capacity(max_len);

    // Compute strides from right to left
    let mut lhs_stride = 1usize;
    let mut rhs_stride = 1usize;
    for i in (0..max_len).rev() {
        let ld = lhs_padded[i];
        let rd = rhs_padded[i];
        debug_assert!(
            ld == rd || ld == 1 || rd == 1,
            "matmul: batch dimensions not broadcastable: {:?} vs {:?}",
            lhs_batch,
            rhs_batch
        );
        broadcast_shape.push(ld.max(rd));
        // Stride is 0 if dimension is 1 (broadcast), otherwise actual stride
        lhs_strides.push(if ld == 1 { 0 } else { lhs_stride });
        rhs_strides.push(if rd == 1 { 0 } else { rhs_stride });
        lhs_stride *= ld;
        rhs_stride *= rd;
    }

    // Reverse to get correct order
    broadcast_shape.reverse();
    lhs_strides.reverse();
    rhs_strides.reverse();

    (broadcast_shape, lhs_strides, rhs_strides)
}

/// Convert a flat batch index to input batch offset using broadcast strides.
#[inline]
fn batch_index_to_offset(b: usize, broadcast_shape: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0;
    let mut remaining = b;
    for i in (0..broadcast_shape.len()).rev() {
        let idx = remaining % broadcast_shape[i];
        offset += idx * strides[i];
        remaining /= broadcast_shape[i];
    }
    offset
}

// ============================================================================
// f32 matmul
// ============================================================================

fn matmul_f32(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_rank = lhs.layout().shape().num_dims();
    let rhs_rank = rhs.layout().shape().num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_strided_f32(lhs, rhs)
    } else {
        matmul_batched_f32(lhs, rhs)
    }
}

/// 2D matmul with strided support: [M, K] x [K, N] -> [M, N]
/// Avoids copying transposed tensors by passing strides directly to gemm.
fn matmul_2d_strided_f32(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape.dims[0];
    let k = lhs_shape.dims[1];
    let n = rhs_shape.dims[1];

    // Get strides for both inputs (handles transposed tensors)
    let (lhs_row_stride, lhs_col_stride) = get_2d_strides(lhs.layout());
    let (rhs_row_stride, rhs_col_stride) = get_2d_strides(rhs.layout());

    // Get data pointers with offset
    let lhs_data: &[f32] = lhs.storage();
    let rhs_data: &[f32] = rhs.storage();
    let lhs_ptr = unsafe { lhs_data.as_ptr().add(lhs.layout().start_offset()) };
    let rhs_ptr = unsafe { rhs_data.as_ptr().add(rhs.layout().start_offset()) };

    // Allocate contiguous output
    let out_shape = Shape::from(vec![m, n]);
    let mut output = FlexTensor::empty(out_shape, DType::F32);
    let out_data: &mut [f32] = output.storage_mut();

    let parallelism = get_parallelism(m, n, k);

    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out_data.as_mut_ptr(),
            1,          // dst col stride (contiguous)
            n as isize, // dst row stride
            false,      // don't read dst
            lhs_ptr,
            lhs_col_stride,
            lhs_row_stride,
            rhs_ptr,
            rhs_col_stride,
            rhs_row_stride,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            parallelism,
        );
    }

    output
}

/// Perform a single batched gemm call for f32.
#[inline]
unsafe fn gemm_single_f32(
    out: *mut f32,
    lhs: *const f32,
    rhs: *const f32,
    m: usize,
    n: usize,
    k: usize,
    parallelism: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out,
            1,
            n as isize,
            false,
            lhs,
            1,
            k as isize,
            rhs,
            1,
            n as isize,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            parallelism,
        );
    }
}

/// Batched matmul for f32: [B..., M, K] x [B..., K, N] -> [B..., M, N]
/// Supports broadcasting on batch dimensions.
fn matmul_batched_f32(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    // For batched, we need contiguous layout to iterate over batches
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape.dims[lhs_rank - 2];
    let k = lhs_shape.dims[lhs_rank - 1];
    let n = rhs_shape.dims[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape.dims[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape.dims[..rhs_rank - 2].to_vec();

    // Compute broadcast batch dimensions
    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let mut output = FlexTensor::empty(out_shape, DType::F32);

    let lhs_data: &[f32] = lhs.storage();
    let rhs_data: &[f32] = rhs.storage();
    let out_data: &mut [f32] = output.storage_mut();

    let per_matrix_ops = m.saturating_mul(n).saturating_mul(k);

    // Strategy:
    // 1. Large matrices: let gemm parallelize internally
    // 2. Small matrices, large batch: parallelize batch loop
    // 3. Small total work: single-threaded
    #[cfg(feature = "rayon")]
    {
        let total_ops = batch_size.saturating_mul(per_matrix_ops);

        // Heuristic: batch parallelism is better when we have enough batches
        // to saturate cores (>=4), even for large matrices. This avoids
        // repeated thread spawn/sync overhead.
        let prefer_batch_parallel = batch_size >= 4 && total_ops >= BATCH_PARALLEL_THRESHOLD;

        if per_matrix_ops >= PARALLEL_THRESHOLD && !prefer_batch_parallel {
            // Large matrices, small batch: use per-matrix parallelism
            let parallelism = gemm::Parallelism::Rayon(0);
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;

                unsafe {
                    gemm_single_f32(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        parallelism,
                    );
                }
            }
        } else if total_ops >= BATCH_PARALLEL_THRESHOLD && batch_size > 1 {
            // Small matrices but enough total work: parallelize batch loop
            use rayon::prelude::*;

            // Split output into chunks, one per batch
            let out_chunks: Vec<&mut [f32]> = out_data.chunks_mut(out_matrix_size).collect();

            out_chunks
                .into_par_iter()
                .enumerate()
                .for_each(|(b, out_chunk)| {
                    let lhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                    let rhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;

                    unsafe {
                        gemm_single_f32(
                            out_chunk.as_mut_ptr(),
                            lhs_data[lhs_offset..].as_ptr(),
                            rhs_data[rhs_offset..].as_ptr(),
                            m,
                            n,
                            k,
                            gemm::Parallelism::None,
                        );
                    }
                });
        } else {
            // Small total work: single-threaded
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;

                unsafe {
                    gemm_single_f32(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = per_matrix_ops; // silence unused warning
        for b in 0..batch_size {
            let lhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
            let rhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
            let out_offset = b * out_matrix_size;

            unsafe {
                gemm_single_f32(
                    out_data[out_offset..].as_mut_ptr(),
                    lhs_data[lhs_offset..].as_ptr(),
                    rhs_data[rhs_offset..].as_ptr(),
                    m,
                    n,
                    k,
                    gemm::Parallelism::None,
                );
            }
        }
    }

    output
}

// ============================================================================
// f64 matmul
// ============================================================================

fn matmul_f64(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_rank = lhs.layout().shape().num_dims();
    let rhs_rank = rhs.layout().shape().num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_strided_f64(lhs, rhs)
    } else {
        matmul_batched_f64(lhs, rhs)
    }
}

fn matmul_2d_strided_f64(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape.dims[0];
    let k = lhs_shape.dims[1];
    let n = rhs_shape.dims[1];

    let (lhs_row_stride, lhs_col_stride) = get_2d_strides(lhs.layout());
    let (rhs_row_stride, rhs_col_stride) = get_2d_strides(rhs.layout());

    let lhs_data: &[f64] = lhs.storage();
    let rhs_data: &[f64] = rhs.storage();
    let lhs_ptr = unsafe { lhs_data.as_ptr().add(lhs.layout().start_offset()) };
    let rhs_ptr = unsafe { rhs_data.as_ptr().add(rhs.layout().start_offset()) };

    let out_shape = Shape::from(vec![m, n]);
    let mut output = FlexTensor::empty(out_shape, DType::F64);
    let out_data: &mut [f64] = output.storage_mut();

    let parallelism = get_parallelism(m, n, k);

    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out_data.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs_ptr,
            lhs_col_stride,
            lhs_row_stride,
            rhs_ptr,
            rhs_col_stride,
            rhs_row_stride,
            0.0f64,
            1.0f64,
            false,
            false,
            false,
            parallelism,
        );
    }

    output
}

/// Perform a single batched gemm call for f64.
#[inline]
unsafe fn gemm_single_f64(
    out: *mut f64,
    lhs: *const f64,
    rhs: *const f64,
    m: usize,
    n: usize,
    k: usize,
    parallelism: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out,
            1,
            n as isize,
            false,
            lhs,
            1,
            k as isize,
            rhs,
            1,
            n as isize,
            0.0f64,
            1.0f64,
            false,
            false,
            false,
            parallelism,
        );
    }
}

fn matmul_batched_f64(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape.dims[lhs_rank - 2];
    let k = lhs_shape.dims[lhs_rank - 1];
    let n = rhs_shape.dims[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape.dims[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape.dims[..rhs_rank - 2].to_vec();

    // Compute broadcast batch dimensions
    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let mut output = FlexTensor::empty(out_shape, DType::F64);

    let lhs_data: &[f64] = lhs.storage();
    let rhs_data: &[f64] = rhs.storage();
    let out_data: &mut [f64] = output.storage_mut();

    let per_matrix_ops = m.saturating_mul(n).saturating_mul(k);

    #[cfg(feature = "rayon")]
    {
        let total_ops = batch_size.saturating_mul(per_matrix_ops);
        let prefer_batch_parallel = batch_size >= 4 && total_ops >= BATCH_PARALLEL_THRESHOLD;

        if per_matrix_ops >= PARALLEL_THRESHOLD && !prefer_batch_parallel {
            let parallelism = gemm::Parallelism::Rayon(0);
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;

                unsafe {
                    gemm_single_f64(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        parallelism,
                    );
                }
            }
        } else if total_ops >= BATCH_PARALLEL_THRESHOLD && batch_size > 1 {
            use rayon::prelude::*;

            let out_chunks: Vec<&mut [f64]> = out_data.chunks_mut(out_matrix_size).collect();

            out_chunks
                .into_par_iter()
                .enumerate()
                .for_each(|(b, out_chunk)| {
                    let lhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                    let rhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;

                    unsafe {
                        gemm_single_f64(
                            out_chunk.as_mut_ptr(),
                            lhs_data[lhs_offset..].as_ptr(),
                            rhs_data[rhs_offset..].as_ptr(),
                            m,
                            n,
                            k,
                            gemm::Parallelism::None,
                        );
                    }
                });
        } else {
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;

                unsafe {
                    gemm_single_f64(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = per_matrix_ops; // silence unused warning
        for b in 0..batch_size {
            let lhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
            let rhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
            let out_offset = b * out_matrix_size;

            unsafe {
                gemm_single_f64(
                    out_data[out_offset..].as_mut_ptr(),
                    lhs_data[lhs_offset..].as_ptr(),
                    rhs_data[rhs_offset..].as_ptr(),
                    m,
                    n,
                    k,
                    gemm::Parallelism::None,
                );
            }
        }
    }

    output
}

// ============================================================================
// f16 matmul (native gemm support)
// ============================================================================

fn matmul_f16(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_rank = lhs.layout().shape().num_dims();
    let rhs_rank = rhs.layout().shape().num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_strided_f16(lhs, rhs)
    } else {
        matmul_batched_f16(lhs, rhs)
    }
}

fn matmul_2d_strided_f16(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape.dims[0];
    let k = lhs_shape.dims[1];
    let n = rhs_shape.dims[1];

    let (lhs_row_stride, lhs_col_stride) = get_2d_strides(lhs.layout());
    let (rhs_row_stride, rhs_col_stride) = get_2d_strides(rhs.layout());

    let lhs_data: &[f16] = lhs.storage();
    let rhs_data: &[f16] = rhs.storage();
    let lhs_ptr = unsafe { lhs_data.as_ptr().add(lhs.layout().start_offset()) };
    let rhs_ptr = unsafe { rhs_data.as_ptr().add(rhs.layout().start_offset()) };

    let out_shape = Shape::from(vec![m, n]);
    let mut output = FlexTensor::empty(out_shape, DType::F16);
    let out_data: &mut [f16] = output.storage_mut();

    let parallelism = get_parallelism(m, n, k);

    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out_data.as_mut_ptr(),
            1,
            n as isize,
            false,
            lhs_ptr,
            lhs_col_stride,
            lhs_row_stride,
            rhs_ptr,
            rhs_col_stride,
            rhs_row_stride,
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            false,
            false,
            false,
            parallelism,
        );
    }

    output
}

/// Perform a single batched gemm call for f16.
#[inline]
unsafe fn gemm_single_f16(
    out: *mut f16,
    lhs: *const f16,
    rhs: *const f16,
    m: usize,
    n: usize,
    k: usize,
    parallelism: gemm::Parallelism,
) {
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out,
            1,
            n as isize,
            false,
            lhs,
            1,
            k as isize,
            rhs,
            1,
            n as isize,
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            false,
            false,
            false,
            parallelism,
        );
    }
}

fn matmul_batched_f16(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape.dims[lhs_rank - 2];
    let k = lhs_shape.dims[lhs_rank - 1];
    let n = rhs_shape.dims[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape.dims[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape.dims[..rhs_rank - 2].to_vec();

    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let mut output = FlexTensor::empty(out_shape, DType::F16);

    let lhs_data: &[f16] = lhs.storage();
    let rhs_data: &[f16] = rhs.storage();
    let out_data: &mut [f16] = output.storage_mut();

    let per_matrix_ops = m.saturating_mul(n).saturating_mul(k);

    #[cfg(feature = "rayon")]
    {
        let total_ops = batch_size.saturating_mul(per_matrix_ops);
        let prefer_batch_parallel = batch_size >= 4 && total_ops >= BATCH_PARALLEL_THRESHOLD;

        if per_matrix_ops >= PARALLEL_THRESHOLD && !prefer_batch_parallel {
            let parallelism = gemm::Parallelism::Rayon(0);
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;
                unsafe {
                    gemm_single_f16(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        parallelism,
                    );
                }
            }
        } else if total_ops >= BATCH_PARALLEL_THRESHOLD && batch_size > 1 {
            use rayon::prelude::*;

            let out_chunks: Vec<&mut [f16]> = out_data.chunks_mut(out_matrix_size).collect();

            out_chunks
                .into_par_iter()
                .enumerate()
                .for_each(|(b, out_chunk)| {
                    let lhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                    let rhs_offset =
                        batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                    unsafe {
                        gemm_single_f16(
                            out_chunk.as_mut_ptr(),
                            lhs_data[lhs_offset..].as_ptr(),
                            rhs_data[rhs_offset..].as_ptr(),
                            m,
                            n,
                            k,
                            gemm::Parallelism::None,
                        );
                    }
                });
        } else {
            for b in 0..batch_size {
                let lhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
                let rhs_offset =
                    batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
                let out_offset = b * out_matrix_size;
                unsafe {
                    gemm_single_f16(
                        out_data[out_offset..].as_mut_ptr(),
                        lhs_data[lhs_offset..].as_ptr(),
                        rhs_data[rhs_offset..].as_ptr(),
                        m,
                        n,
                        k,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let _ = per_matrix_ops;
        for b in 0..batch_size {
            let lhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &lhs_strides) * lhs_matrix_size;
            let rhs_offset =
                batch_index_to_offset(b, &broadcast_shape, &rhs_strides) * rhs_matrix_size;
            let out_offset = b * out_matrix_size;
            unsafe {
                gemm_single_f16(
                    out_data[out_offset..].as_mut_ptr(),
                    lhs_data[lhs_offset..].as_ptr(),
                    rhs_data[rhs_offset..].as_ptr(),
                    m,
                    n,
                    k,
                    gemm::Parallelism::None,
                );
            }
        }
    }

    output
}

// ============================================================================
// bf16 matmul (via f32 conversion)
// ============================================================================

fn matmul_bf16(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    // Convert bf16 -> f32
    let lhs_f32: Vec<f32> = lhs.storage::<bf16>().iter().map(|x| x.to_f32()).collect();
    let rhs_f32: Vec<f32> = rhs.storage::<bf16>().iter().map(|x| x.to_f32()).collect();

    // Create f32 tensors
    let lhs_f32_tensor = FlexTensor::new(
        Bytes::from_elems(lhs_f32),
        Layout::contiguous(lhs_shape.clone()),
        DType::F32,
    );
    let rhs_f32_tensor = FlexTensor::new(
        Bytes::from_elems(rhs_f32),
        Layout::contiguous(rhs_shape.clone()),
        DType::F32,
    );

    // Compute matmul in f32
    let result_f32 = matmul_f32(lhs_f32_tensor, rhs_f32_tensor);

    // Convert f32 -> bf16
    let result_bf16: Vec<bf16> = result_f32
        .storage::<f32>()
        .iter()
        .map(|x| bf16::from_f32(*x))
        .collect();

    FlexTensor::new(
        Bytes::from_elems(result_bf16),
        result_f32.layout().clone(),
        DType::BF16,
    )
}

// ============================================================================
// Integer matmul (naive with SIMD)
// ============================================================================

/// Integer matrix multiplication dispatch.
pub fn int_matmul(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "int_matmul: dtype mismatch");

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    debug_assert!(lhs_rank >= 2, "int_matmul requires at least 2D tensors");
    debug_assert!(rhs_rank >= 2, "int_matmul requires at least 2D tensors");

    let k_lhs = lhs_shape.dims[lhs_rank - 1];
    let k_rhs = rhs_shape.dims[rhs_rank - 2];
    debug_assert_eq!(k_lhs, k_rhs, "int_matmul: inner dimensions must match");

    match lhs.dtype() {
        DType::I32 => matmul_i32(lhs, rhs),
        DType::I64 => matmul_i64(lhs, rhs),
        _ => panic!("int_matmul: unsupported dtype {:?}", lhs.dtype()),
    }
}

/// i32 matmul using naive triple loop with SIMD dot product.
fn matmul_i32(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_i32(&lhs, &rhs)
    } else {
        matmul_batched_i32(lhs, rhs)
    }
}

/// 2D i32 matmul: [M, K] x [K, N] -> [M, N]
/// Transposes rhs to enable contiguous access for dot product.
fn matmul_2d_i32(lhs: &FlexTensor, rhs: &FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape.dims[0];
    let k = lhs_shape.dims[1];
    let n = rhs_shape.dims[1];

    let lhs_data: &[i32] = lhs.storage();
    let rhs_data: &[i32] = rhs.storage();

    // Transpose rhs [K, N] -> [N, K] for contiguous column access
    let mut rhs_t = vec![0i32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs_t[j * k + i] = rhs_data[i * n + j];
        }
    }

    let mut output = vec![0i32; m * n];

    // Now both lhs rows and rhs columns (transposed rows) are contiguous
    for i in 0..m {
        let lhs_row = &lhs_data[i * k..(i + 1) * k];
        for j in 0..n {
            let rhs_col = &rhs_t[j * k..(j + 1) * k];
            output[i * n + j] = dot_i32(lhs_row, rhs_col);
        }
    }

    let out_shape = Shape::from(vec![m, n]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I32,
    )
}

/// SIMD-optimized dot product for i32 slices.
#[inline]
fn dot_i32(a: &[i32], b: &[i32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        dot_i32_neon(a, b)
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    {
        dot_i32_scalar(a, b)
    }
}

/// Scalar dot product fallback.
#[inline]
#[allow(dead_code)]
fn dot_i32_scalar(a: &[i32], b: &[i32]) -> i32 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
    }
    sum
}

/// NEON-optimized dot product for i32.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
fn dot_i32_neon(a: &[i32], b: &[i32]) -> i32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut sum = 0i32;
    let mut i = 0;

    // Process 4 elements at a time (128-bit NEON)
    if len >= 4 {
        unsafe {
            let mut acc = vdupq_n_s32(0);

            while i + 4 <= len {
                let va = vld1q_s32(a.as_ptr().add(i));
                let vb = vld1q_s32(b.as_ptr().add(i));
                // Multiply and accumulate: acc += va * vb
                acc = vmlaq_s32(acc, va, vb);
                i += 4;
            }

            // Horizontal sum of acc vector
            sum = vaddvq_s32(acc);
        }
    }

    // Handle remaining elements
    while i < len {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
        i += 1;
    }

    sum
}

/// Batched i32 matmul: [B..., M, K] x [B..., K, N] -> [B..., M, N]
///
/// Uses naive triple-loop with SIMD dot product and batch-level parallelism.
fn matmul_batched_i32(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape.dims[lhs_rank - 2];
    let k = lhs_shape.dims[lhs_rank - 1];
    let n = rhs_shape.dims[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape.dims[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape.dims[..rhs_rank - 2].to_vec();

    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let rhs_batch_size: usize = rhs_batch.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let lhs_data: &[i32] = lhs.storage();
    let rhs_data: &[i32] = rhs.storage();

    // Transpose rhs per actual rhs batch: [B_rhs, K, N] -> [B_rhs, N, K]
    let mut rhs_transposed = vec![0i32; rhs_batch_size * n * k];
    for b in 0..rhs_batch_size {
        let src_offset = b * rhs_matrix_size;
        let dst_offset = b * n * k;
        for i in 0..k {
            for j in 0..n {
                rhs_transposed[dst_offset + j * k + i] = rhs_data[src_offset + i * n + j];
            }
        }
    }

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let mut output = vec![0i32; batch_size * out_matrix_size];

        output
            .par_chunks_mut(out_matrix_size)
            .enumerate()
            .for_each(|(b, out_slice)| {
                let lhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &lhs_strides);
                let rhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &rhs_strides);
                let lhs_offset = lhs_batch_idx * lhs_matrix_size;
                let rhs_t_offset = rhs_batch_idx * n * k;

                let lhs_slice = &lhs_data[lhs_offset..lhs_offset + lhs_matrix_size];
                let rhs_t_slice = &rhs_transposed[rhs_t_offset..rhs_t_offset + n * k];

                for i in 0..m {
                    let lhs_row = &lhs_slice[i * k..(i + 1) * k];
                    for j in 0..n {
                        let rhs_col = &rhs_t_slice[j * k..(j + 1) * k];
                        out_slice[i * n + j] = dot_i32(lhs_row, rhs_col);
                    }
                }
            });

        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            DType::I32,
        )
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut output = vec![0i32; batch_size * out_matrix_size];

        for b in 0..batch_size {
            let lhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &lhs_strides);
            let rhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &rhs_strides);
            let lhs_offset = lhs_batch_idx * lhs_matrix_size;
            let rhs_t_offset = rhs_batch_idx * n * k;
            let out_offset = b * out_matrix_size;

            let lhs_slice = &lhs_data[lhs_offset..lhs_offset + lhs_matrix_size];
            let rhs_t_slice = &rhs_transposed[rhs_t_offset..rhs_t_offset + n * k];

            for i in 0..m {
                let lhs_row = &lhs_slice[i * k..(i + 1) * k];
                for j in 0..n {
                    let rhs_col = &rhs_t_slice[j * k..(j + 1) * k];
                    output[out_offset + i * n + j] = dot_i32(lhs_row, rhs_col);
                }
            }
        }

        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            DType::I32,
        )
    }
}

/// i64 matmul using naive triple loop.
fn matmul_i64(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_i64(&lhs, &rhs)
    } else {
        matmul_batched_i64(lhs, rhs)
    }
}

/// 2D i64 matmul: [M, K] x [K, N] -> [M, N]
fn matmul_2d_i64(lhs: &FlexTensor, rhs: &FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape.dims[0];
    let k = lhs_shape.dims[1];
    let n = rhs_shape.dims[1];

    let lhs_data: &[i64] = lhs.storage();
    let rhs_data: &[i64] = rhs.storage();

    let mut output = vec![0i64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for l in 0..k {
                sum = sum.wrapping_add(lhs_data[i * k + l].wrapping_mul(rhs_data[l * n + j]));
            }
            output[i * n + j] = sum;
        }
    }

    let out_shape = Shape::from(vec![m, n]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I64,
    )
}

/// Batched i64 matmul with broadcast support
fn matmul_batched_i64(lhs: FlexTensor, rhs: FlexTensor) -> FlexTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape.dims[lhs_rank - 2];
    let k = lhs_shape.dims[lhs_rank - 1];
    let n = rhs_shape.dims[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape.dims[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape.dims[..rhs_rank - 2].to_vec();

    // Compute broadcast batch dimensions
    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let lhs_data: &[i64] = lhs.storage();
    let rhs_data: &[i64] = rhs.storage();

    let mut output = vec![0i64; batch_size * out_matrix_size];

    for b in 0..batch_size {
        let lhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &lhs_strides);
        let rhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &rhs_strides);
        let lhs_offset = lhs_batch_idx * lhs_matrix_size;
        let rhs_offset = rhs_batch_idx * rhs_matrix_size;
        let out_offset = b * out_matrix_size;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i64;
                for l in 0..k {
                    let lhs_idx = lhs_offset + i * k + l;
                    let rhs_idx = rhs_offset + l * n + j;
                    sum = sum.wrapping_add(lhs_data[lhs_idx].wrapping_mul(rhs_data[rhs_idx]));
                }
                output[out_offset + i * n + j] = sum;
            }
        }
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
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
    fn test_matmul_2d_simple() {
        // [2, 3] x [3, 2] -> [2, 2]
        let lhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let rhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);

        assert_eq!(result.layout().shape().dims, vec![2, 2]);

        let result_data = result.into_data();
        let values: Vec<f32> = result_data.to_vec().unwrap();

        // [1 2 3] * [1 2]   = [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
        // [4 5 6]   [3 4]     [4*1+5*3+6*5  4*2+5*4+6*6]   [49 64]
        //           [5 6]
        assert_eq!(values, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_square() {
        let lhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs_data = TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        let values: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(values, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let lhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let identity = TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(identity);

        let result = matmul(lhs, rhs);
        let values: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_transposed_lhs() {
        // Original: [2, 3] with data [[1,2,3], [4,5,6]]
        // Transposed: [3, 2] with logical [[1,4], [2,5], [3,6]]
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        let transposed = tensor.transpose(0, 1); // [3, 2]
        assert!(!transposed.is_contiguous());

        // rhs: [2, 2] identity
        let rhs_data = TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]);
        let rhs = FlexTensor::from_data(rhs_data);

        // [3, 2] x [2, 2] -> [3, 2]
        let result = matmul(transposed, rhs);
        assert_eq!(result.layout().shape().dims, vec![3, 2]);

        let values: Vec<f32> = result.into_data().to_vec().unwrap();
        // Result should be [[1,4], [2,5], [3,6]] * I = [[1,4], [2,5], [3,6]]
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_transposed_rhs() {
        // lhs: [2, 2] identity
        let lhs_data = TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]);
        let lhs = FlexTensor::from_data(lhs_data);

        // Original: [3, 2] -> Transposed: [2, 3]
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let tensor = FlexTensor::from_data(data);
        let transposed = tensor.transpose(0, 1); // [2, 3]
        assert!(!transposed.is_contiguous());

        // [2, 2] x [2, 3] -> [2, 3]
        let result = matmul(lhs, transposed);
        assert_eq!(result.layout().shape().dims, vec![2, 3]);

        let values: Vec<f32> = result.into_data().to_vec().unwrap();
        // I * [[1,3,5], [2,4,6]] = [[1,3,5], [2,4,6]]
        assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_matmul_both_transposed() {
        // lhs: [2, 3] transposed to [3, 2]
        let lhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let lhs = FlexTensor::from_data(lhs_data).transpose(0, 1);

        // rhs: [2, 3] transposed to [3, 2] then we need [2, 3] for matmul
        // Actually let's do: [3, 2] transposed to [2, 3]
        let rhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let rhs = FlexTensor::from_data(rhs_data).transpose(0, 1);

        // [3, 2] x [2, 3] -> [3, 3]
        let result = matmul(lhs, rhs);
        assert_eq!(result.layout().shape().dims, vec![3, 3]);
    }

    #[test]
    fn test_matmul_batched_simple() {
        let lhs_data = TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
            ],
            vec![2, 2, 2],
        );
        let rhs_data = TensorData::new(
            vec![
                1.0f32, 0.0, 0.0, 1.0, // identity batch 0
                2.0, 0.0, 0.0, 2.0, // scaled identity batch 1
            ],
            vec![2, 2, 2],
        );

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        assert_eq!(result.layout().shape().dims, vec![2, 2, 2]);

        let values: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(
            values,
            vec![
                1.0, 2.0, 3.0, 4.0, // batch 0: identity
                10.0, 12.0, 14.0, 16.0, // batch 1: scaled by 2
            ]
        );
    }

    #[test]
    fn test_matmul_f64() {
        let lhs_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs_data = TensorData::new(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        let values: Vec<f64> = result.into_data().to_vec().unwrap();

        assert_eq!(values, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_f16() {
        let lhs_data = TensorData::new(
            vec![
                f16::from_f32(1.0),
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::from_f32(4.0),
            ],
            vec![2, 2],
        );
        let rhs_data = TensorData::new(
            vec![
                f16::from_f32(5.0),
                f16::from_f32(6.0),
                f16::from_f32(7.0),
                f16::from_f32(8.0),
            ],
            vec![2, 2],
        );

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        let values: Vec<f16> = result.into_data().to_vec().unwrap();

        let expected = vec![
            f16::from_f32(19.0),
            f16::from_f32(22.0),
            f16::from_f32(43.0),
            f16::from_f32(50.0),
        ];

        for (a, b) in values.iter().zip(expected.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 0.1, "f16 matmul mismatch");
        }
    }

    #[test]
    fn test_matmul_bf16() {
        let lhs_data = TensorData::new(
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
            vec![2, 2],
        );
        let rhs_data = TensorData::new(
            vec![
                bf16::from_f32(5.0),
                bf16::from_f32(6.0),
                bf16::from_f32(7.0),
                bf16::from_f32(8.0),
            ],
            vec![2, 2],
        );

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        let values: Vec<bf16> = result.into_data().to_vec().unwrap();

        let expected = vec![
            bf16::from_f32(19.0),
            bf16::from_f32(22.0),
            bf16::from_f32(43.0),
            bf16::from_f32(50.0),
        ];

        for (a, b) in values.iter().zip(expected.iter()) {
            assert!(
                (a.to_f32() - b.to_f32()).abs() < 0.5,
                "bf16 matmul mismatch"
            );
        }
    }

    #[test]
    fn test_matmul_rectangular() {
        // [1, 4] x [4, 1] -> [1, 1] (dot product)
        let lhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![1, 4]);
        let rhs_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![4, 1]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = matmul(lhs, rhs);
        assert_eq!(result.layout().shape().dims, vec![1, 1]);

        let values: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(values, vec![30.0]);
    }

    // ========================================================================
    // Integer matmul tests
    // ========================================================================

    #[test]
    fn test_int_matmul_i32_simple() {
        // [2, 3] x [3, 2] -> [2, 2]
        let lhs_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], vec![2, 3]);
        let rhs_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], vec![3, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = int_matmul(lhs, rhs);
        assert_eq!(result.layout().shape().dims, vec![2, 2]);

        let values: Vec<i32> = result.into_data().to_vec().unwrap();
        // [1 2 3] * [1 2]   = [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
        // [4 5 6]   [3 4]     [4*1+5*3+6*5  4*2+5*4+6*6]   [49 64]
        //           [5 6]
        assert_eq!(values, vec![22, 28, 49, 64]);
    }

    #[test]
    fn test_int_matmul_i32_square() {
        let lhs_data = TensorData::new(vec![1i32, 2, 3, 4], vec![2, 2]);
        let rhs_data = TensorData::new(vec![5i32, 6, 7, 8], vec![2, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = int_matmul(lhs, rhs);
        let values: Vec<i32> = result.into_data().to_vec().unwrap();

        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(values, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_int_matmul_i64() {
        let lhs_data = TensorData::new(vec![1i64, 2, 3, 4], vec![2, 2]);
        let rhs_data = TensorData::new(vec![5i64, 6, 7, 8], vec![2, 2]);

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = int_matmul(lhs, rhs);
        let values: Vec<i64> = result.into_data().to_vec().unwrap();

        assert_eq!(values, vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_int_matmul_i32_batched() {
        let lhs_data = TensorData::new(
            vec![
                1i32, 2, 3, 4, // batch 0
                5, 6, 7, 8, // batch 1
            ],
            vec![2, 2, 2],
        );
        let rhs_data = TensorData::new(
            vec![
                1i32, 0, 0, 1, // identity batch 0
                2, 0, 0, 2, // scaled identity batch 1
            ],
            vec![2, 2, 2],
        );

        let lhs = FlexTensor::from_data(lhs_data);
        let rhs = FlexTensor::from_data(rhs_data);

        let result = int_matmul(lhs, rhs);
        assert_eq!(result.layout().shape().dims, vec![2, 2, 2]);

        let values: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(
            values,
            vec![
                1, 2, 3, 4, // batch 0: identity
                10, 12, 14, 16, // batch 1: scaled by 2
            ]
        );
    }
}

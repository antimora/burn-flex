//! Portable SIMD kernels using macerator.
//!
//! These kernels work across all architectures supported by macerator:
//! - aarch64 (NEON)
//! - x86_64 (AVX2, AVX512, SSE)
//! - wasm32 (SIMD128)
//! - Scalar fallback for embedded/other platforms

#[cfg(target_arch = "x86_64")]
use core::iter::Sum;
use core::ops::AddAssign;

#[cfg(target_arch = "x86_64")]
use macerator::ReduceAdd;
use macerator::{ReduceMax, ReduceMin, Simd, VAdd, VOrd, vload_unaligned, vstore_unaligned};

// ============================================================================
// Sum reduction
// ============================================================================

/// Sum all elements in a f32 slice.
///
/// Uses 8-fold unrolled loop that LLVM auto-vectorizes.
/// This matches ndarray's approach and is faster than explicit SIMD dispatch
/// due to lower overhead.
#[inline]
pub fn sum_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    let res = macerator_sum(data);
    #[cfg(not(target_arch = "x86_64"))]
    let res = unrolled_sum_f32(data);
    res
}

/// 8-fold unrolled sum that LLVM auto-vectorizes.
/// This is the same approach used by ndarray's `unrolled_fold`.
#[inline]
#[cfg(not(target_arch = "x86_64"))]
fn unrolled_sum_f32(mut xs: &[f32]) -> f32 {
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    while xs.len() >= 8 {
        p0 += xs[0];
        p1 += xs[1];
        p2 += xs[2];
        p3 += xs[3];
        p4 += xs[4];
        p5 += xs[5];
        p6 += xs[6];
        p7 += xs[7];
        xs = &xs[8..];
    }

    // Combine accumulators in a way that allows further vectorization
    let mut sum = (p0 + p4) + (p1 + p5) + (p2 + p6) + (p3 + p7);

    // Handle remainder
    for &x in xs {
        sum += x;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[macerator::with_simd]
fn macerator_sum<S: Simd, F: VAdd + Sum + ReduceAdd>(mut xs: &[F]) -> F {
    let lanes = F::lanes::<S>();
    let mut sum = F::default().splat::<S>();

    while xs.len() >= lanes {
        sum += unsafe { vload_unaligned(xs.as_ptr()) };
        xs = &xs[lanes..];
    }

    sum.reduce_add() + xs.iter().copied().sum()
}

// ============================================================================
// Scatter-add for dimension reductions
// ============================================================================

/// Scatter-add: for each row, add to corresponding output positions.
/// Used for cache-friendly first-dim and middle-dim reductions.
///
/// # Arguments
/// * `src` - Source data pointer
/// * `dst` - Destination accumulator (must be pre-zeroed)
/// * `num_rows` - Number of rows to sum
/// * `row_len` - Length of each row (columns)
/// * `src_row_stride` - Stride between source rows
#[macerator::with_simd]
pub fn scatter_add_f32<S: Simd, F: VAdd + AddAssign>(
    src: &[F],
    dst: &mut [F],
    num_rows: usize,
    row_len: usize,
    src_row_stride: usize,
) {
    let lanes = F::lanes::<S>();

    for row in 0..num_rows {
        let row_start = row * src_row_stride;
        let row_data = &src[row_start..row_start + row_len];

        let simd_len = row_len / lanes * lanes;

        // SIMD accumulate
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let s = vload_unaligned(row_data.as_ptr().add(i));
                let d = vload_unaligned(dst.as_ptr().add(i));
                vstore_unaligned::<S, _>(dst.as_mut_ptr().add(i), d + s);
            }
            i += lanes;
        }

        // Scalar tail
        for j in simd_len..row_len {
            dst[j] += row_data[j];
        }
    }
}

/// Batched scatter-add for middle-dim reductions.
/// For tensors like [B, M, K] reducing dim=1.
#[macerator::with_simd]
pub fn scatter_add_batched<S: Simd, F: VAdd + AddAssign>(
    src: &[F],
    dst: &mut [F],
    num_batches: usize,
    num_rows: usize,
    row_len: usize,
    batch_stride: usize,
    row_stride: usize,
) {
    let lanes = F::lanes::<S>();

    for batch in 0..num_batches {
        let batch_src_start = batch * batch_stride;
        let batch_dst_start = batch * row_len;
        let batch_dst = &mut dst[batch_dst_start..batch_dst_start + row_len];

        for row in 0..num_rows {
            let row_start = batch_src_start + row * row_stride;
            let row_data = &src[row_start..row_start + row_len];

            let simd_len = row_len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let s = vload_unaligned(row_data.as_ptr().add(i));
                    let d = vload_unaligned(batch_dst.as_ptr().add(i));
                    vstore_unaligned::<S, _>(batch_dst.as_mut_ptr().add(i), d + s);
                }
                i += lanes;
            }

            for j in simd_len..row_len {
                batch_dst[j] += row_data[j];
            }
        }
    }
}

// ============================================================================
// Row-wise sum (last-dim reduction)
// ============================================================================

/// Sum each row, storing results in output slice.
/// Used for last-dim reductions.
///
/// Uses 8-fold unrolled loop per row that LLVM auto-vectorizes.
#[inline]
pub fn sum_rows_f32(src: &[f32], dst: &mut [f32], num_rows: usize, row_len: usize) {
    for (row, dst_val) in dst.iter_mut().enumerate().take(num_rows) {
        let row_start = row * row_len;
        let row_data = &src[row_start..row_start + row_len];
        #[cfg(not(target_arch = "x86_64"))]
        {
            *dst_val = unrolled_sum_f32(row_data);
        }
        #[cfg(target_arch = "x86_64")]
        {
            *dst_val = macerator_sum(row_data);
        }
    }
}

// ============================================================================
// Max/Min reduction
// ============================================================================

/// Find the maximum element in a f32 slice using SIMD.
#[inline]
pub fn max_f32(data: &[f32]) -> f32 {
    macerator_max(data, f32::NEG_INFINITY)
}

/// Find the minimum element in a f32 slice using SIMD.
#[inline]
pub fn min_f32(data: &[f32]) -> f32 {
    macerator_min(data, f32::INFINITY)
}

#[macerator::with_simd]
fn macerator_max<S: Simd, F: VOrd + ReduceMax + PartialOrd>(mut xs: &[F], init: F) -> F {
    let lanes = F::lanes::<S>();
    let mut acc = init.splat::<S>();

    while xs.len() >= lanes {
        let v = unsafe { vload_unaligned(xs.as_ptr()) };
        acc = acc.max(v);
        xs = &xs[lanes..];
    }

    let mut result = acc.reduce_max();
    for &x in xs {
        if x > result {
            result = x;
        }
    }
    result
}

#[macerator::with_simd]
fn macerator_min<S: Simd, F: VOrd + ReduceMin + PartialOrd>(mut xs: &[F], init: F) -> F {
    let lanes = F::lanes::<S>();
    let mut acc = init.splat::<S>();

    while xs.len() >= lanes {
        let v = unsafe { vload_unaligned(xs.as_ptr()) };
        acc = acc.min(v);
        xs = &xs[lanes..];
    }

    let mut result = acc.reduce_min();
    for &x in xs {
        if x < result {
            result = x;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let expected: f32 = data.iter().sum();
        let result = sum_f32(&data);
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_sum_f32_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(sum_f32(&data), 0.0);
    }

    #[test]
    fn test_sum_f32_small() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(sum_f32(&data), 6.0);
    }

    #[test]
    fn test_scatter_add_f32() {
        // Simulate reducing [3, 4] along dim=0 -> [1, 4]
        let src = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        let mut dst = vec![0.0; 4];

        scatter_add_f32(&src, &mut dst, 3, 4, 4);

        assert_eq!(dst, vec![15.0, 18.0, 21.0, 24.0]);
    }

    #[test]
    fn test_sum_rows_f32() {
        // Simulate reducing [3, 4] along dim=1 -> [3, 1]
        let src = vec![
            1.0, 2.0, 3.0, 4.0, // row 0 -> 10
            5.0, 6.0, 7.0, 8.0, // row 1 -> 26
            9.0, 10.0, 11.0, 12.0, // row 2 -> 42
        ];
        let mut dst = vec![0.0; 3];

        sum_rows_f32(&src, &mut dst, 3, 4);

        assert_eq!(dst, vec![10.0, 26.0, 42.0]);
    }

    #[test]
    fn test_max_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(max_f32(&data), 999.0);
    }

    #[test]
    fn test_max_f32_small() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(max_f32(&data), 5.0);
    }

    #[test]
    fn test_max_f32_negative() {
        let data = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        assert_eq!(max_f32(&data), -1.0);
    }

    #[test]
    fn test_min_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(min_f32(&data), 0.0);
    }

    #[test]
    fn test_min_f32_small() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(min_f32(&data), 1.0);
    }

    #[test]
    fn test_min_f32_negative() {
        let data = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        assert_eq!(min_f32(&data), -5.0);
    }
}

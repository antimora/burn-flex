//! Portable SIMD kernels using pulp.
//!
//! These kernels work across all architectures supported by pulp:
//! - aarch64 (NEON)
//! - x86_64 (AVX2, AVX512, SSE)
//! - wasm32 (SIMD128)
//! - Scalar fallback for embedded/other platforms

use pulp::{Arch, Simd, WithSimd};

/// Get the architecture-specific SIMD dispatcher.
/// This detects CPU features at runtime.
#[inline]
pub fn arch() -> Arch {
    Arch::new()
}

// ============================================================================
// Sum reduction
// ============================================================================

/// Sum all elements in a f32 slice using SIMD.
#[inline]
pub fn sum_f32(data: &[f32]) -> f32 {
    arch().dispatch(SumF32(data))
}

struct SumF32<'a>(&'a [f32]);

impl WithSimd for SumF32<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let data = self.0;
        if data.is_empty() {
            return 0.0;
        }

        // Split into SIMD-aligned chunks and scalar tail
        let (head, tail) = S::as_simd_f32s(data);

        // Accumulate SIMD lanes
        let mut acc = simd.splat_f32s(0.0);
        for &chunk in head {
            acc = simd.add_f32s(acc, chunk);
        }

        // Reduce SIMD accumulator to scalar
        let mut sum = simd.reduce_sum_f32s(acc);

        // Add scalar tail
        for &x in tail {
            sum += x;
        }

        sum
    }
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
#[inline]
pub fn scatter_add_f32(
    src: &[f32],
    dst: &mut [f32],
    num_rows: usize,
    row_len: usize,
    src_row_stride: usize,
) {
    arch().dispatch(ScatterAddF32 {
        src,
        dst,
        num_rows,
        row_len,
        src_row_stride,
    });
}

struct ScatterAddF32<'a> {
    src: &'a [f32],
    dst: &'a mut [f32],
    num_rows: usize,
    row_len: usize,
    src_row_stride: usize,
}

impl WithSimd for ScatterAddF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            src,
            dst,
            num_rows,
            row_len,
            src_row_stride,
        } = self;

        // Process each row (sequential memory access)
        for row in 0..num_rows {
            let row_start = row * src_row_stride;
            let row_data = &src[row_start..row_start + row_len];

            // Split into SIMD chunks
            let (src_head, src_tail) = S::as_simd_f32s(row_data);
            let (dst_head, dst_tail) = S::as_mut_simd_f32s(dst);

            // SIMD accumulate
            for (d, s) in dst_head.iter_mut().zip(src_head.iter()) {
                *d = simd.add_f32s(*d, *s);
            }

            // Scalar tail
            for (d, s) in dst_tail.iter_mut().zip(src_tail.iter()) {
                *d += *s;
            }
        }
    }
}

/// Batched scatter-add for middle-dim reductions.
/// For tensors like [B, M, K] reducing dim=1.
#[inline]
pub fn scatter_add_batched_f32(
    src: &[f32],
    dst: &mut [f32],
    num_batches: usize,
    num_rows: usize,
    row_len: usize,
    batch_stride: usize,
    row_stride: usize,
) {
    arch().dispatch(ScatterAddBatchedF32 {
        src,
        dst,
        num_batches,
        num_rows,
        row_len,
        batch_stride,
        row_stride,
    });
}

struct ScatterAddBatchedF32<'a> {
    src: &'a [f32],
    dst: &'a mut [f32],
    num_batches: usize,
    num_rows: usize,
    row_len: usize,
    batch_stride: usize,
    row_stride: usize,
}

impl WithSimd for ScatterAddBatchedF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            src,
            dst,
            num_batches,
            num_rows,
            row_len,
            batch_stride,
            row_stride,
        } = self;

        for batch in 0..num_batches {
            let batch_src_start = batch * batch_stride;
            let batch_dst_start = batch * row_len;
            let batch_dst = &mut dst[batch_dst_start..batch_dst_start + row_len];

            for row in 0..num_rows {
                let row_start = batch_src_start + row * row_stride;
                let row_data = &src[row_start..row_start + row_len];

                let (src_head, src_tail) = S::as_simd_f32s(row_data);
                let (dst_head, dst_tail) = S::as_mut_simd_f32s(batch_dst);

                for (d, s) in dst_head.iter_mut().zip(src_head.iter()) {
                    *d = simd.add_f32s(*d, *s);
                }

                for (d, s) in dst_tail.iter_mut().zip(src_tail.iter()) {
                    *d += *s;
                }
            }
        }
    }
}

// ============================================================================
// Row-wise sum (last-dim reduction)
// ============================================================================

/// Sum each row, storing results in output slice.
/// Used for last-dim reductions.
#[inline]
pub fn sum_rows_f32(src: &[f32], dst: &mut [f32], num_rows: usize, row_len: usize) {
    arch().dispatch(SumRowsF32 {
        src,
        dst,
        num_rows,
        row_len,
    });
}

struct SumRowsF32<'a> {
    src: &'a [f32],
    dst: &'a mut [f32],
    num_rows: usize,
    row_len: usize,
}

impl WithSimd for SumRowsF32<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            src,
            dst,
            num_rows,
            row_len,
        } = self;

        for (row, dst_val) in dst.iter_mut().enumerate().take(num_rows) {
            let row_start = row * row_len;
            let row_data = &src[row_start..row_start + row_len];

            let (head, tail) = S::as_simd_f32s(row_data);

            let mut acc = simd.splat_f32s(0.0);
            for &chunk in head {
                acc = simd.add_f32s(acc, chunk);
            }

            let mut sum = simd.reduce_sum_f32s(acc);
            for &x in tail {
                sum += x;
            }

            *dst_val = sum;
        }
    }
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
}

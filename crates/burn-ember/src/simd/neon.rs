//! NEON SIMD kernels for ARM64 (Apple Silicon, etc.)
//!
//! NEON provides 128-bit registers, processing 4x f32 simultaneously.

use core::arch::aarch64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const LANES: usize = 4;

/// Threshold for parallel execution (elements).
/// Below this, single-threaded is faster due to threading overhead.
/// For memory-bound operations, parallelism helps when data exceeds L3 cache.
/// M3 has ~36MB L3, so we set threshold at ~4M elements (16MB for f32).
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024; // 4M elements

/// SIMD add for f32 slices using NEON.
///
/// # Safety
/// Caller must ensure slices have equal length.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    // SIMD path
    if chunks > 0 {
        // SAFETY: aarch64 always has NEON
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vaddq_f32(va, vb);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    // Scalar tail
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] + b[tail_start + i];
    }
}

/// SIMD sub for f32 slices using NEON.
#[inline]
pub fn sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vsubq_f32(va, vb);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] - b[tail_start + i];
    }
}

/// SIMD mul for f32 slices using NEON.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vmulq_f32(va, vb);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] * b[tail_start + i];
    }
}

/// SIMD div for f32 slices using NEON.
///
/// Note: NEON doesn't have native division, uses reciprocal approximation
/// with Newton-Raphson refinement for speed.
#[inline]
pub fn div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                // Use vdivq_f32 for accurate division (available on ARMv8)
                let vr = vdivq_f32(va, vb);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] / b[tail_start + i];
    }
}

/// SIMD scalar add: add scalar to each element.
#[inline]
pub fn add_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let vs = vdupq_n_f32(scalar); // Broadcast scalar to all lanes

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vr = vaddq_f32(va, vs);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] + scalar;
    }
}

/// SIMD scalar mul: multiply each element by scalar.
#[inline]
pub fn mul_scalar_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());

    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let vs = vdupq_n_f32(scalar);

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vr = vmulq_f32(va, vs);
                vst1q_f32(out_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] * scalar;
    }
}

// ===================
// In-place operations
// ===================

/// SIMD in-place add: a[i] += b[i]
///
/// Uses parallel execution for large arrays when rayon feature is enabled.
#[inline]
pub fn add_inplace_f32(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        add_inplace_f32_parallel(a, b);
        return;
    }

    add_inplace_f32_sequential(a, b);
}

#[inline]
fn add_inplace_f32_sequential(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vaddq_f32(va, vb);
                vst1q_f32(a_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        a[tail_start + i] += b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn add_inplace_f32_parallel(a: &mut [f32], b: &[f32]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            add_inplace_f32_sequential(a_chunk, b_chunk);
        });
}

/// SIMD in-place sub: a[i] -= b[i]
#[inline]
pub fn sub_inplace_f32(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        sub_inplace_f32_parallel(a, b);
        return;
    }

    sub_inplace_f32_sequential(a, b);
}

#[inline]
fn sub_inplace_f32_sequential(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vsubq_f32(va, vb);
                vst1q_f32(a_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        a[tail_start + i] -= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn sub_inplace_f32_parallel(a: &mut [f32], b: &[f32]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            sub_inplace_f32_sequential(a_chunk, b_chunk);
        });
}

/// SIMD in-place mul: a[i] *= b[i]
#[inline]
pub fn mul_inplace_f32(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        mul_inplace_f32_parallel(a, b);
        return;
    }

    mul_inplace_f32_sequential(a, b);
}

#[inline]
fn mul_inplace_f32_sequential(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vmulq_f32(va, vb);
                vst1q_f32(a_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        a[tail_start + i] *= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn mul_inplace_f32_parallel(a: &mut [f32], b: &[f32]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            mul_inplace_f32_sequential(a_chunk, b_chunk);
        });
}

/// SIMD in-place div: a[i] /= b[i]
#[inline]
pub fn div_inplace_f32(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        div_inplace_f32_parallel(a, b);
        return;
    }

    div_inplace_f32_sequential(a, b);
}

#[inline]
fn div_inplace_f32_sequential(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let va = vld1q_f32(a_ptr.add(offset));
                let vb = vld1q_f32(b_ptr.add(offset));
                let vr = vdivq_f32(va, vb);
                vst1q_f32(a_ptr.add(offset), vr);
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        a[tail_start + i] /= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn div_inplace_f32_parallel(a: &mut [f32], b: &[f32]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            div_inplace_f32_sequential(a_chunk, b_chunk);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let mut out = [0.0f32; 7];

        add_f32(&a, &b, &mut out);

        assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_mul_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        let mut out = [0.0f32; 5];

        mul_f32(&a, &b, &mut out);

        assert_eq!(out, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_div_f32() {
        let a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 4.0, 5.0, 8.0];
        let mut out = [0.0f32; 4];

        div_f32(&a, &b, &mut out);

        assert_eq!(out, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_add_scalar_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];

        add_scalar_f32(&a, 10.0, &mut out);

        assert_eq!(out, [11.0, 12.0, 13.0, 14.0, 15.0]);
    }
}

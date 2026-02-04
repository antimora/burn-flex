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

// ===================
// Comparison operations (f32 -> u8 bool)
// ===================

/// Comparison operation type.
#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt, // greater than
    Ge, // greater than or equal
    Lt, // less than
    Le, // less than or equal
    Eq, // equal
    Ne, // not equal
}

/// SIMD comparison for f32 slices, output as u8 (0 or 1).
///
/// Uses parallel execution for large arrays when rayon feature is enabled.
#[inline]
pub fn cmp_f32(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_f32_parallel(a, b, out, op);
        return;
    }

    cmp_f32_sequential(a, b, out, op);
}

#[inline]
fn cmp_f32_sequential(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
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

                // Perform comparison based on operation type
                let mask: uint32x4_t = match op {
                    CmpOp::Gt => vcgtq_f32(va, vb),
                    CmpOp::Ge => vcgeq_f32(va, vb),
                    CmpOp::Lt => vcltq_f32(va, vb),
                    CmpOp::Le => vcleq_f32(va, vb),
                    CmpOp::Eq => vceqq_f32(va, vb),
                    CmpOp::Ne => vmvnq_u32(vceqq_f32(va, vb)),
                };

                // Convert mask (0xFFFFFFFF or 0) to u8 (1 or 0)
                // Shift right by 31 to get 1 or 0
                let shifted = vshrq_n_u32::<31>(mask);
                // Narrow from u32x4 -> u16x4 -> u8x8
                let narrow16 = vmovn_u32(shifted);
                let narrow8 = vmovn_u16(vcombine_u16(narrow16, narrow16));

                // Store first 4 bytes
                vst1_lane_u32::<0>(
                    out_ptr.add(offset) as *mut u32,
                    vreinterpret_u32_u8(narrow8),
                );
            }
        }
    }

    // Scalar tail
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let av = a[tail_start + i];
        let bv = b[tail_start + i];
        out[tail_start + i] = match op {
            CmpOp::Gt => (av > bv) as u8,
            CmpOp::Ge => (av >= bv) as u8,
            CmpOp::Lt => (av < bv) as u8,
            CmpOp::Le => (av <= bv) as u8,
            CmpOp::Eq => (av == bv) as u8,
            CmpOp::Ne => (av != bv) as u8,
        };
    }
}

#[cfg(feature = "rayon")]
fn cmp_f32_parallel(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_f32_sequential(&a[start..end], &b[start..end], out_chunk, op);
        });
}

/// SIMD scalar comparison for f32 slice vs scalar, output as u8.
#[inline]
pub fn cmp_scalar_f32(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_scalar_f32_parallel(a, scalar, out, op);
        return;
    }

    cmp_scalar_f32_sequential(a, scalar, out, op);
}

#[inline]
fn cmp_scalar_f32_sequential(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
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

                let mask: uint32x4_t = match op {
                    CmpOp::Gt => vcgtq_f32(va, vs),
                    CmpOp::Ge => vcgeq_f32(va, vs),
                    CmpOp::Lt => vcltq_f32(va, vs),
                    CmpOp::Le => vcleq_f32(va, vs),
                    CmpOp::Eq => vceqq_f32(va, vs),
                    CmpOp::Ne => vmvnq_u32(vceqq_f32(va, vs)),
                };

                let shifted = vshrq_n_u32::<31>(mask);
                let narrow16 = vmovn_u32(shifted);
                let narrow8 = vmovn_u16(vcombine_u16(narrow16, narrow16));
                vst1_lane_u32::<0>(
                    out_ptr.add(offset) as *mut u32,
                    vreinterpret_u32_u8(narrow8),
                );
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let av = a[tail_start + i];
        out[tail_start + i] = match op {
            CmpOp::Gt => (av > scalar) as u8,
            CmpOp::Ge => (av >= scalar) as u8,
            CmpOp::Lt => (av < scalar) as u8,
            CmpOp::Le => (av <= scalar) as u8,
            CmpOp::Eq => (av == scalar) as u8,
            CmpOp::Ne => (av != scalar) as u8,
        };
    }
}

#[cfg(feature = "rayon")]
fn cmp_scalar_f32_parallel(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_scalar_f32_sequential(&a[start..end], scalar, out_chunk, op);
        });
}

// ===================
// Boolean operations (u8)
// ===================

const U8_LANES: usize = 16; // NEON: 128-bit / 8-bit = 16

/// SIMD boolean NOT: out[i] = !a[i] (0 becomes 1, non-zero becomes 0)
#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_u8_parallel(a, out);
        return;
    }

    bool_not_u8_sequential(a, out);
}

/// SIMD boolean NOT in-place: a[i] = !a[i]
#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_inplace_u8_parallel(a);
        return;
    }

    bool_not_inplace_u8_sequential(a);
}

#[inline]
fn bool_not_inplace_u8_sequential(a: &mut [u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let ptr = a.as_mut_ptr();
            let ones = vdupq_n_u8(1);
            let zeros = vdupq_n_u8(0);

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(ptr.add(offset));
                let is_zero = vceqq_u8(va, zeros);
                let result = vandq_u8(is_zero, ones);
                vst1q_u8(ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        a[tail_start + i] = (a[tail_start + i] == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_inplace_u8_parallel(a: &mut [u8]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        bool_not_inplace_u8_sequential(chunk);
    });
}

#[inline]
fn bool_not_u8_sequential(a: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let out_ptr = out.as_mut_ptr();
            let ones = vdupq_n_u8(1);
            let zeros = vdupq_n_u8(0);

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                // Compare with zero: if a[i] == 0, result is 0xFF, else 0x00
                let is_zero = vceqq_u8(va, zeros);
                // Select 1 where a was 0, else 0
                let result = vandq_u8(is_zero, ones);
                vst1q_u8(out_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        out[tail_start + i] = (a[tail_start + i] == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_u8_parallel(a: &[u8], out: &mut [u8]) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_not_u8_sequential(&a[start..end], out_chunk);
        });
}

/// SIMD boolean AND: out[i] = a[i] & b[i]
#[inline]
pub fn bool_and_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_and_u8_parallel(a, b, out);
        return;
    }

    bool_and_u8_sequential(a, b, out);
}

#[inline]
fn bool_and_u8_sequential(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = vandq_u8(va, vb);
                vst1q_u8(out_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] & b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_and_u8_parallel(a: &[u8], b: &[u8], out: &mut [u8]) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_and_u8_sequential(&a[start..end], &b[start..end], out_chunk);
        });
}

/// SIMD boolean OR: out[i] = a[i] | b[i]
#[inline]
pub fn bool_or_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_or_u8_parallel(a, b, out);
        return;
    }

    bool_or_u8_sequential(a, b, out);
}

#[inline]
fn bool_or_u8_sequential(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = vorrq_u8(va, vb);
                vst1q_u8(out_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] | b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_or_u8_parallel(a: &[u8], b: &[u8], out: &mut [u8]) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_or_u8_sequential(&a[start..end], &b[start..end], out_chunk);
        });
}

/// SIMD boolean XOR: out[i] = a[i] ^ b[i]
#[inline]
pub fn bool_xor_u8(a: &[u8], b: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_xor_u8_parallel(a, b, out);
        return;
    }

    bool_xor_u8_sequential(a, b, out);
}

#[inline]
fn bool_xor_u8_sequential(a: &[u8], b: &[u8], out: &mut [u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let out_ptr = out.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = veorq_u8(va, vb);
                vst1q_u8(out_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        out[tail_start + i] = a[tail_start + i] ^ b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_xor_u8_parallel(a: &[u8], b: &[u8], out: &mut [u8]) {
    const CHUNK_SIZE: usize = 4096;
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_xor_u8_sequential(&a[start..end], &b[start..end], out_chunk);
        });
}

/// SIMD boolean AND in-place: a[i] &= b[i]
#[inline]
pub fn bool_and_inplace_u8(a: &mut [u8], b: &[u8]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_and_inplace_u8_parallel(a, b);
        return;
    }

    bool_and_inplace_u8_sequential(a, b);
}

#[inline]
fn bool_and_inplace_u8_sequential(a: &mut [u8], b: &[u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = vandq_u8(va, vb);
                vst1q_u8(a_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        a[tail_start + i] &= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_and_inplace_u8_parallel(a: &mut [u8], b: &[u8]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            bool_and_inplace_u8_sequential(a_chunk, b_chunk);
        });
}

/// SIMD boolean OR in-place: a[i] |= b[i]
#[inline]
pub fn bool_or_inplace_u8(a: &mut [u8], b: &[u8]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_or_inplace_u8_parallel(a, b);
        return;
    }

    bool_or_inplace_u8_sequential(a, b);
}

#[inline]
fn bool_or_inplace_u8_sequential(a: &mut [u8], b: &[u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = vorrq_u8(va, vb);
                vst1q_u8(a_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        a[tail_start + i] |= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_or_inplace_u8_parallel(a: &mut [u8], b: &[u8]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            bool_or_inplace_u8_sequential(a_chunk, b_chunk);
        });
}

/// SIMD boolean XOR in-place: a[i] ^= b[i]
#[inline]
pub fn bool_xor_inplace_u8(a: &mut [u8], b: &[u8]) {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_xor_inplace_u8_parallel(a, b);
        return;
    }

    bool_xor_inplace_u8_sequential(a, b);
}

#[inline]
fn bool_xor_inplace_u8_sequential(a: &mut [u8], b: &[u8]) {
    let len = a.len();
    let chunks = len / U8_LANES;
    let remainder = len % U8_LANES;

    if chunks > 0 {
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let b_ptr = b.as_ptr();

            for i in 0..chunks {
                let offset = i * U8_LANES;
                let va = vld1q_u8(a_ptr.add(offset));
                let vb = vld1q_u8(b_ptr.add(offset));
                let result = veorq_u8(va, vb);
                vst1q_u8(a_ptr.add(offset), result);
            }
        }
    }

    let tail_start = chunks * U8_LANES;
    for i in 0..remainder {
        a[tail_start + i] ^= b[tail_start + i];
    }
}

#[cfg(feature = "rayon")]
fn bool_xor_inplace_u8_parallel(a: &mut [u8], b: &[u8]) {
    const CHUNK_SIZE: usize = 4096;
    a.par_chunks_mut(CHUNK_SIZE)
        .zip(b.par_chunks(CHUNK_SIZE))
        .for_each(|(a_chunk, b_chunk)| {
            bool_xor_inplace_u8_sequential(a_chunk, b_chunk);
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

    // Comparison tests
    #[test]
    fn test_cmp_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0f32, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let mut out = [0u8; 7];

        cmp_f32(&a, &b, &mut out, CmpOp::Gt);

        // 1>2=F, 2>2=F, 3>2=T, 4>4=F, 5>4=T, 6>4=T, 7>4=T
        assert_eq!(out, [0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn test_cmp_ge_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 2.0, 2.0, 5.0];
        let mut out = [0u8; 4];

        cmp_f32(&a, &b, &mut out, CmpOp::Ge);

        // 1>=2=F, 2>=2=T, 3>=2=T, 4>=5=F
        assert_eq!(out, [0, 1, 1, 0]);
    }

    #[test]
    fn test_cmp_eq_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];

        cmp_f32(&a, &b, &mut out, CmpOp::Eq);

        // 1==1=T, 2==3=F, 3==3=T, 4==5=F, 5==5=T
        assert_eq!(out, [1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_cmp_scalar_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0u8; 5];

        cmp_scalar_f32(&a, 3.0, &mut out, CmpOp::Gt);

        // 1>3=F, 2>3=F, 3>3=F, 4>3=T, 5>3=T
        assert_eq!(out, [0, 0, 0, 1, 1]);
    }

    // Boolean operation tests
    #[test]
    fn test_bool_not_u8() {
        let a = [1u8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0];
        let mut out = [0u8; 18];

        bool_not_u8(&a, &mut out);

        let expected = [0u8, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_and_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];

        bool_and_u8(&a, &b, &mut out);

        let expected = [1u8, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_or_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1];
        let mut out = [0u8; 18];

        bool_or_u8(&a, &b, &mut out);

        let expected = [1u8, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_xor_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];

        bool_xor_u8(&a, &b, &mut out);

        let expected = [0u8, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
        assert_eq!(out, expected);
    }
}

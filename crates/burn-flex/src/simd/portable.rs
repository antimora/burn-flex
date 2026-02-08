//! Portable SIMD kernels using macerator.
//!
//! Replaces platform-specific implementations (neon.rs) with a single
//! portable implementation that auto-dispatches to NEON/AVX2/SSE/SIMD128/scalar.

use macerator::{Scalar, Simd, VBitAnd, VBitOr, VBitXor, VOrd, vload_unaligned, vstore_unaligned};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Threshold for parallel execution (elements).
/// For memory-bound operations, parallelism helps when data exceeds L3 cache.
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024;

#[cfg(feature = "rayon")]
const CHUNK_SIZE: usize = 4096;

// ============================================================================
// f32 in-place binary ops
// ============================================================================

macro_rules! define_inplace_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(a: &mut [f32], b: &[f32]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b);
                return;
            }

            $seq_fn(a, b);
        }

        #[macerator::with_simd]
        #[allow(clippy::assign_op_pattern)]
        fn $seq_fn<S: Simd>(a: &mut [f32], b: &[f32]) {
            let lanes = S::lanes32();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned(a.as_ptr().add(i));
                    let vb = vload_unaligned(b.as_ptr().add(i));
                    vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), va $op vb);
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &mut [f32], b: &[f32]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $seq_fn(a_chunk, b_chunk);
                });
        }
    };
}

define_inplace_f32_op!(add_inplace_f32, add_inplace_f32_seq, add_inplace_f32_par, +);
define_inplace_f32_op!(sub_inplace_f32, sub_inplace_f32_seq, sub_inplace_f32_par, -);
define_inplace_f32_op!(mul_inplace_f32, mul_inplace_f32_seq, mul_inplace_f32_par, *);
define_inplace_f32_op!(div_inplace_f32, div_inplace_f32_seq, div_inplace_f32_par, /);

// ============================================================================
// f32 comparison ops
// ============================================================================

#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

#[inline]
pub fn cmp_f32(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_f32_par(a, b, out, op);
        return;
    }

    cmp_seq(a, b, out, op);
}

#[macerator::with_simd]
fn cmp_seq<S: Simd, T: VOrd + PartialOrd>(a: &[T], b: &[T], out: &mut [u8], op: CmpOp) {
    let lanes = T::lanes::<S>();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, _>(a.as_ptr().add(i));
            let vb = vload_unaligned::<S, _>(b.as_ptr().add(i));

            let out_ptr = out.as_mut_ptr().add(i) as *mut bool;
            let out_mask = match op {
                CmpOp::Gt => va.gt(vb),
                CmpOp::Ge => va.ge(vb),
                CmpOp::Lt => va.lt(vb),
                CmpOp::Le => va.le(vb),
                CmpOp::Eq => va.eq(vb),
                CmpOp::Ne => va.ne(vb),
            };
            // mask_store_as_bool writes `bool` (1 byte per lane, 0 or 1)
            // which has the same repr as u8 0/1
            out_mask.store_as_bool(out_ptr);
        }
        i += lanes;
    }

    // Scalar tail
    for j in simd_len..len {
        out[j] = match op {
            CmpOp::Gt => (a[j] > b[j]) as u8,
            CmpOp::Ge => (a[j] >= b[j]) as u8,
            CmpOp::Lt => (a[j] < b[j]) as u8,
            CmpOp::Le => (a[j] <= b[j]) as u8,
            CmpOp::Eq => (a[j] == b[j]) as u8,
            CmpOp::Ne => (a[j] != b[j]) as u8,
        };
    }
}

#[cfg(feature = "rayon")]
fn cmp_f32_par(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_seq(&a[start..end], &b[start..end], out_chunk, op);
        });
}

#[inline]
pub fn cmp_scalar_f32(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_scalar_f32_par(a, scalar, out, op);
        return;
    }

    cmp_scalar_seq(a, scalar, out, op);
}

#[macerator::with_simd]
fn cmp_scalar_seq<S: Simd, T: VOrd + PartialOrd>(a: &[T], scalar: T, out: &mut [u8], op: CmpOp) {
    let lanes = T::lanes::<S>();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let vs = scalar.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, _>(a.as_ptr().add(i));
            let out_ptr = out.as_mut_ptr().add(i) as *mut bool;
            let out_mask = match op {
                CmpOp::Gt => va.gt(vs),
                CmpOp::Ge => va.ge(vs),
                CmpOp::Lt => va.lt(vs),
                CmpOp::Le => va.le(vs),
                CmpOp::Eq => va.eq(vs),
                CmpOp::Ne => va.ne(vs),
            };
            out_mask.store_as_bool(out_ptr);
        }
        i += lanes;
    }

    for j in simd_len..len {
        out[j] = match op {
            CmpOp::Gt => (a[j] > scalar) as u8,
            CmpOp::Ge => (a[j] >= scalar) as u8,
            CmpOp::Lt => (a[j] < scalar) as u8,
            CmpOp::Le => (a[j] <= scalar) as u8,
            CmpOp::Eq => (a[j] == scalar) as u8,
            CmpOp::Ne => (a[j] != scalar) as u8,
        };
    }
}

#[cfg(feature = "rayon")]
fn cmp_scalar_f32_par(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_scalar_seq(&a[start..end], scalar, out_chunk, op);
        });
}

// ============================================================================
// u8 boolean ops
// ============================================================================

macro_rules! define_bool_binary_u8_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident,
     $inplace_pub:ident, $inplace_seq:ident, $inplace_par:ident,
     $trait:ident, $method:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b, out);
                return;
            }

            $seq_fn(a, b, out);
        }

        #[macerator::with_simd]
        fn $seq_fn<S: Simd>(a: &[u8], b: &[u8], out: &mut [u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        out.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                out[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            out.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(a.len());
                    $seq_fn(&a[start..end], &b[start..end], out_chunk);
                });
        }

        #[inline]
        pub fn $inplace_pub(a: &mut [u8], b: &[u8]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $inplace_par(a, b);
                return;
            }

            $inplace_seq(a, b);
        }

        #[allow(clippy::assign_op_pattern)]
        #[macerator::with_simd]
        fn $inplace_seq<S: Simd>(a: &mut [u8], b: &[u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        a.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $inplace_par(a: &mut [u8], b: &[u8]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $inplace_seq(a_chunk, b_chunk);
                });
        }
    };
}

define_bool_binary_u8_op!(
    bool_and_u8, bool_and_u8_seq, bool_and_u8_par,
    bool_and_inplace_u8, bool_and_inplace_u8_seq, bool_and_inplace_u8_par,
    VBitAnd, vbitand, &);
define_bool_binary_u8_op!(
    bool_or_u8, bool_or_u8_seq, bool_or_u8_par,
    bool_or_inplace_u8, bool_or_inplace_u8_seq, bool_or_inplace_u8_par,
    VBitOr, vbitor, |);
define_bool_binary_u8_op!(
    bool_xor_u8, bool_xor_u8_seq, bool_xor_u8_par,
    bool_xor_inplace_u8, bool_xor_inplace_u8_seq, bool_xor_inplace_u8_par,
    VBitXor, vbitxor, ^);

// Boolean NOT is special (unary), implemented separately.

#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_u8_par(a, out);
        return;
    }

    bool_not_u8_seq(a, out);
}

#[macerator::with_simd]
fn bool_not_u8_seq<S: Simd>(a: &[u8], out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(out.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for j in simd_len..len {
        out[j] = (a[j] == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_u8_par(a: &[u8], out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_not_u8_seq(&a[start..end], out_chunk);
        });
}

#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_inplace_u8_par(a);
        return;
    }

    bool_not_inplace_u8_seq(a);
}

#[macerator::with_simd]
fn bool_not_inplace_u8_seq<S: Simd>(a: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(a.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = (*v == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_inplace_u8_par(a: &mut [u8]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        bool_not_inplace_u8_seq(chunk);
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        add_inplace_f32(&mut a, &b);
        assert_eq!(a, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_sub_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        sub_inplace_f32(&mut a, &b);
        assert_eq!(a, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_mul_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        mul_inplace_f32(&mut a, &b);
        assert_eq!(a, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_div_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 4.0, 5.0, 8.0];
        div_inplace_f32(&mut a, &b);
        assert_eq!(a, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_cmp_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0f32, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let mut out = [0u8; 7];
        cmp_f32(&a, &b, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn test_cmp_ge_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 2.0, 2.0, 5.0];
        let mut out = [0u8; 4];
        cmp_f32(&a, &b, &mut out, CmpOp::Ge);
        assert_eq!(out, [0, 1, 1, 0]);
    }

    #[test]
    fn test_cmp_eq_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Eq);
        assert_eq!(out, [1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_cmp_ne_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Ne);
        assert_eq!(out, [0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_cmp_scalar_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0u8; 5];
        cmp_scalar_f32(&a, 3.0, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_bool_not_u8() {
        let a = [1u8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0];
        let mut out = [0u8; 18];
        bool_not_u8(&a, &mut out);
        let expected = [0u8, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_not_inplace_u8() {
        let mut a = [1u8, 0, 1, 0];
        bool_not_inplace_u8(&mut a);
        assert_eq!(a, [0, 1, 0, 1]);
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

    #[test]
    fn test_bool_and_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_and_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 0, 0, 0]);
    }

    #[test]
    fn test_bool_or_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_or_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 1, 1, 0]);
    }

    #[test]
    fn test_bool_xor_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_xor_inplace_u8(&mut a, &b);
        assert_eq!(a, [0, 1, 1, 0]);
    }
}

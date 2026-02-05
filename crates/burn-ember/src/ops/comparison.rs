//! Comparison operations returning boolean tensors.

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};
use bytemuck::Pod;

use crate::strided_index::StridedIter;
use crate::{EmberTensor, Layout};

#[cfg(feature = "simd")]
use crate::simd;

/// Comparison operation type for SIMD dispatch.
#[derive(Clone, Copy)]
pub enum CompareOp {
    Greater,
    GreaterEqual,
    Lower,
    LowerEqual,
    Equal,
    NotEqual,
}

/// Compare two tensors element-wise, returning a boolean tensor.
pub fn compare<F32Cmp, F64Cmp>(
    lhs: EmberTensor,
    rhs: EmberTensor,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
) -> EmberTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

    // Broadcast to same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_f32(lhs, &rhs, f32_cmp),
        DType::F64 => compare_typed(lhs, &rhs, f64_cmp),
        DType::F16 => compare_f16(lhs, &rhs, |a, b| f32_cmp(a.to_f32(), b.to_f32())),
        DType::BF16 => compare_bf16(lhs, &rhs, |a, b| f32_cmp(a.to_f32(), b.to_f32())),
        _ => panic!("compare: unsupported dtype {:?}", dtype),
    }
}

/// Specialized comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_f32<Cmp>(lhs: EmberTensor, rhs: &EmberTensor, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: both tensors contiguous
    if let (Some((l_start, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) && let Some(simd_op) = detect_cmp_op(&cmp)
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();

        let l_slice = &lhs_storage[l_start..l_end];
        let r_slice = &rhs_storage[r_start..r_end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_f32(l_slice, r_slice, &mut result, simd_op);

        return make_bool_tensor(result, shape);
    }

    // Optimized broadcast path for outer-product style broadcasting
    // Pattern: [N, 1] vs [1, M] -> [N, M] where one has stride 0 in inner dim
    if lhs.layout().num_dims() == 2
        && let Some(simd_op) = detect_cmp_op(&cmp)
        && let Some((result, shape)) = try_broadcast_cmp_f32(&lhs, rhs, simd_op)
    {
        return make_bool_tensor(result, shape);
    }

    // Fallback to generic path
    compare_typed(lhs, rhs, cmp)
}

/// Try optimized outer-product style broadcast comparison.
/// Returns Some((result, shape)) if the pattern matches.
#[cfg(feature = "simd")]
fn try_broadcast_cmp_f32(
    lhs: &EmberTensor,
    rhs: &EmberTensor,
    op: simd::CmpOp,
) -> Option<(Vec<u8>, Shape)> {
    let lhs_strides = lhs.layout().strides();
    let rhs_strides = rhs.layout().strides();
    let shape = lhs.layout().shape().clone();
    let [rows, cols] = shape.dims[..] else {
        return None;
    };

    // Pattern 1: lhs has stride 0 in dim 1 (column broadcast), rhs contiguous
    // lhs[i,j] = lhs_data[i*stride], rhs[i,j] = rhs_data[i*cols + j]
    if lhs_strides[1] == 0 && rhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let r_row_start = r_offset + row * cols;
            let r_slice = &rhs_storage[r_row_start..r_row_start + cols];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                r_slice,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    // Pattern 2: rhs has stride 0 in dim 0 (row broadcast), lhs contiguous
    // lhs[i,j] = lhs_data[i*cols + j], rhs[i,j] = rhs_data[j*stride]
    if rhs_strides[0] == 0 && lhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset();
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs values once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let l_row_start = l_offset + row * cols;
            let l_slice = &lhs_storage[l_row_start..l_row_start + cols];
            let out_start = row * cols;
            // Compare row with broadcast values
            for (j, (&lv, &rv)) in l_slice.iter().zip(rhs_row.iter()).enumerate() {
                result[out_start + j] = match op {
                    simd::CmpOp::Gt => (lv > rv) as u8,
                    simd::CmpOp::Ge => (lv >= rv) as u8,
                    simd::CmpOp::Lt => (lv < rv) as u8,
                    simd::CmpOp::Le => (lv <= rv) as u8,
                    simd::CmpOp::Eq => (lv == rv) as u8,
                    simd::CmpOp::Ne => (lv != rv) as u8,
                };
            }
        }
        return Some((result, shape));
    }

    // Pattern 3: Outer product - lhs stride 0 in dim 1, rhs stride 0 in dim 0
    // This is the [N,1] vs [1,M] case
    if lhs_strides[1] == 0 && rhs_strides[0] == 0 {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs row once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                &rhs_row,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    None
}

/// Swap comparison operation for reversed operand order.
#[cfg(feature = "simd")]
fn swap_cmp_op(op: simd::CmpOp) -> simd::CmpOp {
    match op {
        simd::CmpOp::Gt => simd::CmpOp::Lt, // a > b becomes b < a
        simd::CmpOp::Ge => simd::CmpOp::Le,
        simd::CmpOp::Lt => simd::CmpOp::Gt,
        simd::CmpOp::Le => simd::CmpOp::Ge,
        simd::CmpOp::Eq => simd::CmpOp::Eq, // symmetric
        simd::CmpOp::Ne => simd::CmpOp::Ne,
    }
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_f32<Cmp>(lhs: EmberTensor, rhs: &EmberTensor, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_typed(lhs, rhs, cmp)
}

/// Detect which comparison operation is being performed by testing sample values.
#[cfg(feature = "simd")]
fn detect_cmp_op<Cmp>(cmp: &Cmp) -> Option<simd::CmpOp>
where
    Cmp: Fn(f32, f32) -> bool,
{
    // Test with values that distinguish operations
    let gt = cmp(3.0, 2.0); // true for GT, GE
    let lt = cmp(2.0, 3.0); // true for LT, LE
    let eq = cmp(2.0, 2.0); // true for GE, LE, EQ

    match (gt, lt, eq) {
        (true, false, false) => Some(simd::CmpOp::Gt), // only 3>2 is true
        (true, false, true) => Some(simd::CmpOp::Ge),  // 3>=2 and 2>=2
        (false, true, false) => Some(simd::CmpOp::Lt), // only 2<3 is true
        (false, true, true) => Some(simd::CmpOp::Le),  // 2<3 and 2<=2
        (false, false, true) => Some(simd::CmpOp::Eq), // only 2==2 is true
        (true, true, false) => Some(simd::CmpOp::Ne),  // 3!=2 and 2!=3, but 2!=2 is false
        _ => None,
    }
}

/// Compare tensor with scalar, returning a boolean tensor.
pub fn compare_elem<F32Cmp, F64Cmp>(
    lhs: EmberTensor,
    rhs: f64,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
) -> EmberTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_elem_f32(lhs, rhs as f32, f32_cmp),
        DType::F64 => compare_elem_typed(lhs, rhs, f64_cmp),
        DType::F16 => compare_elem_f16(lhs, rhs as f32, |a, b| f32_cmp(a.to_f32(), b)),
        DType::BF16 => compare_elem_bf16(lhs, rhs as f32, |a, b| f32_cmp(a.to_f32(), b)),
        _ => panic!("compare_elem: unsupported dtype {:?}", dtype),
    }
}

/// Specialized scalar comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_elem_f32<Cmp>(lhs: EmberTensor, rhs: f32, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: tensor is contiguous
    if let Some((start, end)) = lhs.layout().contiguous_offsets()
        && let Some(simd_op) = detect_cmp_op(&cmp)
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let l_slice = &lhs_storage[start..end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_scalar_f32(l_slice, rhs, &mut result, simd_op);

        return make_bool_tensor(result, shape);
    }

    // Fallback to generic path
    compare_elem_typed(lhs, rhs, cmp)
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_elem_f32<Cmp>(lhs: EmberTensor, rhs: f32, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_elem_typed(lhs, rhs, cmp)
}

fn compare_typed<E, Cmp>(lhs: EmberTensor, rhs: &EmberTensor, cmp: Cmp) -> EmberTensor
where
    E: Element + Pod,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();
    let rhs_storage: &[E] = rhs.storage();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        // Fast path for 2D non-contiguous (common for transpose)
        _ if lhs.layout().num_dims() == 2 => {
            compare_2d_strided(lhs_storage, rhs_storage, lhs.layout(), rhs.layout(), cmp)
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape)
}

/// Fast 2D strided comparison using row-based iteration.
#[inline]
fn compare_2d_strided<E, Cmp>(
    lhs: &[E],
    rhs: &[E],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    cmp: Cmp,
) -> Vec<u8>
where
    E: Copy,
    Cmp: Fn(E, E) -> bool,
{
    let (rows, cols, l_row_stride, l_col_stride) = lhs_layout.as_2d_strides().unwrap();
    let (_, _, r_row_stride, r_col_stride) = rhs_layout.as_2d_strides().unwrap();
    let l_offset = lhs_layout.start_offset() as isize;
    let r_offset = rhs_layout.start_offset() as isize;

    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let l_row_start = l_offset + row as isize * l_row_stride;
        let r_row_start = r_offset + row as isize * r_row_stride;
        for col in 0..cols {
            let l_idx = (l_row_start + col as isize * l_col_stride) as usize;
            let r_idx = (r_row_start + col as isize * r_col_stride) as usize;
            result.push(cmp(lhs[l_idx], rhs[r_idx]) as u8);
        }
    }

    result
}

fn compare_f16<Cmp>(lhs: EmberTensor, rhs: &EmberTensor, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f16, f16) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[f16] = bytemuck::cast_slice(lhs.bytes());
    let rhs_storage: &[f16] = bytemuck::cast_slice(rhs.bytes());

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape)
}

fn compare_bf16<Cmp>(lhs: EmberTensor, rhs: &EmberTensor, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(bf16, bf16) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[bf16] = bytemuck::cast_slice(lhs.bytes());
    let rhs_storage: &[bf16] = bytemuck::cast_slice(rhs.bytes());

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape)
}

fn compare_elem_typed<E, Cmp>(lhs: EmberTensor, rhs: E, cmp: Cmp) -> EmberTensor
where
    E: Element + Pod + Copy,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape)
}

fn compare_elem_f16<Cmp>(lhs: EmberTensor, rhs: f32, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(f16, f32) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[f16] = bytemuck::cast_slice(lhs.bytes());

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape)
}

fn compare_elem_bf16<Cmp>(lhs: EmberTensor, rhs: f32, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(bf16, f32) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[bf16] = bytemuck::cast_slice(lhs.bytes());

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape)
}

fn make_bool_tensor(data: Vec<u8>, shape: Shape) -> EmberTensor {
    let bytes = Bytes::from_elems(data);
    EmberTensor::new(bytes, Layout::contiguous(shape), DType::Bool)
}

// Specific comparison functions

pub fn greater(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a > b, |a, b| a > b)
}

pub fn greater_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a > b, |a, b| a > b)
}

pub fn greater_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a >= b, |a, b| a >= b)
}

pub fn greater_equal_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a >= b, |a, b| a >= b)
}

pub fn lower(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a < b, |a, b| a < b)
}

pub fn lower_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a < b, |a, b| a < b)
}

pub fn lower_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a <= b, |a, b| a <= b)
}

pub fn lower_equal_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a <= b, |a, b| a <= b)
}

pub fn equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a == b, |a, b| a == b)
}

pub fn equal_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a == b, |a, b| a == b)
}

pub fn not_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare(lhs, rhs, |a, b| a != b, |a, b| a != b)
}

pub fn not_equal_elem(lhs: EmberTensor, rhs: f64) -> EmberTensor {
    compare_elem(lhs, rhs, |a, b| a != b, |a, b| a != b)
}

// Integer comparison functions

fn compare_int<Cmp>(lhs: EmberTensor, rhs: EmberTensor, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(i64, i64) -> bool,
{
    debug_assert_eq!(lhs.dtype(), DType::I64, "compare_int: expected I64 dtype");
    debug_assert_eq!(rhs.dtype(), DType::I64, "compare_int: expected I64 dtype");

    // Broadcast to same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[i64] = lhs.storage();
    let rhs_storage: &[i64] = rhs.storage();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape)
}

fn compare_int_elem<Cmp>(lhs: EmberTensor, rhs: i64, cmp: Cmp) -> EmberTensor
where
    Cmp: Fn(i64, i64) -> bool,
{
    debug_assert_eq!(
        lhs.dtype(),
        DType::I64,
        "compare_int_elem: expected I64 dtype"
    );

    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[i64] = lhs.storage();

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape)
}

pub fn int_greater(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a > b)
}

pub fn int_greater_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a > b)
}

pub fn int_greater_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a >= b)
}

pub fn int_greater_equal_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a >= b)
}

pub fn int_lower(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a < b)
}

pub fn int_lower_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a < b)
}

pub fn int_lower_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a <= b)
}

pub fn int_lower_equal_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a <= b)
}

pub fn int_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a == b)
}

pub fn int_equal_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a == b)
}

pub fn int_not_equal(lhs: EmberTensor, rhs: EmberTensor) -> EmberTensor {
    compare_int(lhs, rhs, |a, b| a != b)
}

pub fn int_not_equal_elem(lhs: EmberTensor, rhs: i64) -> EmberTensor {
    compare_int_elem(lhs, rhs, |a, b| a != b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_greater() {
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let rhs = EmberTensor::from_data(TensorData::new(vec![2.0f32, 2.0, 1.0], [3]));
        let result = greater(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1]); // 1>2=F, 2>2=F, 3>1=T
    }

    #[test]
    fn test_greater_elem() {
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let result = greater_elem(lhs, 2.0);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1]); // 1>2=F, 2>2=F, 3>2=T
    }

    #[test]
    fn test_equal() {
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let rhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 3.0, 3.0], [3]));
        let result = equal(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 1]); // 1==1=T, 2==3=F, 3==3=T
    }

    // === Non-contiguous tensor tests ===

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> EmberTensor {
        EmberTensor::from_data(TensorData::new(data, vec![rows, cols]))
    }

    #[test]
    fn test_greater_transposed() {
        // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]]
        // Compare with [[2, 2], [2, 2]]
        // [[1, 3], [2, 4]] > [[2, 2], [2, 2]] = [[F, T], [F, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = lhs.transpose(0, 1);
        assert!(!lhs.is_contiguous());

        let rhs = tensor_2d(vec![2.0, 2.0, 2.0, 2.0], 2, 2);
        let result = greater(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 0, 1]);
    }

    #[test]
    fn test_equal_flipped_1d() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Compare [4, 3, 2, 1] == [4, 2, 2, 1] = [T, F, T, T]
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = EmberTensor::from_data(TensorData::new(vec![4.0f32, 2.0, 2.0, 1.0], [4]));
        let result = equal(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 1, 1]);
    }

    #[test]
    fn test_lower_flipped_2d() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // [[3, 4], [1, 2]] < [[2, 5], [2, 1]] = [[F, T], [T, F]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = tensor_2d(vec![2.0, 5.0, 2.0, 1.0], 2, 2);
        let result = lower(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 1, 0]);
    }

    #[test]
    fn test_greater_elem_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // [4, 3, 2, 1] > 2.5 = [T, T, F, F]
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let result = greater_elem(lhs, 2.5);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 1, 0, 0]);
    }

    #[test]
    fn test_equal_both_transposed() {
        // Both tensors transposed
        // [[1, 2], [3, 4]]^T -> [[1, 3], [2, 4]]
        // [[1, 3], [2, 4]]^T -> [[1, 2], [3, 4]]
        // [[1, 3], [2, 4]] == [[1, 2], [3, 4]] = [[T, F], [F, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).transpose(0, 1);
        let rhs = tensor_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).transpose(0, 1);
        assert!(!lhs.is_contiguous());
        assert!(!rhs.is_contiguous());

        let result = equal(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 0, 1]);
    }

    #[test]
    fn test_not_equal_narrowed() {
        // [1, 2, 3, 4, 5, 6] narrowed to [2, 3, 4, 5]
        // [2, 3, 4, 5] != [2, 2, 4, 4] = [F, T, F, T]
        let lhs = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6]));
        let lhs = lhs.narrow(0, 1, 4);

        let rhs = EmberTensor::from_data(TensorData::new(vec![2.0f32, 2.0, 4.0, 4.0], [4]));
        let result = not_equal(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 0, 1]);
    }

    #[test]
    fn test_int_greater_flipped() {
        // Integer comparison with flipped tensor
        let lhs = EmberTensor::from_data(TensorData::new(vec![1i64, 2, 3, 4], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        // [4, 3, 2, 1] > [3, 3, 3, 3] = [T, F, F, F]
        let rhs = EmberTensor::from_data(TensorData::new(vec![3i64, 3, 3, 3], [4]));
        let result = int_greater(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 0, 0]);
    }

    #[test]
    fn test_lower_flipped_both_axes() {
        // [[1, 2], [3, 4]] flipped on both axes -> [[4, 3], [2, 1]]
        // [[4, 3], [2, 1]] < [[3, 3], [3, 3]] = [[F, F], [T, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = crate::ops::flip::flip(lhs, &[0, 1]);
        assert!(lhs.layout().strides()[0] < 0);
        assert!(lhs.layout().strides()[1] < 0);

        let rhs = tensor_2d(vec![3.0, 3.0, 3.0, 3.0], 2, 2);
        let result = lower(lhs, rhs);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1, 1]);
    }
}

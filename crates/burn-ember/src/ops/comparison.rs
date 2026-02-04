//! Comparison operations returning boolean tensors.

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};
use bytemuck::Pod;

use crate::strided_index::StridedIter;
use crate::{EmberTensor, Layout};

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
        DType::F32 => compare_typed(lhs, &rhs, f32_cmp),
        DType::F64 => compare_typed(lhs, &rhs, f64_cmp),
        DType::F16 => compare_f16(lhs, &rhs, |a, b| f32_cmp(a.to_f32(), b.to_f32())),
        DType::BF16 => compare_bf16(lhs, &rhs, |a, b| f32_cmp(a.to_f32(), b.to_f32())),
        _ => panic!("compare: unsupported dtype {:?}", dtype),
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
        DType::F32 => compare_elem_typed(lhs, rhs as f32, f32_cmp),
        DType::F64 => compare_elem_typed(lhs, rhs, f64_cmp),
        DType::F16 => compare_elem_f16(lhs, rhs as f32, |a, b| f32_cmp(a.to_f32(), b)),
        DType::BF16 => compare_elem_bf16(lhs, rhs as f32, |a, b| f32_cmp(a.to_f32(), b)),
        _ => panic!("compare_elem: unsupported dtype {:?}", dtype),
    }
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
}

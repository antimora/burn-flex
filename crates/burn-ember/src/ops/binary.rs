//! Binary tensor operations (add, sub, mul, div).

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::strided_index::StridedIter;
use crate::{EmberTensor, Layout};

/// Apply a binary operation element-wise to two tensors.
///
/// Requires tensors to have the same shape. Attempts in-place mutation
/// when lhs is contiguous; otherwise allocates a new tensor.
pub fn binary_op<F32Op, F64Op>(
    lhs: EmberTensor,
    rhs: EmberTensor,
    f32_op: F32Op,
    f64_op: F64Op,
) -> EmberTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    debug_assert_eq!(
        lhs.layout().shape(),
        rhs.layout().shape(),
        "binary_op: shape mismatch"
    );
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "binary_op: dtype mismatch");

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => binary_op_typed(lhs, &rhs, f32_op),
        DType::F64 => binary_op_typed(lhs, &rhs, f64_op),
        DType::F16 => binary_op_half(lhs, &rhs, |a, b| {
            f16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        DType::BF16 => binary_op_bhalf(lhs, &rhs, |a, b| {
            bf16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        _ => panic!("binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Binary operation with in-place optimization for Pod types.
fn binary_op_typed<E, Op>(mut lhs: EmberTensor, rhs: &EmberTensor, op: Op) -> EmberTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    let rhs_storage: &[E] = rhs.storage();

    // In-place fast path: lhs contiguous at offset 0, rhs contiguous
    if let (Some((0, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        let lhs_storage: &mut [E] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let dtype = lhs.dtype();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<E> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    make_tensor(result, shape, dtype)
}

/// Binary operation for f16.
fn binary_op_half<Op>(mut lhs: EmberTensor, rhs: &EmberTensor, op: Op) -> EmberTensor
where
    Op: Fn(f16, f16) -> f16,
{
    let rhs_storage: &[f16] = rhs.storage();

    // In-place fast path
    if let (Some((0, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        let lhs_storage: &mut [f16] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[f16] = lhs.storage();

    let result: Vec<f16> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    make_tensor(result, shape, DType::F16)
}

/// Binary operation for bf16.
fn binary_op_bhalf<Op>(mut lhs: EmberTensor, rhs: &EmberTensor, op: Op) -> EmberTensor
where
    Op: Fn(bf16, bf16) -> bf16,
{
    let rhs_storage: &[bf16] = rhs.storage();

    // In-place fast path
    if let (Some((0, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        let lhs_storage: &mut [bf16] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[bf16] = lhs.storage();

    let result: Vec<bf16> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                .collect()
        }
    };

    make_tensor(result, shape, DType::BF16)
}

/// Apply a scalar operation to each element of a tensor.
///
/// Attempts in-place mutation when tensor is contiguous at offset 0.
pub fn scalar_op<F32Op, F64Op>(
    tensor: EmberTensor,
    scalar: f64,
    f32_op: F32Op,
    f64_op: F64Op,
) -> EmberTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => scalar_op_typed(tensor, scalar as f32, f32_op),
        DType::F64 => scalar_op_typed(tensor, scalar, f64_op),
        DType::F16 => scalar_op_half(tensor, f16::from_f64(scalar), |a, b| {
            f16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        DType::BF16 => scalar_op_bhalf(tensor, bf16::from_f64(scalar), |a, b| {
            bf16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        _ => panic!("scalar_op: unsupported dtype {:?}", dtype),
    }
}

fn scalar_op_typed<E, Op>(mut tensor: EmberTensor, scalar: E, op: Op) -> EmberTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    // In-place fast path
    if let Some((0, end)) = tensor.layout().contiguous_offsets() {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let storage: &[E] = tensor.storage();

    let result: Vec<E> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, dtype)
}

fn scalar_op_half<Op>(mut tensor: EmberTensor, scalar: f16, op: Op) -> EmberTensor
where
    Op: Fn(f16, f16) -> f16,
{
    // In-place fast path
    if let Some((0, end)) = tensor.layout().contiguous_offsets() {
        let storage: &mut [f16] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let storage: &[f16] = tensor.storage();

    let result: Vec<f16> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, DType::F16)
}

fn scalar_op_bhalf<Op>(mut tensor: EmberTensor, scalar: bf16, op: Op) -> EmberTensor
where
    Op: Fn(bf16, bf16) -> bf16,
{
    // In-place fast path
    if let Some((0, end)) = tensor.layout().contiguous_offsets() {
        let storage: &mut [bf16] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let storage: &[bf16] = tensor.storage();

    let result: Vec<bf16> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, DType::BF16)
}

/// Helper to construct a tensor from result data.
fn make_tensor<E: bytemuck::Pod + Send + Sync>(
    data: Vec<E>,
    shape: Shape,
    dtype: DType,
) -> EmberTensor {
    let bytes = Bytes::from_elems(data);
    let layout = Layout::contiguous(shape);
    EmberTensor::new(bytes, layout, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::TensorData;

    // ===================
    // Binary ops: f32
    // ===================

    #[test]
    fn test_binary_add_contiguous_f32() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_sub_contiguous_f32() {
        let a =
            EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x - y, |x, y| x - y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![9.0, 18.0, 27.0, 36.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mul_contiguous_f32() {
        let a = EmberTensor::from_data(TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![2.0, 6.0, 12.0, 20.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_div_contiguous_f32() {
        let a =
            EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 5.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x / y, |x, y| x / y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![5.0, 5.0, 6.0, 5.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Binary ops: f64
    // ===================

    #[test]
    fn test_binary_add_contiguous_f64() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![5.0f64, 6.0, 7.0, 8.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f64> = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mul_contiguous_f64() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.5f64, 2.5, 3.5, 4.5], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![2.0f64, 2.0, 2.0, 2.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f64> = vec![3.0, 5.0, 7.0, 9.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    // ===================
    // Non-contiguous
    // ===================

    #[test]
    fn test_binary_mul_non_contiguous_transposed() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], vec![2, 2]));

        let a_t = a.transpose(0, 1);
        let b_t = b.transpose(0, 1);

        let result = binary_op(a_t, b_t, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        // a_t = [[1, 3], [2, 4]], b_t = [[2, 4], [3, 5]]
        // result = [[2, 12], [6, 20]]
        let expected: Vec<f32> = vec![2.0, 12.0, 6.0, 20.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_add_non_contiguous_narrowed() {
        // Create [4, 4] tensor and narrow to [2, 4]
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let a = EmberTensor::from_data(TensorData::new(data.clone(), vec![4, 4]));
        let b = EmberTensor::from_data(TensorData::new(data, vec![4, 4]));

        let a_narrow = a.narrow(0, 1, 2); // rows 1-2
        let b_narrow = b.narrow(0, 1, 2);

        let result = binary_op(a_narrow, b_narrow, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        // rows 1-2: [4,5,6,7], [8,9,10,11] doubled
        let expected: Vec<f32> = vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_mixed_contiguous_non_contiguous() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let b =
            EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));

        // a is contiguous, b is transposed (non-contiguous)
        let b_t = b.transpose(0, 1);

        let result = binary_op(a, b_t, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        // a = [[1,2], [3,4]], b_t = [[10,30], [20,40]]
        // result = [[11,32], [23,44]]
        let expected: Vec<f32> = vec![11.0, 32.0, 23.0, 44.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Scalar ops
    // ===================

    #[test]
    fn test_scalar_add_f32() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![11.0, 12.0, 13.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_sub_f32() {
        let a = EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0], vec![3]));
        let result = scalar_op(a, 5.0, |x, y| x - y, |x, y| x - y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![5.0, 15.0, 25.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_mul_f32() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let result = scalar_op(a, 3.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![3.0, 6.0, 9.0, 12.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_div_f32() {
        let a =
            EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
        let result = scalar_op(a, 10.0, |x, y| x / y, |x, y| x / y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_add_f64() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, 100.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f64> = vec![101.0, 102.0, 103.0];
        assert_eq!(data.as_slice::<f64>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_non_contiguous() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]));
        let a_t = a.transpose(0, 1);

        let result = scalar_op(a_t, 10.0, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        // a_t = [[1, 3], [2, 4]] + 10 = [[11, 13], [12, 14]]
        let expected: Vec<f32> = vec![11.0, 13.0, 12.0, 14.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ===================
    // Edge cases
    // ===================

    #[test]
    fn test_binary_single_element() {
        let a = EmberTensor::from_data(TensorData::new(vec![5.0f32], vec![1]));
        let b = EmberTensor::from_data(TensorData::new(vec![3.0f32], vec![1]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        assert_eq!(data.as_slice::<f32>().unwrap(), &[8.0f32]);
    }

    #[test]
    fn test_binary_1d_tensor() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]));
        let b = EmberTensor::from_data(TensorData::new(vec![5.0f32, 4.0, 3.0, 2.0, 1.0], vec![5]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![6.0, 6.0, 6.0, 6.0, 6.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_binary_3d_tensor() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32; 24], vec![2, 3, 4]));
        let b = EmberTensor::from_data(TensorData::new(vec![2.0f32; 24], vec![2, 3, 4]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![2.0; 24];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_single_element() {
        let a = EmberTensor::from_data(TensorData::new(vec![7.0f32], vec![1]));
        let result = scalar_op(a, 3.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        assert_eq!(data.as_slice::<f32>().unwrap(), &[21.0f32]);
    }

    #[test]
    fn test_binary_negative_values() {
        let a = EmberTensor::from_data(TensorData::new(vec![-1.0f32, -2.0, 3.0, 4.0], vec![2, 2]));
        let b = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, -3.0, -4.0], vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    #[test]
    fn test_scalar_negative_value() {
        let a = EmberTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], vec![3]));
        let result = scalar_op(a, -1.0, |x, y| x * y, |x, y| x * y);
        let data = result.into_data();

        let expected: Vec<f32> = vec![-1.0, -2.0, -3.0];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }
}

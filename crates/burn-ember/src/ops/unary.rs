//! Unary tensor operations (exp, log, sqrt, sin, cos, etc.).

use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{Bytes, bf16, f16};

use crate::layout::StridedBlocks;
use crate::{EmberTensor, Layout};

/// Apply a unary operation element-wise to a tensor.
pub fn unary_op<F32Op, F64Op>(tensor: EmberTensor, f32_op: F32Op, f64_op: F64Op) -> EmberTensor
where
    F32Op: Fn(f32) -> f32 + Copy,
    F64Op: Fn(f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => unary_op_typed(tensor, f32_op),
        DType::F64 => unary_op_typed(tensor, f64_op),
        DType::F16 => unary_op_f16(tensor, |x| f16::from_f32(f32_op(x.to_f32()))),
        DType::BF16 => unary_op_bf16(tensor, |x| bf16::from_f32(f32_op(x.to_f32()))),
        _ => panic!("unary_op: unsupported dtype {:?}", dtype),
    }
}

/// Generic unary operation for any element type.
fn unary_op_typed<E, Op>(mut tensor: EmberTensor, op: Op) -> EmberTensor
where
    E: burn_backend::Element + bytemuck::Pod,
    Op: Fn(E) -> E,
{
    let n = tensor.layout().num_elements();

    // In-place fast path: unique, contiguous tensor at offset 0
    if tensor.is_unique() && tensor.layout().is_contiguous() && tensor.layout().start_offset() == 0
    {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..n].iter_mut() {
            *x = op(*x);
        }
        return tensor;
    }

    // Allocating path for non-contiguous or offset tensors
    let layout = tensor.layout().clone();
    let src: &[E] = tensor.storage();

    // Check for negative strides (from flip operations)
    let has_negative_strides = layout.strides().iter().any(|&s| s < 0);

    // Fast path: storage exactly matches tensor view (covers transposed tensors)
    // Iterate in storage order (contiguous) and preserve original layout.
    // Only valid when all strides are positive.
    if !has_negative_strides && layout.start_offset() == 0 && src.len() == n {
        let result: Vec<E> = src.iter().map(|&x| op(x)).collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(bytes, layout, E::dtype());
    }

    // Fallback for negative strides: use StridedIter for correct element order
    if has_negative_strides {
        let result: Vec<E> = crate::strided_index::StridedIter::new(&layout)
            .map(|idx| op(src[idx]))
            .collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(
            bytes,
            Layout::contiguous(layout.shape().clone()),
            E::dtype(),
        );
    }

    // General path for views/slices with offset or extra storage
    let result = match layout.strided_blocks() {
        // Single contiguous block (with offset)
        StridedBlocks::Single { start, len } => {
            src[start..start + len].iter().map(|&x| op(x)).collect()
        }
        // Strided: iterate over contiguous blocks
        StridedBlocks::Multiple {
            block_len,
            num_blocks,
            ..
        } => {
            let blocks = layout.strided_blocks();
            let mut result = Vec::with_capacity(n);

            if block_len == 1 {
                for block_start in blocks.block_starts() {
                    result.push(op(src[block_start]));
                }
            } else {
                for block_start in blocks.block_starts() {
                    for i in 0..block_len {
                        result.push(op(src[block_start + i]));
                    }
                }
            }
            debug_assert_eq!(result.len(), num_blocks * block_len);
            result
        }
    };

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(layout.shape().clone()),
        E::dtype(),
    )
}

/// Unary operation for f16.
fn unary_op_f16<Op>(mut tensor: EmberTensor, op: Op) -> EmberTensor
where
    Op: Fn(f16) -> f16,
{
    let n = tensor.layout().num_elements();

    // In-place fast path: unique, contiguous tensor at offset 0
    if tensor.is_unique() && tensor.layout().is_contiguous() && tensor.layout().start_offset() == 0
    {
        let storage: &mut [f16] = tensor.storage_mut();
        for x in storage[..n].iter_mut() {
            *x = op(*x);
        }
        return tensor;
    }

    // Allocating path
    let layout = tensor.layout().clone();
    let src: &[f16] = bytemuck::cast_slice(tensor.bytes());

    // Check for negative strides (from flip operations)
    let has_negative_strides = layout.strides().iter().any(|&s| s < 0);

    // Fast path: storage exactly matches tensor view (only with positive strides)
    if !has_negative_strides && layout.start_offset() == 0 && src.len() == n {
        let result: Vec<f16> = src.iter().map(|&x| op(x)).collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(bytes, layout, DType::F16);
    }

    // Fallback for negative strides: use StridedIter for correct element order
    if has_negative_strides {
        let result: Vec<f16> = crate::strided_index::StridedIter::new(&layout)
            .map(|idx| op(src[idx]))
            .collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(
            bytes,
            Layout::contiguous(layout.shape().clone()),
            DType::F16,
        );
    }

    let result = match layout.strided_blocks() {
        StridedBlocks::Single { start, len } => {
            src[start..start + len].iter().map(|&x| op(x)).collect()
        }
        StridedBlocks::Multiple { block_len, .. } => {
            let blocks = layout.strided_blocks();
            let mut result = Vec::with_capacity(n);

            if block_len == 1 {
                for block_start in blocks.block_starts() {
                    result.push(op(src[block_start]));
                }
            } else {
                for block_start in blocks.block_starts() {
                    for i in 0..block_len {
                        result.push(op(src[block_start + i]));
                    }
                }
            }
            result
        }
    };

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(layout.shape().clone()),
        DType::F16,
    )
}

/// Unary operation for bf16.
fn unary_op_bf16<Op>(mut tensor: EmberTensor, op: Op) -> EmberTensor
where
    Op: Fn(bf16) -> bf16,
{
    let n = tensor.layout().num_elements();

    // In-place fast path: unique, contiguous tensor at offset 0
    if tensor.is_unique() && tensor.layout().is_contiguous() && tensor.layout().start_offset() == 0
    {
        let storage: &mut [bf16] = tensor.storage_mut();
        for x in storage[..n].iter_mut() {
            *x = op(*x);
        }
        return tensor;
    }

    // Allocating path
    let layout = tensor.layout().clone();
    let src: &[bf16] = bytemuck::cast_slice(tensor.bytes());

    // Check for negative strides (from flip operations)
    let has_negative_strides = layout.strides().iter().any(|&s| s < 0);

    // Fast path: storage exactly matches tensor view (only with positive strides)
    if !has_negative_strides && layout.start_offset() == 0 && src.len() == n {
        let result: Vec<bf16> = src.iter().map(|&x| op(x)).collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(bytes, layout, DType::BF16);
    }

    // Fallback for negative strides: use StridedIter for correct element order
    if has_negative_strides {
        let result: Vec<bf16> = crate::strided_index::StridedIter::new(&layout)
            .map(|idx| op(src[idx]))
            .collect();
        let bytes = Bytes::from_elems(result);
        return EmberTensor::new(
            bytes,
            Layout::contiguous(layout.shape().clone()),
            DType::BF16,
        );
    }

    let result = match layout.strided_blocks() {
        StridedBlocks::Single { start, len } => {
            src[start..start + len].iter().map(|&x| op(x)).collect()
        }
        StridedBlocks::Multiple { block_len, .. } => {
            let blocks = layout.strided_blocks();
            let mut result = Vec::with_capacity(n);

            if block_len == 1 {
                for block_start in blocks.block_starts() {
                    result.push(op(src[block_start]));
                }
            } else {
                for block_start in blocks.block_starts() {
                    for i in 0..block_len {
                        result.push(op(src[block_start + i]));
                    }
                }
            }
            result
        }
    };

    let bytes = Bytes::from_elems(result);
    EmberTensor::new(
        bytes,
        Layout::contiguous(layout.shape().clone()),
        DType::BF16,
    )
}

// ============================================================================
// Specific unary operations
// ============================================================================

/// Exponential: e^x
pub fn exp(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::exp, f64::exp)
}

/// Natural logarithm: ln(x)
pub fn log(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::ln, f64::ln)
}

/// Natural logarithm of (1 + x): ln(1 + x)
pub fn log1p(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::ln_1p, f64::ln_1p)
}

/// Square root
pub fn sqrt(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::sqrt, f64::sqrt)
}

/// Absolute value
pub fn abs(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::abs, f64::abs)
}

/// Reciprocal: 1/x
pub fn recip(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, |x| 1.0 / x, |x| 1.0 / x)
}

/// Cosine
pub fn cos(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::cos, f64::cos)
}

/// Sine
pub fn sin(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::sin, f64::sin)
}

/// Tangent
pub fn tan(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::tan, f64::tan)
}

/// Hyperbolic cosine
pub fn cosh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::cosh, f64::cosh)
}

/// Hyperbolic sine
pub fn sinh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::sinh, f64::sinh)
}

/// Hyperbolic tangent
pub fn tanh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::tanh, f64::tanh)
}

/// Inverse cosine (arccos)
pub fn acos(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::acos, f64::acos)
}

/// Inverse hyperbolic cosine
pub fn acosh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::acosh, f64::acosh)
}

/// Inverse sine (arcsin)
pub fn asin(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::asin, f64::asin)
}

/// Inverse hyperbolic sine
pub fn asinh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::asinh, f64::asinh)
}

/// Inverse tangent (arctan)
pub fn atan(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::atan, f64::atan)
}

/// Inverse hyperbolic tangent
pub fn atanh(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::atanh, f64::atanh)
}

/// Round to nearest integer
pub fn round(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::round, f64::round)
}

/// Floor (round down)
pub fn floor(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::floor, f64::floor)
}

/// Ceiling (round up)
pub fn ceil(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::ceil, f64::ceil)
}

/// Truncate (round towards zero)
pub fn trunc(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, f32::trunc, f64::trunc)
}

/// Error function
pub fn erf(tensor: EmberTensor) -> EmberTensor {
    unary_op(tensor, erf_f32, erf_f64)
}

// ============================================================================
// Error function implementation
// ============================================================================

/// Approximation of the error function for f32.
/// Uses the Horner form of the approximation from Abramowitz and Stegun.
fn erf_f32(x: f32) -> f32 {
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.3275911_f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximation of the error function for f64.
fn erf_f64(x: f64) -> f64 {
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    fn tensor_from_vec(data: Vec<f32>) -> EmberTensor {
        let shape = burn_std::Shape::from(vec![data.len()]);
        EmberTensor::from_data(TensorData::new(data, shape.dims.clone()))
    }

    fn assert_approx_eq(result: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < tol,
                "got {}, expected {}, diff {}",
                r,
                e,
                (r - e).abs()
            );
        }
    }

    #[test]
    fn test_exp() {
        let tensor = tensor_from_vec(vec![0.0, 1.0, 2.0]);
        let result = exp(tensor);
        let data: &[f32] = result.storage();
        assert_approx_eq(
            data,
            &[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            1e-5,
        );
    }

    #[test]
    fn test_log() {
        let tensor = tensor_from_vec(vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = log(tensor);
        let data: &[f32] = result.storage();
        assert_approx_eq(data, &[0.0, 1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_sqrt() {
        let tensor = tensor_from_vec(vec![0.0, 1.0, 4.0, 9.0]);
        let result = sqrt(tensor);
        let data: &[f32] = result.storage();
        assert_approx_eq(data, &[0.0, 1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_abs() {
        let tensor = tensor_from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
        let result = abs(tensor);
        let data: &[f32] = result.storage();
        assert_approx_eq(data, &[3.0, 1.0, 0.0, 1.0, 3.0], 1e-5);
    }

    #[test]
    fn test_sin_cos() {
        let tensor = tensor_from_vec(vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]);

        let sin_result = sin(tensor.clone());
        let sin_data: &[f32] = sin_result.storage();
        assert_approx_eq(sin_data, &[0.0, 1.0, 0.0], 1e-5);

        let cos_result = cos(tensor);
        let cos_data: &[f32] = cos_result.storage();
        assert_approx_eq(cos_data, &[1.0, 0.0, -1.0], 1e-5);
    }

    #[test]
    fn test_tanh() {
        let tensor = tensor_from_vec(vec![-2.0, 0.0, 2.0]);
        let result = tanh(tensor);
        let data: &[f32] = result.storage();
        let expected: Vec<f32> = vec![-2.0f32.tanh(), 0.0, 2.0f32.tanh()];
        assert_approx_eq(data, &expected, 1e-5);
    }

    #[test]
    fn test_round_floor_ceil() {
        let tensor = tensor_from_vec(vec![-1.5, -0.5, 0.5, 1.5]);

        // Rust uses "round half away from zero": -0.5 -> -1, 0.5 -> 1
        let round_result = round(tensor.clone());
        let round_data: &[f32] = round_result.storage();
        assert_approx_eq(round_data, &[-2.0, -1.0, 1.0, 2.0], 1e-5);

        let floor_result = floor(tensor.clone());
        let floor_data: &[f32] = floor_result.storage();
        assert_approx_eq(floor_data, &[-2.0, -1.0, 0.0, 1.0], 1e-5);

        let ceil_result = ceil(tensor);
        let ceil_data: &[f32] = ceil_result.storage();
        assert_approx_eq(ceil_data, &[-1.0, 0.0, 1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_erf() {
        let tensor = tensor_from_vec(vec![0.0, 0.5, 1.0, 2.0]);
        let result = erf(tensor);
        let data: &[f32] = result.storage();
        // Expected values from standard erf tables
        assert_approx_eq(data, &[0.0, 0.5205, 0.8427, 0.9953], 1e-3);
    }

    // === Non-contiguous tensor tests ===

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> EmberTensor {
        EmberTensor::from_data(TensorData::new(data, vec![rows, cols]))
    }

    #[test]
    fn test_exp_transposed() {
        // [[0, 1], [2, 3]] transposed -> [[0, 2], [1, 3]]
        // Storage order: [0, 1, 2, 3], but logical order after transpose: [0, 2, 1, 3]
        let tensor = tensor_2d(vec![0.0, 1.0, 2.0, 3.0], 2, 2);
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());

        let result = exp(transposed);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // exp([0, 2, 1, 3]) = [1.0, e^2, e, e^3]
        let e = std::f32::consts::E;
        assert_approx_eq(&data, &[1.0, e * e, e, e * e * e], 1e-5);
    }

    #[test]
    fn test_sqrt_narrowed() {
        // Original: [1, 4, 9, 16, 25, 36] shape [6]
        // Narrow to middle 4 elements: [4, 9, 16, 25]
        let tensor = tensor_from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
        let narrowed = tensor.narrow(0, 1, 4);
        assert!(!narrowed.is_contiguous() || narrowed.layout().start_offset() != 0);

        let result = sqrt(narrowed);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_approx_eq(&data, &[2.0, 3.0, 4.0, 5.0], 1e-5);
    }

    #[test]
    fn test_abs_flipped() {
        // Test with negative strides from flip
        // [1, -2, 3, -4] flipped -> [-4, 3, -2, 1]
        let tensor = tensor_from_vec(vec![1.0, -2.0, 3.0, -4.0]);
        let flipped = crate::ops::flip::flip(tensor, &[0]);

        // Verify it's using negative strides (zero-copy)
        assert!(flipped.layout().strides()[0] < 0);

        let result = abs(flipped);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // abs([-4, 3, -2, 1]) = [4, 3, 2, 1]
        assert_approx_eq(&data, &[4.0, 3.0, 2.0, 1.0], 1e-5);
    }

    #[test]
    fn test_sqrt_flipped_2d() {
        // [[1, 4], [9, 16]] with axis 0 flipped -> [[9, 16], [1, 4]]
        // sqrt of that -> [[3, 4], [1, 2]]
        let tensor = tensor_2d(vec![1.0, 4.0, 9.0, 16.0], 2, 2);
        let flipped = crate::ops::flip::flip(tensor, &[0]);

        // Axis 0 stride should be negative
        assert!(flipped.layout().strides()[0] < 0);

        let result = sqrt(flipped);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // sqrt([[9, 16], [1, 4]]) = [[3, 4], [1, 2]]
        assert_approx_eq(&data, &[3.0, 4.0, 1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_cos_flipped_axis1() {
        // [[0, pi], [pi/2, 3pi/2]] with axis 1 flipped -> [[pi, 0], [3pi/2, pi/2]]
        // cos of that -> [[-1, 1], [0, 0]]
        use std::f32::consts::{FRAC_PI_2, PI};
        let tensor = tensor_2d(vec![0.0, PI, FRAC_PI_2, 3.0 * FRAC_PI_2], 2, 2);
        let flipped = crate::ops::flip::flip(tensor, &[1]);

        // Axis 1 stride should be negative
        assert!(flipped.layout().strides()[1] < 0);

        let result = cos(flipped);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // cos([[pi, 0], [3pi/2, pi/2]]) = [[-1, 1], [0, 0]]
        assert_approx_eq(&data, &[-1.0, 1.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn test_log_3d_transposed() {
        // 3D tensor with permuted dimensions
        // Shape [2, 2, 2] -> permute to [2, 2, 2] with different strides
        let e = std::f32::consts::E;
        let data = vec![1.0, e, e * e, e * e * e, 1.0, e, e * e, e * e * e];
        let tensor = EmberTensor::from_data(TensorData::new(data, vec![2, 2, 2]));
        let permuted = tensor.permute(&[2, 0, 1]); // Swap dimensions around
        assert!(!permuted.is_contiguous());

        let result = log(permuted);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // All values should be 0, 1, 2, or 3 depending on permutation
        for &v in &out {
            assert!(v >= -0.01 && v <= 3.01, "unexpected log value: {}", v);
        }
    }
}

//! Fused scaled dot-product attention.
//!
//! Computes: softmax(Q @ K^T * scale + bias) @ V
//!
//! The implementation fuses scale, softcap, masking, bias, and softmax into a
//! single pass over the attention scores matrix, reducing intermediate tensor
//! allocations from ~12 (in the generic fallback) to 3.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::ops::AttentionModuleOptions;
use burn_backend::DType;
use burn_std::Bytes;
use bytemuck::Pod;
use num_traits::Float;

use crate::{FlexTensor, Layout};

/// Dispatch attention by dtype.
pub fn attention(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
    let dtype = query.dtype();
    debug_assert_eq!(key.dtype(), dtype, "attention: key dtype mismatch");
    debug_assert_eq!(value.dtype(), dtype, "attention: value dtype mismatch");
    if let Some(ref b) = attn_bias {
        debug_assert_eq!(b.dtype(), dtype, "attention: attn_bias dtype mismatch");
    }

    match dtype {
        DType::F32 => attention_impl::<f32>(query, key, value, mask, attn_bias, options),
        DType::F64 => attention_impl::<f64>(query, key, value, mask, attn_bias, options),
        DType::F16 => {
            use burn_std::f16;
            let result = attention_impl::<f32>(
                cast_to_f32(query, f16::to_f32),
                cast_to_f32(key, f16::to_f32),
                cast_to_f32(value, f16::to_f32),
                mask,
                attn_bias.map(|b| cast_to_f32(b, f16::to_f32)),
                options,
            );
            cast_from_f32(result, f16::from_f32)
        }
        DType::BF16 => {
            use burn_std::bf16;
            let result = attention_impl::<f32>(
                cast_to_f32(query, bf16::to_f32),
                cast_to_f32(key, bf16::to_f32),
                cast_to_f32(value, bf16::to_f32),
                mask,
                attn_bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                options,
            );
            cast_from_f32(result, bf16::from_f32)
        }
        dtype => panic!("attention: unsupported dtype {:?}", dtype),
    }
}

fn cast_to_f32<E: burn_backend::Element + Pod + Copy>(
    tensor: FlexTensor,
    to_f32: fn(E) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[E] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|&v| to_f32(v)).collect();
    FlexTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(shape),
        DType::F32,
    )
}

fn cast_from_f32<E: burn_backend::Element + Pod + Copy>(
    tensor: FlexTensor,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[f32] = tensor.storage();
    let half_data: Vec<E> = data.iter().map(|&v| from_f32(v)).collect();
    FlexTensor::new(
        Bytes::from_elems(half_data),
        Layout::contiguous(shape),
        E::dtype(),
    )
}

/// Core attention implementation, generic over float type.
///
/// Input shapes (all 4D):
///   query: \[batch, heads, seq_q, head_dim\]
///   key:   \[batch, heads, seq_k, head_dim\]
///   value: \[batch, heads, seq_k, val_dim\]
///   mask:  \[batch, heads, seq_q, seq_k\] (optional, bool stored as u8)
///   attn_bias: \[batch, heads, seq_q, seq_k\] (optional)
///
/// Output: \[batch, heads, seq_q, val_dim\]
fn attention_impl<T>(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor
where
    T: Float + Pod + Copy + burn_backend::Element,
    T: core::ops::AddAssign,
{
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    let q_shape = query.layout().shape();
    let ndims = q_shape.num_dims();
    let head_dim = q_shape[ndims - 1];

    // Step 1: Q @ K^T via existing matmul
    let transposed_key = key.transpose(ndims - 2, ndims - 1);
    let scores = crate::ops::matmul::matmul(query, transposed_key);

    // Step 2: Fused scale + softcap + mask + bias + softmax
    let weights = fused_softmax::<T>(scores, mask.as_ref(), attn_bias.as_ref(), &options, head_dim);

    // Step 3: weights @ V
    crate::ops::matmul::matmul(weights, value)
}

/// Fused softmax with scale, softcap, masking, and bias.
///
/// Operates on scores of shape \[batch, heads, seq_q, seq_k\].
/// Applies all transformations and softmax row-wise along the last dimension
/// in a single allocation.
///
/// Two-pass approach per row:
///   Pass 1: apply scale/softcap/mask/bias, find row max
///   Pass 2: exp(x - max), sum, normalize
fn fused_softmax<T>(
    scores: FlexTensor,
    mask: Option<&FlexTensor>,
    attn_bias: Option<&FlexTensor>,
    options: &AttentionModuleOptions,
    head_dim: usize,
) -> FlexTensor
where
    T: Float + Pod + Copy + burn_backend::Element,
    T: core::ops::AddAssign,
{
    let scores = scores.to_contiguous();
    let shape = scores.layout().shape().clone();
    let ndims = shape.num_dims();
    assert!(ndims >= 2, "scores must be at least 2D");

    let seq_q = shape[ndims - 2];
    let seq_k = shape[ndims - 1];
    let num_rows_total: usize = shape[..ndims - 2].iter().product::<usize>() * seq_q;

    let scores_data: &[T] = scores.storage();

    let scores_numel = scores_data.len();

    let mask_tensor = mask.map(|m| {
        let m = m.to_contiguous();
        debug_assert_eq!(
            m.layout().num_elements(),
            scores_numel,
            "attention: mask shape must match scores shape"
        );
        m
    });
    let mask_data: Option<&[u8]> = mask_tensor.as_ref().map(|m| m.bytes());

    let bias_tensor = attn_bias.map(|b| {
        let b = b.to_contiguous();
        debug_assert_eq!(
            b.layout().num_elements(),
            scores_numel,
            "attention: attn_bias shape must match scores shape"
        );
        b
    });
    let bias_data: Option<&[T]> = bias_tensor.as_ref().map(|b| b.storage());

    let scale = T::from(
        options
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()),
    )
    .unwrap();

    let softcap: Option<T> = options.softcap.map(|s| T::from(s).unwrap());
    let neg_inf = T::neg_infinity();

    let causal_offset = if options.is_causal {
        Some(seq_k as isize - seq_q as isize)
    } else {
        None
    };

    let mut output = vec![T::zero(); scores_data.len()];

    for row_idx in 0..num_rows_total {
        let q_pos = row_idx % seq_q;
        let row_start = row_idx * seq_k;
        let scores_row = &scores_data[row_start..row_start + seq_k];
        let out_row = &mut output[row_start..row_start + seq_k];

        let mask_row = mask_data.map(|m| &m[row_start..row_start + seq_k]);
        let bias_row = bias_data.map(|b| &b[row_start..row_start + seq_k]);

        fused_softmax_row(
            scores_row,
            out_row,
            mask_row,
            bias_row,
            scale,
            softcap,
            neg_inf,
            causal_offset,
            q_pos,
        );
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        T::dtype(),
    )
}

/// Process a single row of attention scores through the fused pipeline.
///
/// Applies (in logical order):
///   1. Scale: x *= scale
///   2. Softcap (optional): x = softcap * tanh(x / softcap)
///   3. Bool mask: x = -inf where mask\[k\] != 0
///   4. Causal mask: x = -inf where k > q_pos + causal_offset
///   5. Additive bias: x += bias\[k\]
///   6. Softmax: exp(x - max) / sum(exp(x - max))
#[inline]
fn fused_softmax_row<T>(
    scores: &[T],
    output: &mut [T],
    mask: Option<&[u8]>,
    bias: Option<&[T]>,
    scale: T,
    softcap: Option<T>,
    neg_inf: T,
    causal_offset: Option<isize>,
    q_pos: usize,
) where
    T: Float + Copy + core::ops::AddAssign,
{
    let seq_k = scores.len();

    // Pass 1: apply transformations and find row max
    let mut row_max = neg_inf;

    for k in 0..seq_k {
        let mut val = scores[k] * scale;

        if let Some(cap) = softcap {
            val = cap * (val / cap).tanh();
        }

        if let Some(m) = mask {
            if m[k] != 0 {
                val = neg_inf;
            }
        }

        if let Some(offset) = causal_offset {
            if (k as isize) > (q_pos as isize) + offset {
                val = neg_inf;
            }
        }

        if let Some(b) = bias {
            val = val + b[k];
        }

        output[k] = val;
        if val > row_max {
            row_max = val;
        }
    }

    // Handle all-masked rows: if every position was masked, row_max stays -inf
    // and exp(-inf - -inf) = exp(NaN) = NaN. Output zeros instead.
    if row_max == neg_inf {
        for k in 0..seq_k {
            output[k] = T::zero();
        }
        return;
    }

    // Pass 2: exp(x - max) and sum
    let mut sum = T::zero();
    for k in 0..seq_k {
        let e = (output[k] - row_max).exp();
        output[k] = e;
        sum += e;
    }

    // Normalize
    let inv_sum = T::one() / sum;
    for k in 0..seq_k {
        output[k] = output[k] * inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use burn_backend::ops::AttentionModuleOptions;
    use burn_tensor::{Tensor, TensorData};

    use crate::Flex;

    /// Helper: create Q/K/V for single-batch, single-head attention.
    fn make_qkv(
        q: &[&[f32]],
        k: &[&[f32]],
        v: &[&[f32]],
    ) -> (Tensor<Flex, 4>, Tensor<Flex, 4>, Tensor<Flex, 4>) {
        let seq_q = q.len();
        let seq_k = k.len();
        let head_dim = q[0].len();
        let val_dim = v[0].len();

        let q_flat: Vec<f32> = q.iter().flat_map(|r| r.iter().copied()).collect();
        let k_flat: Vec<f32> = k.iter().flat_map(|r| r.iter().copied()).collect();
        let v_flat: Vec<f32> = v.iter().flat_map(|r| r.iter().copied()).collect();

        let dev = Default::default();
        let qt = Tensor::from_data(TensorData::new(q_flat, [1, 1, seq_q, head_dim]), &dev);
        let kt = Tensor::from_data(TensorData::new(k_flat, [1, 1, seq_k, head_dim]), &dev);
        let vt = Tensor::from_data(TensorData::new(v_flat, [1, 1, seq_k, val_dim]), &dev);
        (qt, kt, vt)
    }

    #[test]
    fn test_basic() {
        // Q=K=identity so each query attends most to itself
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        let result = burn_tensor::module::attention(q, k, v, None, None, Default::default());
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        // softmax([1/sqrt(2), 0]) = [0.670, 0.330]
        // row 0: 0.670*10 + 0.330*20 = 13.30
        // row 1: 0.330*10 + 0.670*20 = 16.70
        assert_eq!(data.len(), 2);
        assert!((data[0] - 13.30).abs() < 0.1, "got {}", data[0]);
        assert!((data[1] - 16.70).abs() < 0.1, "got {}", data[1]);
    }

    #[test]
    fn test_causal_mask() {
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        let opts = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let result = burn_tensor::module::attention(q, k, v, None, None, opts);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        // Row 0: can only see position 0, output = V[0] = 10.0
        assert!((data[0] - 10.0).abs() < 1e-5, "got {}", data[0]);
        // Row 1: sees both positions (same as non-causal)
        assert!((data[1] - 16.70).abs() < 0.1, "got {}", data[1]);
    }

    #[test]
    fn test_bool_mask() {
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        let dev = Default::default();
        use burn_tensor::Bool;
        let mask: Tensor<Flex, 4, Bool> = Tensor::from_data(
            TensorData::from([[[[true, false], [true, false]]]]),
            &dev,
        );

        let result =
            burn_tensor::module::attention(q, k, v, Some(mask), None, Default::default());
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        // Position 0 masked for all queries, output = V[1] = 20.0
        assert!((data[0] - 20.0).abs() < 1e-4, "got {}", data[0]);
        assert!((data[1] - 20.0).abs() < 1e-4, "got {}", data[1]);
    }

    #[test]
    fn test_additive_bias() {
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        let dev = Default::default();
        // Large bias toward position 1
        let bias: Tensor<Flex, 4> = Tensor::from_data(
            TensorData::new(vec![0.0f32, 100.0, 0.0, 100.0], [1, 1, 2, 2]),
            &dev,
        );

        let result =
            burn_tensor::module::attention(q, k, v, None, Some(bias), Default::default());
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        // Output ~ V[1] = 20.0
        assert!((data[0] - 20.0).abs() < 0.1, "got {}", data[0]);
        assert!((data[1] - 20.0).abs() < 0.1, "got {}", data[1]);
    }

    #[test]
    fn test_custom_scale() {
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        // Very large scale saturates softmax
        let opts = AttentionModuleOptions {
            scale: Some(100.0),
            ..Default::default()
        };
        let result = burn_tensor::module::attention(q, k, v, None, None, opts);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        assert!((data[0] - 10.0).abs() < 0.1, "got {}", data[0]);
        assert!((data[1] - 20.0).abs() < 0.1, "got {}", data[1]);
    }

    #[test]
    fn test_softcap() {
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[10.0], &[20.0]],
        );

        // Softcap squishes scores toward uniform, output ~ 15
        let opts = AttentionModuleOptions {
            softcap: Some(0.1),
            ..Default::default()
        };
        let result = burn_tensor::module::attention(q, k, v, None, None, opts);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        assert!((data[0] - 15.0).abs() < 0.5, "got {}", data[0]);
        assert!((data[1] - 15.0).abs() < 0.5, "got {}", data[1]);
    }

    #[test]
    fn test_cross_attention() {
        // seq_q=2, seq_k=3, head_dim=2, val_dim=1
        let (q, k, v) = make_qkv(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            &[&[1.0, 0.0], &[0.0, 1.0], &[0.5, 0.5]],
            &[&[10.0], &[20.0], &[30.0]],
        );

        let result = burn_tensor::module::attention(q, k, v, None, None, Default::default());
        let data: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(data.len(), 2);
        // Each output is a weighted combination of V[0..3]
        for &val in &data {
            assert!(val >= 9.0 && val <= 31.0, "unexpected value {val}");
        }
    }

    #[test]
    fn test_causal_cross_attention() {
        // seq_q=2, seq_k=4: causal mask aligns at bottom-right
        let dev = Default::default();
        let q: Tensor<Flex, 4> = Tensor::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [1, 1, 2, 2]),
            &dev,
        );
        let k: Tensor<Flex, 4> = Tensor::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5], [1, 1, 4, 2]),
            &dev,
        );
        let v: Tensor<Flex, 4> = Tensor::from_data(
            TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [1, 1, 4, 1]),
            &dev,
        );

        let opts = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let result_causal =
            burn_tensor::module::attention(q.clone(), k.clone(), v.clone(), None, None, opts);
        let data_causal: Vec<f32> = result_causal.into_data().to_vec().unwrap();

        let result_full =
            burn_tensor::module::attention(q, k, v, None, None, Default::default());
        let data_full: Vec<f32> = result_full.into_data().to_vec().unwrap();

        // With causal offset = seq_k - seq_q = 2:
        // Row 0 (q_pos=0): can attend to k=0,1,2 but NOT k=3 (v=40.0)
        // Row 1 (q_pos=1): can attend to all 4 positions
        assert_eq!(data_causal.len(), 2);

        // Causal hides v=40.0 from first query, so output must be less than non-causal
        assert!(
            data_causal[0] < data_full[0],
            "expected causal[0] < full[0], got {} vs {}",
            data_causal[0],
            data_full[0]
        );

        // Second query sees all positions in both cases
        assert!(
            (data_causal[1] - data_full[1]).abs() < 1e-5,
            "expected causal[1] ~= full[1], got {} vs {}",
            data_causal[1],
            data_full[1]
        );
    }
}

//! Activation function operations for the Flex backend.
//!
//! Each activation is implemented as a single-pass unary operation,
//! replacing the default multi-op compositions from Burn's trait defaults.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::Scalar;
use burn_backend::ops::ActivationOps;
use burn_backend::tensor::FloatTensor;
use burn_backend::{DType, TensorMetadata};
use burn_std::{Bytes, bf16, f16};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::ToPrimitive;

use crate::ops::binary::binary_op;
use crate::ops::unary::unary_op;
use crate::{Flex, FlexTensor, Layout};

impl ActivationOps<Flex> for Flex {
    fn relu(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary_op(tensor, |x: f32| x.max(0.0), |x: f64| x.max(0.0))
    }

    fn relu_backward(output: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // grad * (output > 0): zero the gradient where output was zero
        binary_op(
            output,
            grad,
            |out: f32, g| if out > 0.0 { g } else { 0.0 },
            |out: f64, g| if out > 0.0 { g } else { 0.0 },
            None,
        )
    }

    fn leaky_relu(tensor: FloatTensor<Flex>, negative_slope: Scalar) -> FloatTensor<Flex> {
        let ns32 = negative_slope.to_f32().unwrap();
        let ns64 = negative_slope.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| if x >= 0.0 { x } else { ns32 * x },
            move |x: f64| if x >= 0.0 { x } else { ns64 * x },
        )
    }

    fn prelu(tensor: FloatTensor<Flex>, alpha: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // x if x >= 0, alpha * x otherwise
        binary_op(
            tensor,
            alpha,
            |x: f32, a| if x >= 0.0 { x } else { a * x },
            |x: f64, a| if x >= 0.0 { x } else { a * x },
            None,
        )
    }

    fn gelu(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        unary_op(
            tensor,
            move |x: f32| 0.5 * x * (1.0 + erf_f32(x / sqrt2_f32)),
            move |x: f64| 0.5 * x * (1.0 + erf_f64(x / sqrt2_f64)),
        )
    }

    fn gelu_backward(x: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // d/dx[gelu(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        use crate::ops::unary::{erf_f32, erf_f64};
        let sqrt2_f32: f32 = core::f32::consts::SQRT_2;
        let sqrt2_f64: f64 = core::f64::consts::SQRT_2;
        let inv_sqrt_2pi_f32: f32 = 1.0 / (2.0 * core::f32::consts::PI).sqrt();
        let inv_sqrt_2pi_f64: f64 = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
        binary_op(
            x,
            grad,
            move |x: f32, g| {
                let cdf = 0.5 * (1.0 + erf_f32(x / sqrt2_f32));
                let pdf = inv_sqrt_2pi_f32 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            move |x: f64, g| {
                let cdf = 0.5 * (1.0 + erf_f64(x / sqrt2_f64));
                let pdf = inv_sqrt_2pi_f64 * (-0.5 * x * x).exp();
                g * (cdf + x * pdf)
            },
            None,
        )
    }

    fn sigmoid(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary_op(tensor, sigmoid_f32, sigmoid_f64)
    }

    fn sigmoid_backward(output: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // grad * output * (1 - output)
        binary_op(
            output,
            grad,
            |s: f32, g| g * s * (1.0 - s),
            |s: f64, g| g * s * (1.0 - s),
            None,
        )
    }

    fn hard_sigmoid(tensor: FloatTensor<Flex>, alpha: Scalar, beta: Scalar) -> FloatTensor<Flex> {
        let alpha32 = alpha.to_f32().unwrap();
        let beta32 = beta.to_f32().unwrap();
        let alpha64 = alpha.to_f64().unwrap();
        let beta64 = beta.to_f64().unwrap();
        unary_op(
            tensor,
            move |x: f32| (alpha32 * x + beta32).clamp(0.0, 1.0),
            move |x: f64| (alpha64 * x + beta64).clamp(0.0, 1.0),
        )
    }

    fn log_sigmoid(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // Numerically stable: -softplus(-x) = -log(1 + exp(-x))
        // For x >= 0: -log(1 + exp(-x))  (standard form, exp(-x) is small)
        // For x < 0: x - log(1 + exp(x))  (avoids exp of large positive)
        unary_op(
            tensor,
            |x: f32| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
            |x: f64| {
                if x >= 0.0 {
                    -((-x).exp().ln_1p())
                } else {
                    x - x.exp().ln_1p()
                }
            },
        )
    }

    fn log_sigmoid_backward(x: FloatTensor<Flex>, grad: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // d/dx[log_sigmoid(x)] = sigmoid(-x) * (-1) * (-1) = 1 - sigmoid(x) = sigmoid(-x)
        // So: grad * sigmoid(-x)
        binary_op(
            x,
            grad,
            |x: f32, g| g * sigmoid_f32(-x),
            |x: f64, g| g * sigmoid_f64(-x),
            None,
        )
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ============================================================================
// Fused softmax
// ============================================================================
//
// `burn-backend::ActivationOps` does not expose a `softmax` trait method (as
// of burn 0.21), so `burn_tensor::activation::softmax` decomposes into 5
// primitive tensor ops: `max_dim`, `sub` (broadcast), `exp`, `sum_dim`,
// `div` (broadcast). Each is a separate full-tensor pass with intermediate
// allocations — for attention scores ([heads, seq, seq]) this runs ~30x
// slower than candle's fused `softmax_last_dim`.
//
// Until the upstream burn API grows a `softmax` hook in `ActivationOps`,
// we expose this as a standalone `pub fn` on burn-flex. Users who want the
// fast path call `burn_flex::softmax(...)` directly; the upstream
// `activation::softmax` path still works (slowly).
//
// The implementation uses the standard 3-pass row-wise algorithm (max,
// exp+sum, normalize) which keeps each row cache-hot across the three
// passes. Parallelized over rows with rayon. Currently supports softmax
// along the last dim only; other axes panic (wav2vec2 attention only needs
// last-axis softmax, and permute+softmax+permute is a straightforward
// extension for the future).

/// Fused softmax along `dim`, avoiding the 5-op decomposition used by
/// `burn_tensor::activation::softmax`.
///
/// Single pass per row (reduce max, then exp+sum in one sweep, then
/// normalize in one sweep), with each row staying cache-hot. Rows are
/// processed in parallel via rayon.
///
/// Currently optimized only for the last axis of the tensor. Softmax along
/// other axes panics; callers should permute first or fall back to
/// `burn_tensor::activation::softmax`.
pub fn softmax(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
    let rank = tensor.shape().num_dims();
    assert!(
        dim < rank,
        "softmax dim {} out of range for rank {}",
        dim,
        rank
    );
    assert!(
        dim == rank - 1,
        "burn_flex::softmax currently only supports softmax along the last axis \
         (got dim={} for rank {}). Permute the tensor or fall back to \
         burn_tensor::activation::softmax for other axes.",
        dim,
        rank
    );

    let tensor = tensor.to_contiguous();
    match tensor.dtype() {
        DType::F32 => softmax_last_f32(tensor),
        DType::F64 => softmax_last_f64(tensor),
        DType::F16 => softmax_last_f16(tensor),
        DType::BF16 => softmax_last_bf16(tensor),
        dtype => panic!("softmax: unsupported dtype {:?}", dtype),
    }
}

fn softmax_last_f32(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape().clone();
    // Shape derefs to &[usize], so we can use slice methods directly.
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[f32] = tensor.storage();
    let mut output: Vec<f32> = vec![0.0f32; input.len()];

    // Row-parallel via rayon. Each task owns a contiguous range of rows
    // and calls the SIMD-specialized sweep once, so macerator's dispatch
    // is amortized over the whole range rather than paid per row.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        // Keep chunks coarse enough (~64 rows) so each rayon task does
        // real work and the dispatch shim runs once per task.
        const ROWS_PER_TASK: usize = 64;
        let chunk_bytes = ROWS_PER_TASK * last;
        output
            .par_chunks_mut(chunk_bytes)
            .zip(input.par_chunks(chunk_bytes))
            .for_each(|(o, i)| softmax_rows_f32_simd(i, o, last));
    }
    #[cfg(not(feature = "rayon"))]
    {
        softmax_rows_f32_simd(input, &mut output, last);
    }

    FlexTensor::new(Bytes::from_elems(output), Layout::contiguous(shape), DType::F32)
}

/// SIMD-dispatched row sweep for f32 softmax.
///
/// Macerator picks the best backend (NEON on aarch64, AVX/AVX-512 on x86_64,
/// SIMD128 on wasm32, scalar fallback elsewhere) once per call. The inner
/// row kernel runs fully in the selected ISA for the max-reduce and
/// normalize passes; the exp pass stays scalar because macerator does not
/// (yet) expose a vectorized `exp`. The exp pass is memory-bandwidth bound
/// anyway — every element still has to be read, exp'd, and written — so
/// the SIMD wins come from the other two passes.
#[macerator::with_simd]
fn softmax_rows_f32_simd<S: macerator::Simd>(
    input: &[f32],
    output: &mut [f32],
    row_len: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % row_len, 0);
    for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
        softmax_row_f32_simd::<S>(in_row, out_row);
    }
}

/// Inner row kernel for a single softmax row. `#[inline(always)]` so it
/// inlines into `softmax_rows_f32_simd`'s loop body for each monomorphized S,
/// avoiding a per-row call boundary.
#[inline(always)]
fn softmax_row_f32_simd<S: macerator::Simd>(input: &[f32], output: &mut [f32]) {
    use macerator::{Scalar, vload_unaligned, vstore_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let len = input.len();
    let simd_len = len / lanes * lanes;

    // Pass 1: row max for numerical stability.
    // SIMD max-reduction across the row, scalar tail.
    let (mut max_val, tail_start) = if simd_len >= lanes {
        let mut max_vec = unsafe { vload_unaligned::<S, _>(input.as_ptr()) };
        let mut j = lanes;
        while j < simd_len {
            let v = unsafe { vload_unaligned::<S, _>(input.as_ptr().add(j)) };
            max_vec = max_vec.max(v);
            j += lanes;
        }
        (max_vec.reduce_max(), simd_len)
    } else {
        (f32::NEG_INFINITY, 0)
    };
    for &x in &input[tail_start..] {
        if x > max_val {
            max_val = x;
        }
    }

    // Pass 2: compute exp(x - max), store in output, accumulate sum.
    // Scalar exp (no SIMD exp in macerator). This pass is the one that
    // actually does memory reads + writes on the whole row, so scalar
    // here still lands us at memory bandwidth.
    let mut sum = 0.0f32;
    for idx in 0..len {
        let e = (input[idx] - max_val).exp();
        output[idx] = e;
        sum += e;
    }

    // Pass 3: normalize — multiply every element by 1/sum.
    // SIMD splat + multiply, scalar tail.
    let inv = 1.0f32 / sum;
    let inv_vec = inv.splat::<S>();
    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(output.as_ptr().add(i));
            vstore_unaligned::<S, _>(output.as_mut_ptr().add(i), v * inv_vec);
        }
        i += lanes;
    }
    for x in &mut output[i..] {
        *x *= inv;
    }
}

fn softmax_last_f64(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape().clone();
    // Shape derefs to &[usize], so we can use slice methods directly.
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[f64] = tensor.storage();
    let mut output: Vec<f64> = vec![0.0f64; input.len()];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(last)
            .zip(input.par_chunks(last))
            .for_each(|(o, i)| softmax_row_f64(i, o));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, o) in input.chunks(last).zip(output.chunks_mut(last)) {
            softmax_row_f64(i, o);
        }
    }

    FlexTensor::new(Bytes::from_elems(output), Layout::contiguous(shape), DType::F64)
}

#[inline]
fn softmax_row_f64(input: &[f64], output: &mut [f64]) {
    let mut max_val = f64::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }
    let mut sum = 0.0f64;
    for (i, &x) in input.iter().enumerate() {
        let e = (x - max_val).exp();
        output[i] = e;
        sum += e;
    }
    let inv = 1.0f64 / sum;
    for x in output.iter_mut() {
        *x *= inv;
    }
}

// f16 / bf16: convert to f32 for the numeric work (standard practice for
// half-precision accuracy; the exp() saturates very fast in f16 dynamic
// range). Row-level parallelism still applies.
fn softmax_last_f16(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape().clone();
    // Shape derefs to &[usize], so we can use slice methods directly.
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[f16] = tensor.storage();
    let mut output: Vec<f16> = vec![f16::from_f32(0.0); input.len()];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(last)
            .zip(input.par_chunks(last))
            .for_each(|(o, i)| softmax_row_f16(i, o));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, o) in input.chunks(last).zip(output.chunks_mut(last)) {
            softmax_row_f16(i, o);
        }
    }

    FlexTensor::new(Bytes::from_elems(output), Layout::contiguous(shape), DType::F16)
}

#[inline]
fn softmax_row_f16(input: &[f16], output: &mut [f16]) {
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        let xf = x.to_f32();
        if xf > max_val {
            max_val = xf;
        }
    }
    let mut sum = 0.0f32;
    let mut scratch: [f32; 0] = [];
    let _ = &mut scratch; // silence unused var if we later add a real scratch buffer
    // First compute exps into output (as f32 via storage manipulation via
    // Vec<f32>? too much ceremony — write as f16 directly, we'll reload on
    // normalize).
    for (i, &x) in input.iter().enumerate() {
        let e = (x.to_f32() - max_val).exp();
        output[i] = f16::from_f32(e);
        sum += e;
    }
    let inv = 1.0f32 / sum;
    for x in output.iter_mut() {
        *x = f16::from_f32(x.to_f32() * inv);
    }
}

fn softmax_last_bf16(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape().clone();
    // Shape derefs to &[usize], so we can use slice methods directly.
    let last = *shape.last().expect("softmax: empty shape");
    if last == 0 {
        return tensor;
    }
    let input: &[bf16] = tensor.storage();
    let mut output: Vec<bf16> = vec![bf16::from_f32(0.0); input.len()];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(last)
            .zip(input.par_chunks(last))
            .for_each(|(o, i)| softmax_row_bf16(i, o));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, o) in input.chunks(last).zip(output.chunks_mut(last)) {
            softmax_row_bf16(i, o);
        }
    }

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        DType::BF16,
    )
}

#[inline]
fn softmax_row_bf16(input: &[bf16], output: &mut [bf16]) {
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        let xf = x.to_f32();
        if xf > max_val {
            max_val = xf;
        }
    }
    let mut sum = 0.0f32;
    for (i, &x) in input.iter().enumerate() {
        let e = (x.to_f32() - max_val).exp();
        output[i] = bf16::from_f32(e);
        sum += e;
    }
    let inv = 1.0f32 / sum;
    for x in output.iter_mut() {
        *x = bf16::from_f32(x.to_f32() * inv);
    }
}

#[cfg(test)]
mod tests {
    use burn_backend::Tolerance;
    use burn_tensor::{Tensor, TensorData, activation};

    use crate::Flex;

    #[test]
    fn test_relu() {
        let t: Tensor<Flex, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::relu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_sigmoid() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // sigmoid(-10) ~ 0, sigmoid(0) = 0.5, sigmoid(10) ~ 1
        activation::sigmoid(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.5, 1.0]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_gelu() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-3.0f32, 0.0, 3.0], &Default::default());
        // gelu(0) = 0, gelu(-3) ~ -0.004, gelu(3) ~ 2.996
        activation::gelu(t).into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.0, 0.0, 3.0]),
            Tolerance::absolute(0.01),
        );
    }

    #[test]
    fn test_leaky_relu() {
        let t: Tensor<Flex, 1> =
            Tensor::from_data([-2.0f32, -1.0, 0.0, 1.0, 2.0], &Default::default());
        activation::leaky_relu(t, 0.01)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-0.02, -0.01, 0.0, 1.0, 2.0]),
                Tolerance::absolute(1e-6),
            );
    }

    #[test]
    fn test_softmax_1d() {
        use burn_tensor::TensorPrimitive;
        // softmax([1, 2, 3]) should equal the reference impl
        let t: Tensor<Flex, 1> = Tensor::from_data([1.0f32, 2.0, 3.0], &Default::default());
        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let result = crate::ops::activation::softmax(primitive, 0);
        let result: Tensor<Flex, 1> = Tensor::from_primitive(TensorPrimitive::Float(result));
        // e^1=2.7183, e^2=7.389, e^3=20.0855, sum=30.193
        // normalized: 0.09003, 0.24473, 0.66524
        result.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.09003, 0.24473, 0.66524]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_2d_last_axis() {
        use burn_tensor::TensorPrimitive;
        // Cross-check against burn_tensor::activation::softmax on the same input
        let data = [[-1.0f32, 0.0, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]];
        let t: Tensor<Flex, 2> = Tensor::from_data(data, &Default::default());
        let reference = activation::softmax(t.clone(), 1);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 1);
        let fused: Tensor<Flex, 2> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused.into_data().assert_approx_eq::<f32>(
            &reference.into_data(),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_softmax_3d_attention_shape() {
        // wav2vec2-like attention scores [heads, seq_q, seq_k], softmax over seq_k.
        use burn_tensor::TensorPrimitive;
        let t: Tensor<Flex, 3> = Tensor::from_data(
            [[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]],
            &Default::default(),
        );
        let reference = activation::softmax(t.clone(), 2);

        let primitive = match t.into_primitive() {
            TensorPrimitive::Float(x) => x,
            _ => unreachable!(),
        };
        let fused = crate::ops::activation::softmax(primitive, 2);
        let fused: Tensor<Flex, 3> = Tensor::from_primitive(TensorPrimitive::Float(fused));

        fused.into_data().assert_approx_eq::<f32>(
            &reference.into_data(),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_log_sigmoid() {
        let t: Tensor<Flex, 1> = Tensor::from_data([-10.0f32, 0.0, 10.0], &Default::default());
        // log_sigmoid(-10) ~ -10, log_sigmoid(0) = ln(0.5) = -0.6931..., log_sigmoid(10) ~ 0
        activation::log_sigmoid(t)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([-10.0, -0.6931472, 0.0]),
                Tolerance::absolute(1e-3),
            );
    }
}

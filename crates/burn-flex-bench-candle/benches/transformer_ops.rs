//! Transformer per-layer ops: layer_norm, gelu, softmax.
//!
//! These run once per transformer layer in wav2vec2-large (24 layers total),
//! so any gap here is multiplied 24x in end-to-end inference. Shapes mirror
//! wav2vec2-large: hidden=1024, intermediate=4096, num_heads=16.
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench transformer_ops
//! ```

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData, TensorPrimitive, activation};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

/// Wrapper that extracts the Flex primitive, calls burn-flex's fused
/// softmax, and wraps the result back. This is the fast path until
/// burn-backend's `ActivationOps` gains a `softmax` trait method.
fn flex_fused_softmax<const D: usize>(tensor: Tensor<Flex, D>, dim: usize) -> Tensor<Flex, D> {
    let primitive = match tensor.into_primitive() {
        TensorPrimitive::Float(inner) => inner,
        TensorPrimitive::QFloat(_) => unimplemented!("softmax on quantized"),
    };
    let result = burn_flex::ops::activation::softmax(primitive, dim);
    Tensor::from_primitive(TensorPrimitive::Float(result))
}

/// Wrapper for burn-flex's fused layer_norm. Same pattern as softmax.
/// users opt into the fast path via this function until burn upstream
/// grows a layer_norm backend hook.
fn flex_fused_layer_norm(
    x: Tensor<Flex, 2>,
    gamma: Tensor<Flex, 1>,
    beta: Tensor<Flex, 1>,
) -> Tensor<Flex, 2> {
    let x_p = match x.into_primitive() {
        TensorPrimitive::Float(inner) => inner,
        _ => unimplemented!(),
    };
    let g_p = match gamma.into_primitive() {
        TensorPrimitive::Float(inner) => inner,
        _ => unimplemented!(),
    };
    let b_p = match beta.into_primitive() {
        TensorPrimitive::Float(inner) => inner,
        _ => unimplemented!(),
    };
    let result = burn_flex::ops::activation::layer_norm(x_p, g_p, Some(b_p), 1e-5);
    Tensor::from_primitive(TensorPrimitive::Float(result))
}

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Transformer ops: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_2d(rows: usize, cols: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(
        TensorData::new(fill(rows * cols), [rows, cols]),
        &Default::default(),
    )
}

fn flex_1d(n: usize) -> Tensor<Flex, 1> {
    Tensor::from_data(TensorData::new(fill(n), [n]), &Default::default())
}

fn flex_3d(d0: usize, d1: usize, d2: usize) -> Tensor<Flex, 3> {
    Tensor::from_data(
        TensorData::new(fill(d0 * d1 * d2), [d0, d1, d2]),
        &Default::default(),
    )
}

fn candle_2d(rows: usize, cols: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(rows * cols), (rows, cols), &CandleDevice::Cpu).unwrap()
}

fn candle_1d(n: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(n), n, &CandleDevice::Cpu).unwrap()
}

fn candle_3d(d0: usize, d1: usize, d2: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(d0 * d1 * d2), (d0, d1, d2), &CandleDevice::Cpu).unwrap()
}

/// Named-field bench shape for 2D and "seq-square" (3D attention) ops.
/// The `label` field is what divan prints as the bench arg name.
#[derive(Copy, Clone)]
struct Shape2D {
    rows: usize,
    cols: usize,
    label: &'static str,
}

impl std::fmt::Display for Shape2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

// ============================================================================
// gelu: one call per transformer layer, on the FFN intermediate [seq, 4096]
// ============================================================================

const GELU_SHAPES: &[Shape2D] = &[
    Shape2D {
        rows: 50,
        cols: 4096,
        label: "ffn_inter_1s",
    },
    Shape2D {
        rows: 150,
        cols: 4096,
        label: "ffn_inter_3s",
    },
    Shape2D {
        rows: 50,
        cols: 1024,
        label: "hidden_1s",
    },
    Shape2D {
        rows: 150,
        cols: 1024,
        label: "hidden_3s",
    },
];

#[divan::bench_group(name = "flex/gelu")]
mod flex_gelu {
    use super::*;

    #[divan::bench(args = GELU_SHAPES)]
    fn gelu(bencher: Bencher, shape: &Shape2D) {
        let x = flex_2d(shape.rows, shape.cols);
        bencher.bench(|| activation::gelu(x.clone()));
    }
}

#[divan::bench_group(name = "candle/gelu")]
mod candle_gelu {
    use super::*;

    #[divan::bench(args = GELU_SHAPES)]
    fn gelu(bencher: Bencher, shape: &Shape2D) {
        let x = candle_2d(shape.rows, shape.cols);
        bencher.bench(|| x.gelu().unwrap());
    }
}

// ============================================================================
// softmax: attention scores, applied over the last dim of [heads, seq, seq].
// Shape2D.rows = heads, Shape2D.cols = seq; the 3D tensor is built as
// [heads, seq, seq] so softmax(dim=2) normalizes the last axis.
// ============================================================================

const SOFTMAX_SHAPES: &[Shape2D] = &[
    Shape2D {
        rows: 16,
        cols: 50,
        label: "attn_1s",
    },
    Shape2D {
        rows: 16,
        cols: 150,
        label: "attn_3s",
    },
];

/// Old path: burn_tensor::activation::softmax, which decomposes into 5
/// primitive ops (max_dim, sub, exp, sum_dim, div). This is what every
/// burn-flex user gets today by calling the standard import.
#[divan::bench_group(name = "flex/softmax_decomposed")]
mod flex_softmax_decomposed {
    use super::*;

    #[divan::bench(args = SOFTMAX_SHAPES)]
    fn softmax(bencher: Bencher, shape: &Shape2D) {
        let x = flex_3d(shape.rows, shape.cols, shape.cols);
        bencher.bench(|| activation::softmax(x.clone(), 2));
    }
}

/// New path: burn_flex::ops::activation::softmax, fused single-pass
/// per row. Until the upstream `ActivationOps` trait has a softmax hook,
/// users call this directly via the `flex_fused_softmax` wrapper above.
#[divan::bench_group(name = "flex/softmax_fused")]
mod flex_softmax_fused {
    use super::*;

    #[divan::bench(args = SOFTMAX_SHAPES)]
    fn softmax(bencher: Bencher, shape: &Shape2D) {
        let x = flex_3d(shape.rows, shape.cols, shape.cols);
        bencher.bench(|| flex_fused_softmax(x.clone(), 2));
    }
}

#[divan::bench_group(name = "candle/softmax")]
mod candle_softmax {
    use super::*;

    #[divan::bench(args = SOFTMAX_SHAPES)]
    fn softmax(bencher: Bencher, shape: &Shape2D) {
        let x = candle_3d(shape.rows, shape.cols, shape.cols);
        bencher.bench(|| candle_nn::ops::softmax_last_dim(&x).unwrap());
    }
}

// ============================================================================
// layer_norm: [seq, 1024], one call before attention, one before FFN, per layer
// ============================================================================

const LN_SHAPES: &[Shape2D] = &[
    Shape2D {
        rows: 50,
        cols: 1024,
        label: "hidden_1s",
    },
    Shape2D {
        rows: 150,
        cols: 1024,
        label: "hidden_3s",
    },
];

/// Manual layer_norm for burn-flex using primitive tensor ops.
///
/// Matches the computation done by `nn::LayerNorm`: normalize across the last
/// axis with per-row mean and variance, then affine transform by gamma/beta.
/// Using primitives on both sides keeps the comparison apples-to-apples.
fn flex_layer_norm(
    x: Tensor<Flex, 2>,
    gamma: Tensor<Flex, 1>,
    beta: Tensor<Flex, 1>,
) -> Tensor<Flex, 2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().powi_scalar(2).mean_dim(1);
    let eps = 1e-5f32;
    let normed = centered / (var + eps).sqrt();
    // Broadcast gamma/beta [hidden] over the [rows, hidden] tensor
    normed * gamma.unsqueeze::<2>() + beta.unsqueeze::<2>()
}

/// Decomposed path: primitive tensor ops (mean, sub, var, sqrt, mul, add)
/// matching what burn::nn::LayerNorm::forward produces today.
#[divan::bench_group(name = "flex/layer_norm_decomposed")]
mod flex_layer_norm_decomposed {
    use super::*;

    #[divan::bench(args = LN_SHAPES)]
    fn layer_norm(bencher: Bencher, shape: &Shape2D) {
        let (rows, hidden) = (shape.rows, shape.cols);
        let x = flex_2d(rows, hidden);
        let gamma = flex_1d(hidden);
        let beta = flex_1d(hidden);
        bencher.bench(|| flex_layer_norm(x.clone(), gamma.clone(), beta.clone()));
    }
}

/// Fused path: burn_flex::ops::activation::layer_norm. Two SIMD passes
/// per row (sum+sumsq, then normalize+affine), one macerator dispatch per
/// chunk of rows.
#[divan::bench_group(name = "flex/layer_norm_fused")]
mod flex_layer_norm_fused {
    use super::*;

    #[divan::bench(args = LN_SHAPES)]
    fn layer_norm(bencher: Bencher, shape: &Shape2D) {
        let (rows, hidden) = (shape.rows, shape.cols);
        let x = flex_2d(rows, hidden);
        let gamma = flex_1d(hidden);
        let beta = flex_1d(hidden);
        bencher.bench(|| flex_fused_layer_norm(x.clone(), gamma.clone(), beta.clone()));
    }
}

#[divan::bench_group(name = "candle/layer_norm")]
mod candle_layer_norm_bench {
    use super::*;

    #[divan::bench(args = LN_SHAPES)]
    fn layer_norm(bencher: Bencher, shape: &Shape2D) {
        let (rows, hidden) = (shape.rows, shape.cols);
        let x = candle_2d(rows, hidden);
        let gamma = candle_1d(hidden);
        let beta = candle_1d(hidden);
        // candle_nn::ops::layer_norm is the fused path candle uses.
        bencher.bench(|| candle_nn::ops::layer_norm(&x, &gamma, &beta, 1e-5).unwrap());
    }
}

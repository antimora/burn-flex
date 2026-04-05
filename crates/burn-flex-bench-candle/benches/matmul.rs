//! Direct matmul comparison: burn-flex vs candle (pure Rust, no BLAS).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench matmul
//! ```
//!
//! Both backends receive identical input data. Candle is built without any
//! BLAS feature (see Cargo.toml) so we are comparing Rust kernels only.

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Matmul: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

/// Deterministic filler so both backends see byte-identical inputs.
fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_matrix(rows: usize, cols: usize) -> Tensor<Flex, 2> {
    let data = fill(rows * cols);
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn candle_matrix(rows: usize, cols: usize) -> CandleTensor {
    let data = fill(rows * cols);
    CandleTensor::from_vec(data, (rows, cols), &CandleDevice::Cpu).unwrap()
}

// Square matmul: useful for raw GEMM throughput comparison across sizes.
#[divan::bench_group(name = "flex/square")]
mod flex_square {
    use super::*;

    #[divan::bench(args = [128, 256, 512, 1024])]
    fn matmul(bencher: Bencher, n: usize) {
        let a = flex_matrix(n, n);
        let b = flex_matrix(n, n);
        bencher.bench(|| a.clone().matmul(b.clone()));
    }
}

#[divan::bench_group(name = "candle/square")]
mod candle_square {
    use super::*;

    #[divan::bench(args = [128, 256, 512, 1024])]
    fn matmul(bencher: Bencher, n: usize) {
        let a = candle_matrix(n, n);
        let b = candle_matrix(n, n);
        bencher.bench(|| a.matmul(&b).unwrap());
    }
}

// ============================================================================
// wav2vec2-large transformer matmul shapes
// ============================================================================
//
// wav2vec2-large: hidden_size=1024, intermediate_size=4096, num_heads=16,
// head_dim=64, 24 transformer layers. Feature extractor downsamples 16kHz
// audio 320x: 1s -> ~50 encoder frames, 3s -> ~150.
//
// Per transformer layer the 2D matmuls are:
//   - QKV proj (x3):    [seq, 1024] x [1024, 1024]
//   - Attn out proj:    [seq, 1024] x [1024, 1024]
//   - FFN up:           [seq, 1024] x [1024, 4096]
//   - FFN down:         [seq, 4096] x [4096, 1024]
//
// These run 24 layers per forward pass, so any gap here is multiplied 24x.

#[derive(Copy, Clone)]
struct Shape(usize, usize, usize, &'static str);

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.3)
    }
}

const SHAPES: &[Shape] = &[
    Shape(50, 1024, 1024, "qkv_proj_1s"),
    Shape(50, 1024, 4096, "ffn_up_1s"),
    Shape(50, 4096, 1024, "ffn_down_1s"),
    Shape(150, 1024, 1024, "qkv_proj_3s"),
    Shape(150, 1024, 4096, "ffn_up_3s"),
    Shape(150, 4096, 1024, "ffn_down_3s"),
];

#[divan::bench_group(name = "flex/transformer")]
mod flex_transformer {
    use super::*;

    #[divan::bench(args = SHAPES)]
    fn matmul(bencher: Bencher, shape: &Shape) {
        let Shape(m, k, n, _) = *shape;
        let a = flex_matrix(m, k);
        let b = flex_matrix(k, n);
        bencher.bench(|| a.clone().matmul(b.clone()));
    }
}

#[divan::bench_group(name = "candle/transformer")]
mod candle_transformer {
    use super::*;

    #[divan::bench(args = SHAPES)]
    fn matmul(bencher: Bencher, shape: &Shape) {
        let Shape(m, k, n, _) = *shape;
        let a = candle_matrix(m, k);
        let b = candle_matrix(k, n);
        bencher.bench(|| a.matmul(&b).unwrap());
    }
}

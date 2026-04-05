//! Batched matmul vs candle: attention Q@K^T and A@V shapes.
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench matmul_batched
//! ```
//!
//! wav2vec2-large attention uses 16 heads, head_dim=64.
//! QK^T: [heads, seq, head_dim] @ [heads, head_dim, seq] -> [heads, seq, seq]
//! AV:   [heads, seq, seq]      @ [heads, seq, head_dim] -> [heads, seq, head_dim]

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Batched matmul: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_3d(b: usize, m: usize, n: usize) -> Tensor<Flex, 3> {
    Tensor::from_data(
        TensorData::new(fill(b * m * n), [b, m, n]),
        &Default::default(),
    )
}
fn candle_3d(b: usize, m: usize, n: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(b * m * n), (b, m, n), &CandleDevice::Cpu).unwrap()
}

#[derive(Copy, Clone)]
struct Shape {
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    label: &'static str,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

// wav2vec2-large: heads=16, head_dim=64. 1s audio -> seq=50, 3s -> seq=150.
const SHAPES: &[Shape] = &[
    // QK^T
    Shape {
        batch: 16,
        m: 50,
        k: 64,
        n: 50,
        label: "qk_1s",
    },
    Shape {
        batch: 16,
        m: 150,
        k: 64,
        n: 150,
        label: "qk_3s",
    },
    // AV
    Shape {
        batch: 16,
        m: 50,
        k: 50,
        n: 64,
        label: "av_1s",
    },
    Shape {
        batch: 16,
        m: 150,
        k: 150,
        n: 64,
        label: "av_3s",
    },
    // Larger batched matmul to stress the kernel past the microbench floor.
    Shape {
        batch: 32,
        m: 128,
        k: 128,
        n: 128,
        label: "batched_128",
    },
];

#[divan::bench_group(name = "flex/batched")]
mod flex_batched {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn matmul(bencher: Bencher, s: &Shape) {
        let a = flex_3d(s.batch, s.m, s.k);
        let b = flex_3d(s.batch, s.k, s.n);
        bencher.bench(|| a.clone().matmul(b.clone()));
    }
}
#[divan::bench_group(name = "candle/batched")]
mod candle_batched {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn matmul(bencher: Bencher, s: &Shape) {
        let a = candle_3d(s.batch, s.m, s.k);
        let b = candle_3d(s.batch, s.k, s.n);
        bencher.bench(|| a.matmul(&b).unwrap());
    }
}

// A realistic attention step: Q@K^T is often called on a transposed K produced
// as a view. We replicate that here to include the view-dispatch cost.
#[divan::bench_group(name = "flex/qk_transposed_view")]
mod flex_qk_transposed {
    use super::*;
    #[divan::bench(args = [(16, 50, 64), (16, 150, 64)])]
    fn qk(bencher: Bencher, dims: (usize, usize, usize)) {
        let (h, seq, hd) = dims;
        let q = flex_3d(h, seq, hd);
        let k = flex_3d(h, seq, hd);
        bencher.bench(|| q.clone().matmul(k.clone().swap_dims(1, 2)));
    }
}
#[divan::bench_group(name = "candle/qk_transposed_view")]
mod candle_qk_transposed {
    use super::*;
    #[divan::bench(args = [(16, 50, 64), (16, 150, 64)])]
    fn qk(bencher: Bencher, dims: (usize, usize, usize)) {
        let (h, seq, hd) = dims;
        let q = candle_3d(h, seq, hd);
        let k = candle_3d(h, seq, hd);
        bencher.bench(|| q.matmul(&k.transpose(1, 2).unwrap()).unwrap());
    }
}

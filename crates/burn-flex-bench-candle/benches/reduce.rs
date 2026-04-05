//! Reductions vs candle (pure Rust).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench reduce
//! ```
//!
//! Reductions along the *last* dim are the common SIMD-friendly case;
//! reductions along a non-last dim are where many backends lose performance
//! because the inner stride is larger than 1.

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Reductions: burn-flex vs candle (pure Rust)");
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

fn candle_2d(rows: usize, cols: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(rows * cols), (rows, cols), &CandleDevice::Cpu).unwrap()
}

#[derive(Copy, Clone)]
struct Shape {
    rows: usize,
    cols: usize,
    label: &'static str,
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

/// Square-ish and transformer-ish shapes.
const SHAPES: &[Shape] = &[
    Shape {
        rows: 256,
        cols: 256,
        label: "256x256",
    },
    Shape {
        rows: 1024,
        cols: 1024,
        label: "1024x1024",
    },
    Shape {
        rows: 150,
        cols: 1024,
        label: "hidden_3s",
    },
    Shape {
        rows: 150,
        cols: 4096,
        label: "ffn_3s",
    },
];

// ============================================================================
// Full reductions
// ============================================================================

#[divan::bench_group(name = "flex/sum")]
mod flex_sum {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().sum());
    }
}
#[divan::bench_group(name = "candle/sum")]
mod candle_sum {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.sum_all().unwrap());
    }
}

#[divan::bench_group(name = "flex/mean")]
mod flex_mean {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn mean(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().mean());
    }
}
#[divan::bench_group(name = "candle/mean")]
mod candle_mean {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn mean(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.mean_all().unwrap());
    }
}

#[divan::bench_group(name = "flex/max")]
mod flex_max {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().max());
    }
}
#[divan::bench_group(name = "candle/max")]
mod candle_max {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        // Max over all dims: reduce dim 0 then dim 0 of result.
        bencher.bench(|| x.flatten_all().unwrap().max(0).unwrap());
    }
}

#[divan::bench_group(name = "flex/min")]
mod flex_min {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn min(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().min());
    }
}
#[divan::bench_group(name = "candle/min")]
mod candle_min {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn min(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.flatten_all().unwrap().min(0).unwrap());
    }
}

// ============================================================================
// Dim reductions — last dim (cols). This is the fast path: inner stride = 1.
// ============================================================================

#[divan::bench_group(name = "flex/sum_last")]
mod flex_sum_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum_dim(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().sum_dim(1));
    }
}
#[divan::bench_group(name = "candle/sum_last")]
mod candle_sum_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum_dim(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.sum_keepdim(1).unwrap());
    }
}

#[divan::bench_group(name = "flex/mean_last")]
mod flex_mean_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn mean_dim(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().mean_dim(1));
    }
}
#[divan::bench_group(name = "candle/mean_last")]
mod candle_mean_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn mean_dim(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.mean_keepdim(1).unwrap());
    }
}

#[divan::bench_group(name = "flex/max_last")]
mod flex_max_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max_dim(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().max_dim(1));
    }
}
#[divan::bench_group(name = "candle/max_last")]
mod candle_max_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max_dim(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.max_keepdim(1).unwrap());
    }
}

#[divan::bench_group(name = "flex/argmax_last")]
mod flex_argmax_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn argmax(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().argmax(1));
    }
}
#[divan::bench_group(name = "candle/argmax_last")]
mod candle_argmax_last {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn argmax(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.argmax_keepdim(1).unwrap());
    }
}

// ============================================================================
// Dim reductions — non-last dim (rows). Inner stride = cols, which stresses
// the backend's ability to vectorize across non-unit strides.
// ============================================================================

#[divan::bench_group(name = "flex/sum_first")]
mod flex_sum_first {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum_dim(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().sum_dim(0));
    }
}
#[divan::bench_group(name = "candle/sum_first")]
mod candle_sum_first {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn sum_dim(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.sum_keepdim(0).unwrap());
    }
}

#[divan::bench_group(name = "flex/max_first")]
mod flex_max_first {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max_dim(bencher: Bencher, s: &Shape) {
        let x = flex_2d(s.rows, s.cols);
        bencher.bench(|| x.clone().max_dim(0));
    }
}
#[divan::bench_group(name = "candle/max_first")]
mod candle_max_first {
    use super::*;
    #[divan::bench(args = SHAPES)]
    fn max_dim(bencher: Bencher, s: &Shape) {
        let x = candle_2d(s.rows, s.cols);
        bencher.bench(|| x.max_keepdim(0).unwrap());
    }
}

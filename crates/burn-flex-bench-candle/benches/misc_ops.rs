//! Misc ops: interpolate (nearest2d), cumsum, sort.
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench misc_ops
//! ```

use burn_flex::Flex;
use burn_tensor::{
    Tensor, TensorData, module,
    ops::{InterpolateMode, InterpolateOptions},
};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Misc ops: burn-flex vs candle (pure Rust)");
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

fn flex_4d(b: usize, c: usize, h: usize, w: usize) -> Tensor<Flex, 4> {
    Tensor::from_data(
        TensorData::new(fill(b * c * h * w), [b, c, h, w]),
        &Default::default(),
    )
}
fn candle_4d(b: usize, c: usize, h: usize, w: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(b * c * h * w), (b, c, h, w), &CandleDevice::Cpu).unwrap()
}

// ============================================================================
// interpolate nearest2d
// ============================================================================

#[derive(Copy, Clone)]
struct Up {
    label: &'static str,
    b: usize,
    c: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
}

impl std::fmt::Display for Up {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label)
    }
}

const UPS: &[Up] = &[
    Up {
        label: "upsample_2x_64",
        b: 1,
        c: 3,
        h_in: 64,
        w_in: 64,
        h_out: 128,
        w_out: 128,
    },
    Up {
        label: "upsample_4x_32",
        b: 1,
        c: 3,
        h_in: 32,
        w_in: 32,
        h_out: 128,
        w_out: 128,
    },
    Up {
        label: "downsample_256_to_128",
        b: 1,
        c: 3,
        h_in: 256,
        w_in: 256,
        h_out: 128,
        w_out: 128,
    },
];

#[divan::bench_group(name = "flex/nearest2d")]
mod flex_nearest {
    use super::*;
    #[divan::bench(args = UPS)]
    fn nearest(bencher: Bencher, u: &Up) {
        let x = flex_4d(u.b, u.c, u.h_in, u.w_in);
        let opts = InterpolateOptions::new(InterpolateMode::Nearest);
        bencher.bench(|| module::interpolate(x.clone(), [u.h_out, u.w_out], opts.clone()));
    }
}
#[divan::bench_group(name = "candle/nearest2d")]
mod candle_nearest {
    use super::*;
    #[divan::bench(args = UPS)]
    fn nearest(bencher: Bencher, u: &Up) {
        let x = candle_4d(u.b, u.c, u.h_in, u.w_in);
        bencher.bench(|| x.upsample_nearest2d(u.h_out, u.w_out).unwrap());
    }
}

// ============================================================================
// cumsum — prefix sum along a dim.
// ============================================================================

#[divan::bench_group(name = "flex/cumsum_last")]
mod flex_cumsum {
    use super::*;
    #[divan::bench(args = [(256usize, 256usize), (1024, 1024)])]
    fn cumsum(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = flex_2d(r, c);
        bencher.bench(|| x.clone().cumsum(1));
    }
}
#[divan::bench_group(name = "candle/cumsum_last")]
mod candle_cumsum {
    use super::*;
    #[divan::bench(args = [(256usize, 256usize), (1024, 1024)])]
    fn cumsum(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = candle_2d(r, c);
        bencher.bench(|| x.cumsum(1).unwrap());
    }
}

// ============================================================================
// sort — candle only has sort_last_dim, so we mirror that on burn.
// ============================================================================

#[divan::bench_group(name = "flex/sort_last")]
mod flex_sort {
    use super::*;
    #[divan::bench(args = [(256usize, 256usize), (1024, 1024)])]
    fn sort(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = flex_2d(r, c);
        bencher.bench(|| x.clone().sort(1));
    }
}
#[divan::bench_group(name = "candle/sort_last")]
mod candle_sort {
    use super::*;
    #[divan::bench(args = [(256usize, 256usize), (1024, 1024)])]
    fn sort(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = candle_2d(r, c);
        bencher.bench(|| x.sort_last_dim(true).unwrap());
    }
}

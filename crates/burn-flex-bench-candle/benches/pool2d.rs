//! max_pool2d + avg_pool2d vs candle (pure Rust).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench pool2d
//! ```
//!
//! Candle's pool API only takes a kernel size and stride (no padding / no
//! dilation), so we stick to zero-padding, dilation=1 configs on both sides.

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData, module};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Pool2d: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_in(b: usize, c: usize, h: usize, w: usize) -> Tensor<Flex, 4> {
    Tensor::from_data(
        TensorData::new(fill(b * c * h * w), [b, c, h, w]),
        &Default::default(),
    )
}
fn candle_in(b: usize, c: usize, h: usize, w: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(b * c * h * w), (b, c, h, w), &CandleDevice::Cpu).unwrap()
}

#[derive(Copy, Clone)]
struct P {
    name: &'static str,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
}

impl std::fmt::Display for P {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

const POOLS: &[P] = &[
    P {
        name: "k2s2_64x56",
        b: 1,
        c: 64,
        h: 56,
        w: 56,
        k: 2,
        stride: 2,
    },
    P {
        name: "k2s2_128x28",
        b: 1,
        c: 128,
        h: 28,
        w: 28,
        k: 2,
        stride: 2,
    },
    P {
        name: "k3s2_64x112",
        b: 1,
        c: 64,
        h: 112,
        w: 112,
        k: 3,
        stride: 2,
    },
    P {
        name: "k3s2_batch8_64x56",
        b: 8,
        c: 64,
        h: 56,
        w: 56,
        k: 3,
        stride: 2,
    },
];

#[divan::bench_group(name = "flex/max_pool2d")]
mod flex_max_pool {
    use super::*;
    #[divan::bench(args = POOLS)]
    fn max_pool2d(bencher: Bencher, p: &P) {
        let x = flex_in(p.b, p.c, p.h, p.w);
        bencher.bench(|| {
            module::max_pool2d(
                x.clone(),
                [p.k, p.k],
                [p.stride, p.stride],
                [0, 0],
                [1, 1],
                false,
            )
        });
    }
}
#[divan::bench_group(name = "candle/max_pool2d")]
mod candle_max_pool {
    use super::*;
    #[divan::bench(args = POOLS)]
    fn max_pool2d(bencher: Bencher, p: &P) {
        let x = candle_in(p.b, p.c, p.h, p.w);
        bencher.bench(|| {
            x.max_pool2d_with_stride((p.k, p.k), (p.stride, p.stride))
                .unwrap()
        });
    }
}

#[divan::bench_group(name = "flex/avg_pool2d")]
mod flex_avg_pool {
    use super::*;
    #[divan::bench(args = POOLS)]
    fn avg_pool2d(bencher: Bencher, p: &P) {
        let x = flex_in(p.b, p.c, p.h, p.w);
        bencher.bench(|| {
            module::avg_pool2d(
                x.clone(),
                [p.k, p.k],
                [p.stride, p.stride],
                [0, 0],
                false,
                false,
            )
        });
    }
}
#[divan::bench_group(name = "candle/avg_pool2d")]
mod candle_avg_pool {
    use super::*;
    #[divan::bench(args = POOLS)]
    fn avg_pool2d(bencher: Bencher, p: &P) {
        let x = candle_in(p.b, p.c, p.h, p.w);
        bencher.bench(|| {
            x.avg_pool2d_with_stride((p.k, p.k), (p.stride, p.stride))
                .unwrap()
        });
    }
}

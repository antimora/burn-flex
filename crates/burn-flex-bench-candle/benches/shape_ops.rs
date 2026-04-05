//! Shape / layout ops vs candle: transpose, reshape, narrow, expand, cat,
//! repeat, permute, flip.
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench shape_ops
//! ```
//!
//! Transpose/reshape/narrow/expand/permute are normally O(1) view operations
//! on both backends; these benches measure view-creation overhead plus any
//! materialization the backend chooses to do. `cat` and `repeat` always copy.

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Shape ops: burn-flex vs candle (pure Rust)");
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
fn flex_3d(d0: usize, d1: usize, d2: usize) -> Tensor<Flex, 3> {
    Tensor::from_data(
        TensorData::new(fill(d0 * d1 * d2), [d0, d1, d2]),
        &Default::default(),
    )
}
fn candle_2d(rows: usize, cols: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(rows * cols), (rows, cols), &CandleDevice::Cpu).unwrap()
}
fn candle_3d(d0: usize, d1: usize, d2: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(d0 * d1 * d2), (d0, d1, d2), &CandleDevice::Cpu).unwrap()
}

// ============================================================================
// transpose / swap_dims — view op, so this is pure overhead measurement.
// ============================================================================

#[divan::bench_group(name = "flex/transpose")]
mod flex_transpose {
    use super::*;
    #[divan::bench(args = [256, 1024])]
    fn transpose(bencher: Bencher, n: usize) {
        let x = flex_2d(n, n);
        bencher.bench(|| x.clone().swap_dims(0, 1));
    }
}
#[divan::bench_group(name = "candle/transpose")]
mod candle_transpose {
    use super::*;
    #[divan::bench(args = [256, 1024])]
    fn transpose(bencher: Bencher, n: usize) {
        let x = candle_2d(n, n);
        bencher.bench(|| x.t().unwrap());
    }
}

// Transpose followed by a unary exp forces both backends to actually walk the
// non-contiguous layout. This is where strided traversal cost shows up.
#[divan::bench_group(name = "flex/transpose_then_exp")]
mod flex_transpose_then_exp {
    use super::*;
    #[divan::bench(args = [256, 1024])]
    fn transpose_exp(bencher: Bencher, n: usize) {
        let x = flex_2d(n, n);
        bencher.bench(|| x.clone().swap_dims(0, 1).exp());
    }
}
#[divan::bench_group(name = "candle/transpose_then_exp")]
mod candle_transpose_then_exp {
    use super::*;
    #[divan::bench(args = [256, 1024])]
    fn transpose_exp(bencher: Bencher, n: usize) {
        let x = candle_2d(n, n);
        bencher.bench(|| x.t().unwrap().exp().unwrap());
    }
}

// ============================================================================
// reshape — typically a view, but may materialize if layout isn't contiguous.
// ============================================================================

#[divan::bench_group(name = "flex/reshape")]
mod flex_reshape {
    use super::*;
    #[divan::bench(args = [(256, 256), (1024, 1024)])]
    fn reshape(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = flex_2d(r, c);
        bencher.bench(|| x.clone().reshape([r * c]));
    }
}
#[divan::bench_group(name = "candle/reshape")]
mod candle_reshape {
    use super::*;
    #[divan::bench(args = [(256, 256), (1024, 1024)])]
    fn reshape(bencher: Bencher, dims: (usize, usize)) {
        let (r, c) = dims;
        let x = candle_2d(r, c);
        bencher.bench(|| x.reshape((r * c,)).unwrap());
    }
}

// ============================================================================
// narrow / slice
// ============================================================================

#[divan::bench_group(name = "flex/narrow")]
mod flex_narrow {
    use super::*;
    #[divan::bench]
    fn narrow_1024(bencher: Bencher) {
        let x = flex_2d(1024, 1024);
        bencher.bench(|| x.clone().slice([0..512, 0..512]));
    }
}
#[divan::bench_group(name = "candle/narrow")]
mod candle_narrow {
    use super::*;
    #[divan::bench]
    fn narrow_1024(bencher: Bencher) {
        let x = candle_2d(1024, 1024);
        bencher.bench(|| {
            x.narrow(0, 0, 512)
                .and_then(|t| t.narrow(1, 0, 512))
                .unwrap()
        });
    }
}

// ============================================================================
// expand / broadcast_as
// ============================================================================

#[divan::bench_group(name = "flex/expand")]
mod flex_expand {
    use super::*;
    #[divan::bench]
    fn expand_row_1024(bencher: Bencher) {
        let x: Tensor<Flex, 2> =
            Tensor::from_data(TensorData::new(fill(1024), [1, 1024]), &Default::default());
        bencher.bench(|| x.clone().expand([1024, 1024]));
    }
}
#[divan::bench_group(name = "candle/expand")]
mod candle_expand {
    use super::*;
    #[divan::bench]
    fn expand_row_1024(bencher: Bencher) {
        let x = CandleTensor::from_vec(fill(1024), (1, 1024), &CandleDevice::Cpu).unwrap();
        bencher.bench(|| x.broadcast_as((1024, 1024)).unwrap());
    }
}

// ============================================================================
// cat — always materializes. This is the interesting one.
// ============================================================================

#[divan::bench_group(name = "flex/cat_dim0")]
mod flex_cat_dim0 {
    use super::*;
    #[divan::bench]
    fn cat_2x_1024(bencher: Bencher) {
        let a = flex_2d(1024, 1024);
        let b = flex_2d(1024, 1024);
        bencher.bench(|| Tensor::cat(vec![a.clone(), b.clone()], 0));
    }
}
#[divan::bench_group(name = "candle/cat_dim0")]
mod candle_cat_dim0 {
    use super::*;
    #[divan::bench]
    fn cat_2x_1024(bencher: Bencher) {
        let a = candle_2d(1024, 1024);
        let b = candle_2d(1024, 1024);
        bencher.bench(|| CandleTensor::cat(&[&a, &b], 0).unwrap());
    }
}

#[divan::bench_group(name = "flex/cat_last")]
mod flex_cat_last {
    use super::*;
    #[divan::bench]
    fn cat_2x_1024(bencher: Bencher) {
        let a = flex_2d(1024, 1024);
        let b = flex_2d(1024, 1024);
        bencher.bench(|| Tensor::cat(vec![a.clone(), b.clone()], 1));
    }
}
#[divan::bench_group(name = "candle/cat_last")]
mod candle_cat_last {
    use super::*;
    #[divan::bench]
    fn cat_2x_1024(bencher: Bencher) {
        let a = candle_2d(1024, 1024);
        let b = candle_2d(1024, 1024);
        bencher.bench(|| CandleTensor::cat(&[&a, &b], 1).unwrap());
    }
}

// ============================================================================
// repeat_dim / repeat
// ============================================================================

#[divan::bench_group(name = "flex/repeat_dim")]
mod flex_repeat {
    use super::*;
    #[divan::bench]
    fn repeat_256_4x(bencher: Bencher) {
        let x = flex_2d(256, 256);
        bencher.bench(|| x.clone().repeat_dim(0, 4));
    }
}
#[divan::bench_group(name = "candle/repeat_dim")]
mod candle_repeat {
    use super::*;
    #[divan::bench]
    fn repeat_256_4x(bencher: Bencher) {
        let x = candle_2d(256, 256);
        // candle's `repeat` takes a shape-like multiplier.
        bencher.bench(|| x.repeat((4, 1)).unwrap());
    }
}

// ============================================================================
// permute — 3D swap-two-dims variant
// ============================================================================

#[divan::bench_group(name = "flex/permute_3d")]
mod flex_permute {
    use super::*;
    #[divan::bench]
    fn permute_64x128x256(bencher: Bencher) {
        let x = flex_3d(64, 128, 256);
        bencher.bench(|| x.clone().permute([2, 0, 1]));
    }
}
#[divan::bench_group(name = "candle/permute_3d")]
mod candle_permute {
    use super::*;
    #[divan::bench]
    fn permute_64x128x256(bencher: Bencher) {
        let x = candle_3d(64, 128, 256);
        bencher.bench(|| x.permute((2, 0, 1)).unwrap());
    }
}

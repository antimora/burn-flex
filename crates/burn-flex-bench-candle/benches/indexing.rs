//! Indexing ops vs candle: gather, select (index_select), scatter_add,
//! where_cond (mask_where).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench indexing
//! ```

use burn_flex::Flex;
use burn_tensor::{IndexingUpdateOp, Int, Tensor, TensorData};
use candle_core::{DType as CDType, Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Indexing: burn-flex vs candle (pure Rust)");
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

fn flex_idx_2d(rows: usize, cols: usize, max_idx: usize) -> Tensor<Flex, 2, Int> {
    let data: Vec<i64> = (0..rows * cols).map(|i| (i % max_idx) as i64).collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}
fn flex_idx_1d(n: usize, max_idx: usize) -> Tensor<Flex, 1, Int> {
    let data: Vec<i64> = (0..n).map(|i| (i % max_idx) as i64).collect();
    Tensor::from_data(TensorData::new(data, [n]), &Default::default())
}

fn candle_idx_2d(rows: usize, cols: usize, max_idx: usize) -> CandleTensor {
    let data: Vec<u32> = (0..rows * cols).map(|i| (i % max_idx) as u32).collect();
    CandleTensor::from_vec(data, (rows, cols), &CandleDevice::Cpu).unwrap()
}
fn candle_idx_1d(n: usize, max_idx: usize) -> CandleTensor {
    let data: Vec<u32> = (0..n).map(|i| (i % max_idx) as u32).collect();
    CandleTensor::from_vec(data, n, &CandleDevice::Cpu).unwrap()
}

// ============================================================================
// gather — indices tensor same rank as source, one index per output element.
// ============================================================================

#[divan::bench_group(name = "flex/gather_last")]
mod flex_gather {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn gather(bencher: Bencher, n: usize) {
        let x = flex_2d(n, n);
        let idx = flex_idx_2d(n, n / 2, n);
        bencher.bench(|| x.clone().gather(1, idx.clone()));
    }
}
#[divan::bench_group(name = "candle/gather_last")]
mod candle_gather {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn gather(bencher: Bencher, n: usize) {
        let x = candle_2d(n, n);
        let idx = candle_idx_2d(n, n / 2, n);
        bencher.bench(|| x.gather(&idx, 1).unwrap());
    }
}

// ============================================================================
// index_select / select — 1D index over one dim.
// ============================================================================

#[divan::bench_group(name = "flex/select_dim0")]
mod flex_select {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn select(bencher: Bencher, n: usize) {
        let x = flex_2d(n, n);
        let idx = flex_idx_1d(n / 2, n);
        bencher.bench(|| x.clone().select(0, idx.clone()));
    }
}
#[divan::bench_group(name = "candle/select_dim0")]
mod candle_select {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn select(bencher: Bencher, n: usize) {
        let x = candle_2d(n, n);
        let idx = candle_idx_1d(n / 2, n);
        bencher.bench(|| x.index_select(&idx, 0).unwrap());
    }
}

// ============================================================================
// scatter_add — same-rank indices, accumulates values into source.
// ============================================================================

#[divan::bench_group(name = "flex/scatter_add_last")]
mod flex_scatter {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn scatter(bencher: Bencher, n: usize) {
        let x = flex_2d(n, n);
        let idx = flex_idx_2d(n, n / 2, n);
        let vals = flex_2d(n, n / 2);
        bencher.bench(|| {
            x.clone()
                .scatter(1, idx.clone(), vals.clone(), IndexingUpdateOp::Add)
        });
    }
}
#[divan::bench_group(name = "candle/scatter_add_last")]
mod candle_scatter {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn scatter(bencher: Bencher, n: usize) {
        let x = candle_2d(n, n);
        let idx = candle_idx_2d(n, n / 2, n);
        let vals = candle_2d(n, n / 2);
        bencher.bench(|| x.scatter_add(&idx, &vals, 1).unwrap());
    }
}

// ============================================================================
// where_cond / mask_where — select between two tensors based on a bool mask.
// ============================================================================

#[divan::bench_group(name = "flex/where_cond")]
mod flex_where {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn where_cond(bencher: Bencher, n: usize) {
        let a = flex_2d(n, n);
        let b = flex_2d(n, n);
        let mask = a.clone().greater(b.clone());
        bencher.bench(|| a.clone().mask_where(mask.clone(), b.clone()));
    }
}
#[divan::bench_group(name = "candle/where_cond")]
mod candle_where {
    use super::*;
    #[divan::bench(args = [256usize, 1024])]
    fn where_cond(bencher: Bencher, n: usize) {
        let a = candle_2d(n, n);
        let b = candle_2d(n, n);
        // a.gt(b) produces a u8 mask; where_cond picks a where mask is true else b.
        let mask = a.gt(&b).unwrap().to_dtype(CDType::U8).unwrap();
        bencher.bench(|| mask.where_cond(&a, &b).unwrap());
    }
}

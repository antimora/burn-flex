//! Elementwise ops: unary, binary, comparison vs candle (pure Rust).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench elementwise
//! ```

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Elementwise: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

/// [0.1, 1.1) — safe for log/sqrt/recip.
fn fill_pos(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 + (i % 1000) as f32 / 1000.0).collect()
}

/// [-0.5, 0.5) — signed values for generic ops.
fn fill_signed(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_1d_pos(n: usize) -> Tensor<Flex, 1> {
    Tensor::from_data(TensorData::new(fill_pos(n), [n]), &Default::default())
}

fn flex_1d(n: usize) -> Tensor<Flex, 1> {
    Tensor::from_data(TensorData::new(fill_signed(n), [n]), &Default::default())
}

fn candle_1d_pos(n: usize) -> CandleTensor {
    CandleTensor::from_vec(fill_pos(n), n, &CandleDevice::Cpu).unwrap()
}

fn candle_1d(n: usize) -> CandleTensor {
    CandleTensor::from_vec(fill_signed(n), n, &CandleDevice::Cpu).unwrap()
}

/// Sizes: transformer-ish (50k ~= seq=50 * hidden=1024) and a full 1M.
const SIZES: &[usize] = &[50 * 1024, 150 * 1024, 1024 * 1024];

// ============================================================================
// Unary
// ============================================================================

#[divan::bench_group(name = "flex/exp")]
mod flex_exp {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn exp(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().exp());
    }
}
#[divan::bench_group(name = "candle/exp")]
mod candle_exp {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn exp(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.exp().unwrap());
    }
}

#[divan::bench_group(name = "flex/log")]
mod flex_log {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn log(bencher: Bencher, n: &usize) {
        let x = flex_1d_pos(*n);
        bencher.bench(|| x.clone().log());
    }
}
#[divan::bench_group(name = "candle/log")]
mod candle_log {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn log(bencher: Bencher, n: &usize) {
        let x = candle_1d_pos(*n);
        bencher.bench(|| x.log().unwrap());
    }
}

#[divan::bench_group(name = "flex/sqrt")]
mod flex_sqrt {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sqrt(bencher: Bencher, n: &usize) {
        let x = flex_1d_pos(*n);
        bencher.bench(|| x.clone().sqrt());
    }
}
#[divan::bench_group(name = "candle/sqrt")]
mod candle_sqrt {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sqrt(bencher: Bencher, n: &usize) {
        let x = candle_1d_pos(*n);
        bencher.bench(|| x.sqrt().unwrap());
    }
}

#[divan::bench_group(name = "flex/recip")]
mod flex_recip {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn recip(bencher: Bencher, n: &usize) {
        let x = flex_1d_pos(*n);
        bencher.bench(|| x.clone().recip());
    }
}
#[divan::bench_group(name = "candle/recip")]
mod candle_recip {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn recip(bencher: Bencher, n: &usize) {
        let x = candle_1d_pos(*n);
        bencher.bench(|| x.recip().unwrap());
    }
}

#[divan::bench_group(name = "flex/abs")]
mod flex_abs {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn abs(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().abs());
    }
}
#[divan::bench_group(name = "candle/abs")]
mod candle_abs {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn abs(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.abs().unwrap());
    }
}

#[divan::bench_group(name = "flex/neg")]
mod flex_neg {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn neg(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().neg());
    }
}
#[divan::bench_group(name = "candle/neg")]
mod candle_neg {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn neg(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.neg().unwrap());
    }
}

#[divan::bench_group(name = "flex/tanh")]
mod flex_tanh {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn tanh(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().tanh());
    }
}
#[divan::bench_group(name = "candle/tanh")]
mod candle_tanh {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn tanh(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.tanh().unwrap());
    }
}

#[divan::bench_group(name = "flex/sin")]
mod flex_sin {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sin(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().sin());
    }
}
#[divan::bench_group(name = "candle/sin")]
mod candle_sin {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sin(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.sin().unwrap());
    }
}

#[divan::bench_group(name = "flex/cos")]
mod flex_cos {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn cos(bencher: Bencher, n: &usize) {
        let x = flex_1d(*n);
        bencher.bench(|| x.clone().cos());
    }
}
#[divan::bench_group(name = "candle/cos")]
mod candle_cos {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn cos(bencher: Bencher, n: &usize) {
        let x = candle_1d(*n);
        bencher.bench(|| x.cos().unwrap());
    }
}

// ============================================================================
// Binary tensor-tensor
// ============================================================================

#[divan::bench_group(name = "flex/add")]
mod flex_add {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn add(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d(*n);
        bencher.bench(|| a.clone().add(b.clone()));
    }
}
#[divan::bench_group(name = "candle/add")]
mod candle_add {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn add(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d(*n);
        bencher.bench(|| (&a + &b).unwrap());
    }
}

#[divan::bench_group(name = "flex/sub")]
mod flex_sub {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sub(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d(*n);
        bencher.bench(|| a.clone().sub(b.clone()));
    }
}
#[divan::bench_group(name = "candle/sub")]
mod candle_sub {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn sub(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d(*n);
        bencher.bench(|| (&a - &b).unwrap());
    }
}

#[divan::bench_group(name = "flex/mul")]
mod flex_mul {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn mul(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d(*n);
        bencher.bench(|| a.clone().mul(b.clone()));
    }
}
#[divan::bench_group(name = "candle/mul")]
mod candle_mul {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn mul(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d(*n);
        bencher.bench(|| (&a * &b).unwrap());
    }
}

#[divan::bench_group(name = "flex/div")]
mod flex_div {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn div(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d_pos(*n);
        bencher.bench(|| a.clone().div(b.clone()));
    }
}
#[divan::bench_group(name = "candle/div")]
mod candle_div {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn div(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d_pos(*n);
        bencher.bench(|| (&a / &b).unwrap());
    }
}

// ============================================================================
// Binary tensor-scalar (candle maps this onto `affine(mul, add)`)
// ============================================================================

#[divan::bench_group(name = "flex/add_scalar")]
mod flex_add_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn add_scalar(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        bencher.bench(|| a.clone().add_scalar(0.5));
    }
}
#[divan::bench_group(name = "candle/add_scalar")]
mod candle_add_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn add_scalar(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        // candle has no add_scalar; affine(1.0, 0.5) is the idiomatic form.
        bencher.bench(|| a.affine(1.0, 0.5).unwrap());
    }
}

#[divan::bench_group(name = "flex/mul_scalar")]
mod flex_mul_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn mul_scalar(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        bencher.bench(|| a.clone().mul_scalar(2.5));
    }
}
#[divan::bench_group(name = "candle/mul_scalar")]
mod candle_mul_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn mul_scalar(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        bencher.bench(|| a.affine(2.5, 0.0).unwrap());
    }
}

#[divan::bench_group(name = "flex/powf_scalar")]
mod flex_powf_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn powf_scalar(bencher: Bencher, n: &usize) {
        let a = flex_1d_pos(*n);
        bencher.bench(|| a.clone().powf_scalar(2.5));
    }
}
#[divan::bench_group(name = "candle/powf_scalar")]
mod candle_powf_scalar {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn powf_scalar(bencher: Bencher, n: &usize) {
        let a = candle_1d_pos(*n);
        bencher.bench(|| a.powf(2.5).unwrap());
    }
}

// ============================================================================
// Comparison (result is bool/u8). Used heavily in masking paths.
// ============================================================================

#[divan::bench_group(name = "flex/greater")]
mod flex_greater {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn greater(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d(*n);
        bencher.bench(|| a.clone().greater(b.clone()));
    }
}
#[divan::bench_group(name = "candle/greater")]
mod candle_greater {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn greater(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d(*n);
        bencher.bench(|| a.gt(&b).unwrap());
    }
}

#[divan::bench_group(name = "flex/equal")]
mod flex_equal {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn equal(bencher: Bencher, n: &usize) {
        let a = flex_1d(*n);
        let b = flex_1d(*n);
        bencher.bench(|| a.clone().equal(b.clone()));
    }
}
#[divan::bench_group(name = "candle/equal")]
mod candle_equal {
    use super::*;
    #[divan::bench(args = SIZES)]
    fn equal(bencher: Bencher, n: &usize) {
        let a = candle_1d(*n);
        let b = candle_1d(*n);
        bencher.bench(|| a.eq(&b).unwrap());
    }
}

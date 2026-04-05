//! conv2d + conv_transpose2d vs candle (pure Rust).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench conv2d
//! ```
//!
//! Shapes mirror common image-model building blocks (ResNet early/mid layers)
//! so the numbers are meaningful for vision workloads, not only audio.

use burn_flex::Flex;
use burn_tensor::{
    Tensor, TensorData, module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Conv2d: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

fn flex_input(b: usize, c: usize, h: usize, w: usize) -> Tensor<Flex, 4> {
    Tensor::from_data(
        TensorData::new(fill(b * c * h * w), [b, c, h, w]),
        &Default::default(),
    )
}
fn flex_kernel(oc: usize, ic: usize, kh: usize, kw: usize) -> Tensor<Flex, 4> {
    Tensor::from_data(
        TensorData::new(fill(oc * ic * kh * kw), [oc, ic, kh, kw]),
        &Default::default(),
    )
}
fn candle_input(b: usize, c: usize, h: usize, w: usize) -> CandleTensor {
    CandleTensor::from_vec(fill(b * c * h * w), (b, c, h, w), &CandleDevice::Cpu).unwrap()
}
fn candle_kernel(oc: usize, ic: usize, kh: usize, kw: usize) -> CandleTensor {
    CandleTensor::from_vec(
        fill(oc * ic * kh * kw),
        (oc, ic, kh, kw),
        &CandleDevice::Cpu,
    )
    .unwrap()
}

#[derive(Copy, Clone)]
struct Conv {
    name: &'static str,
    b: usize,
    ic: usize,
    h: usize,
    w: usize,
    oc: usize,
    k: usize,
    stride: usize,
    padding: usize,
}

impl std::fmt::Display for Conv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

// Square 3x3 kernels, single padding value. ResNet-ish.
const LAYERS: &[Conv] = &[
    Conv {
        name: "resnet_conv1_k7_s2",
        b: 1,
        ic: 3,
        h: 224,
        w: 224,
        oc: 64,
        k: 7,
        stride: 2,
        padding: 3,
    },
    Conv {
        name: "resnet_l1_64x56",
        b: 1,
        ic: 64,
        h: 56,
        w: 56,
        oc: 64,
        k: 3,
        stride: 1,
        padding: 1,
    },
    Conv {
        name: "resnet_l2_128x28",
        b: 1,
        ic: 128,
        h: 28,
        w: 28,
        oc: 128,
        k: 3,
        stride: 1,
        padding: 1,
    },
    Conv {
        name: "resnet_l3_256x14",
        b: 1,
        ic: 256,
        h: 14,
        w: 14,
        oc: 256,
        k: 3,
        stride: 1,
        padding: 1,
    },
    Conv {
        name: "resnet_l4_512x7",
        b: 1,
        ic: 512,
        h: 7,
        w: 7,
        oc: 512,
        k: 3,
        stride: 1,
        padding: 1,
    },
    // 1x1 pointwise — almost pure gemm, useful to confirm matmul-dominated path.
    Conv {
        name: "pointwise_1x1_256x56",
        b: 1,
        ic: 256,
        h: 56,
        w: 56,
        oc: 64,
        k: 1,
        stride: 1,
        padding: 0,
    },
];

#[divan::bench_group(name = "flex/conv2d")]
mod flex_conv2d {
    use super::*;
    #[divan::bench(args = LAYERS)]
    fn conv2d(bencher: Bencher, l: &Conv) {
        let x = flex_input(l.b, l.ic, l.h, l.w);
        let w = flex_kernel(l.oc, l.ic, l.k, l.k);
        let opts = ConvOptions::new([l.stride, l.stride], [l.padding, l.padding], [1, 1], 1);
        bencher.bench(|| module::conv2d(x.clone(), w.clone(), None, opts.clone()));
    }
}
#[divan::bench_group(name = "candle/conv2d")]
mod candle_conv2d {
    use super::*;
    #[divan::bench(args = LAYERS)]
    fn conv2d(bencher: Bencher, l: &Conv) {
        let x = candle_input(l.b, l.ic, l.h, l.w);
        let w = candle_kernel(l.oc, l.ic, l.k, l.k);
        bencher.bench(|| x.conv2d(&w, l.padding, l.stride, 1, 1).unwrap());
    }
}

// ============================================================================
// conv_transpose2d — upsampling layer common in segmentation / GAN decoders.
// ============================================================================

#[derive(Copy, Clone)]
struct ConvT {
    name: &'static str,
    b: usize,
    ic: usize,
    h: usize,
    w: usize,
    oc: usize,
    k: usize,
    stride: usize,
}

impl std::fmt::Display for ConvT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

const CT_LAYERS: &[ConvT] = &[
    ConvT {
        name: "deconv_128_to_256",
        b: 1,
        ic: 128,
        h: 16,
        w: 16,
        oc: 64,
        k: 4,
        stride: 2,
    },
    ConvT {
        name: "deconv_64_to_128",
        b: 1,
        ic: 64,
        h: 32,
        w: 32,
        oc: 32,
        k: 4,
        stride: 2,
    },
];

#[divan::bench_group(name = "flex/conv_transpose2d")]
mod flex_convt {
    use super::*;
    #[divan::bench(args = CT_LAYERS)]
    fn conv_t(bencher: Bencher, l: &ConvT) {
        let x = flex_input(l.b, l.ic, l.h, l.w);
        // conv_transpose weight layout is [in_ch, out_ch, kh, kw] in burn.
        let w = flex_kernel(l.ic, l.oc, l.k, l.k);
        let opts = ConvTransposeOptions::new([l.stride, l.stride], [1, 1], [0, 0], [1, 1], 1);
        bencher.bench(|| module::conv_transpose2d(x.clone(), w.clone(), None, opts.clone()));
    }
}
#[divan::bench_group(name = "candle/conv_transpose2d")]
mod candle_convt {
    use super::*;
    #[divan::bench(args = CT_LAYERS)]
    fn conv_t(bencher: Bencher, l: &ConvT) {
        let x = candle_input(l.b, l.ic, l.h, l.w);
        // candle uses [in_ch, out_ch, kh, kw] for conv_transpose too.
        let w = candle_kernel(l.ic, l.oc, l.k, l.k);
        bencher.bench(|| {
            x.conv_transpose2d(
                &w, /*padding*/ 1, /*output_padding*/ 0, l.stride, 1,
            )
            .unwrap()
        });
    }
}

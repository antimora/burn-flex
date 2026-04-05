//! Direct conv1d comparison: burn-flex vs candle (pure Rust, no BLAS).
//!
//! Run with:
//! ```bash
//! cargo bench -p burn-flex-bench-candle --bench conv1d
//! ```
//!
//! Target shapes mirror the wav2vec2 feature extractor, which is 7 strided
//! conv1d layers that downsample raw audio 320x before it reaches the
//! transformer. These convs dominate the front of wav2vec2 inference.

use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData, module, ops::ConvOptions};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Conv1d: burn-flex vs candle (pure Rust)");
    println!();
    divan::main();
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect()
}

/// One layer of the wav2vec2 feature extractor.
///
/// Layers, in order, applied to 16kHz raw audio:
///   L0:  in=1,   out=512, kernel=10, stride=5
///   L1:  in=512, out=512, kernel=3,  stride=2
///   L2:  in=512, out=512, kernel=3,  stride=2
///   L3:  in=512, out=512, kernel=3,  stride=2
///   L4:  in=512, out=512, kernel=3,  stride=2
///   L5:  in=512, out=512, kernel=2,  stride=2
///   L6:  in=512, out=512, kernel=2,  stride=2
///
/// Total downsampling: 5*2*2*2*2*2*2 = 320x.
#[derive(Copy, Clone)]
struct Layer {
    name: &'static str,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    // Input length (in samples) seen by this layer when processing 1s of 16kHz audio.
    input_len: usize,
}

// Input lengths are computed assuming 1s = 16000 samples, no padding (as in
// wav2vec2's `no_padding` feature extractor), then floor((L - k)/s) + 1.
const LAYERS_1S: &[Layer] = &[
    Layer {
        name: "L0_k10s5",
        in_ch: 1,
        out_ch: 512,
        kernel: 10,
        stride: 5,
        input_len: 16000,
    },
    Layer {
        name: "L1_k3s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 3,
        stride: 2,
        input_len: 3199,
    },
    Layer {
        name: "L2_k3s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 3,
        stride: 2,
        input_len: 1599,
    },
    Layer {
        name: "L3_k3s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 3,
        stride: 2,
        input_len: 799,
    },
    Layer {
        name: "L4_k3s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 3,
        stride: 2,
        input_len: 399,
    },
    Layer {
        name: "L5_k2s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 2,
        stride: 2,
        input_len: 199,
    },
    Layer {
        name: "L6_k2s2",
        in_ch: 512,
        out_ch: 512,
        kernel: 2,
        stride: 2,
        input_len: 99,
    },
];

fn flex_input(batch: usize, ch: usize, len: usize) -> Tensor<Flex, 3> {
    let data = fill(batch * ch * len);
    Tensor::from_data(TensorData::new(data, [batch, ch, len]), &Default::default())
}

fn flex_kernel(out_ch: usize, in_ch: usize, k: usize) -> Tensor<Flex, 3> {
    let data = fill(out_ch * in_ch * k);
    Tensor::from_data(
        TensorData::new(data, [out_ch, in_ch, k]),
        &Default::default(),
    )
}

fn candle_input(batch: usize, ch: usize, len: usize) -> CandleTensor {
    let data = fill(batch * ch * len);
    CandleTensor::from_vec(data, (batch, ch, len), &CandleDevice::Cpu).unwrap()
}

fn candle_kernel(out_ch: usize, in_ch: usize, k: usize) -> CandleTensor {
    let data = fill(out_ch * in_ch * k);
    CandleTensor::from_vec(data, (out_ch, in_ch, k), &CandleDevice::Cpu).unwrap()
}

#[divan::bench_group(name = "flex/wav2vec2_1s")]
mod flex_w2v2 {
    use super::*;

    #[divan::bench(args = LAYERS_1S)]
    fn conv1d(bencher: Bencher, layer: &Layer) {
        let x = flex_input(1, layer.in_ch, layer.input_len);
        let w = flex_kernel(layer.out_ch, layer.in_ch, layer.kernel);
        let opts = ConvOptions::new([layer.stride], [0], [1], 1);
        bencher.bench(|| module::conv1d(x.clone(), w.clone(), None, opts.clone()));
    }
}

#[divan::bench_group(name = "candle/wav2vec2_1s")]
mod candle_w2v2 {
    use super::*;

    #[divan::bench(args = LAYERS_1S)]
    fn conv1d(bencher: Bencher, layer: &Layer) {
        let x = candle_input(1, layer.in_ch, layer.input_len);
        let w = candle_kernel(layer.out_ch, layer.in_ch, layer.kernel);
        let stride = layer.stride;
        bencher.bench(|| x.conv1d(&w, 0, stride, 1, 1).unwrap());
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

# Benchmarks: Flex vs Candle

Direct per-op comparison between `burn-flex` and `candle-core` on CPU, using pure-Rust kernels on
both sides (no BLAS, no Accelerate, no MKL). The goal is apples-to-apples kernel quality, not
end-to-end framework throughput.

**Hardware**: Apple M3 Max (12 perf + 4 efficiency cores, LPDDR5) **Date**: 2026-04-05 **Candle
version**: 0.10 (pinned in `crates/burn-flex-bench-candle/Cargo.toml`) **Burn-flex features**:
`std`, `simd`, `rayon` (defaults)

## How to Read

- **Median** time reported (lower is better)
- Numbers inside a cell are the median of 100 samples via [divan](https://docs.rs/divan)
- Bold marks the winner (no bold = within noise / tie)
- All inputs are byte-identical on both sides (same deterministic filler)

## Running

```sh
cargo bench -p burn-flex-bench-candle --bench matmul
cargo bench -p burn-flex-bench-candle --bench conv1d
cargo bench -p burn-flex-bench-candle --bench transformer_ops
```

---

## Transformer ops (wav2vec2-large shapes)

These run once per transformer layer (24 layers in wav2vec2-large), so any per-op gap is multiplied
24× in end-to-end inference. Two of the three are served by burn-flex's fused row kernels (softmax,
layer_norm); gelu uses burn-tensor's trait hook that both backends have.

### softmax (last axis)

| Shape                             | Flex (fused) | Flex (decomposed) | Candle | Flex vs Candle   |
| --------------------------------- | ------------ | ----------------- | ------ | ---------------- |
| `[16, 50, 50]` (attn, 1s audio)   | **65 µs**    | 655 µs            | 95 µs  | **1.46× faster** |
| `[16, 150, 150]` (attn, 3s audio) | **134 µs**   | 6.13 ms           | 194 µs | **1.45× faster** |

Flex decomposed is `burn_tensor::activation::softmax` — a 5-op decomposition into
`max_dim + sub + exp + sum_dim + div` with full-tensor intermediates. The fused path runs
`burn_flex::ops::activation::softmax`, a three-pass row kernel (max, exp+sum, normalize) dispatched
through macerator once per chunk of rows. See
[UPSTREAM_ISSUE.md](crates/burn-flex-bench-candle/UPSTREAM_ISSUE.md) for the proposal to add a
`softmax` hook to burn's `ActivationOps` trait so the fused path is available without a
crate-specific wrapper.

### layer_norm (last axis)

| Shape                            | Flex (fused) | Flex (decomposed) | Candle | Flex vs Candle   |
| -------------------------------- | ------------ | ----------------- | ------ | ---------------- |
| `[50, 1024]` (hidden, 1s audio)  | **19 µs**    | 137 µs            | 68 µs  | **3.59× faster** |
| `[150, 1024]` (hidden, 3s audio) | **55 µs**    | 393 µs            | 88 µs  | **1.60× faster** |

Same story as softmax. The fused kernel does two passes per row (Welford-style mean/M2, then
normalize+affine), gamma and beta are read once per element per chunk and stay in L1. Decomposed
path is `burn::nn::LayerNorm::forward`, which expands into six primitive ops with the full 6-buffer
intermediate chain.

### gelu (tanh approximation)

| Shape                         | Flex    | Candle  |
| ----------------------------- | ------- | ------- |
| `[50, 4096]` (ffn inter, 1s)  | 396 µs  | 375 µs  |
| `[150, 4096]` (ffn inter, 3s) | 1.14 ms | 1.13 ms |
| `[50, 1024]` (hidden, 1s)     | 93 µs   | 93 µs   |
| `[150, 1024]` (hidden, 3s)    | 295 µs  | 284 µs  |

Tied everywhere. gelu is the proof that the softmax/layer_norm gap is architectural, not
kernel-quality: `ActivationOps::gelu` already exists as a trait hook, so burn-flex's optimized impl
runs directly with no wrapper.

---

## Matmul (f32)

### Square

| N    | Flex        | Candle  | Flex vs Candle      |
| ---- | ----------- | ------- | ------------------- |
| 128  | **44 µs**   | 103 µs  | **2.35× faster**    |
| 256  | 177 µs      | 171 µs  | tied (1.04× slower) |
| 512  | 610 µs      | 613 µs  | tied                |
| 1024 | **2.56 ms** | 2.69 ms | **1.05× faster**    |

Flex wins decisively at N=128 (gemm crate's small-matrix microkernel beats candle's matmul path) and
ties or wins everywhere else. The 256 case is within single-sample noise.

### Transformer shapes (wav2vec2-large)

| Op                                      | Flex        | Candle  | Flex vs Candle   |
| --------------------------------------- | ----------- | ------- | ---------------- |
| qkv_proj, 1s `[50, 1024]×[1024, 3072]`  | **464 µs**  | 490 µs  | **1.06× faster** |
| qkv_proj, 3s `[150, 1024]×[1024, 3072]` | **660 µs**  | 678 µs  | 1.03× faster     |
| ffn_up, 1s `[50, 1024]×[1024, 4096]`    | 1.86 ms     | 1.87 ms | tied             |
| ffn_up, 3s `[150, 1024]×[1024, 4096]`   | 1.81 ms     | 1.85 ms | tied             |
| ffn_down, 1s `[50, 4096]×[4096, 1024]`  | **1.19 ms** | 1.25 ms | **1.05× faster** |
| ffn_down, 3s `[150, 4096]×[4096, 1024]` | 2.35 ms     | 2.38 ms | tied             |

Both backends delegate to heavily-tuned gemm routines (gemm crate on the flex side, candle-core's
internal matmul on the other). This is close to the theoretical ceiling for pure-Rust f32 gemm on M3
Max; the win happens in the ops that surround gemm.

---

## Conv1d (wav2vec2-large feature extractor, 1s of 16 kHz audio)

Seven strided conv1d layers that downsample raw audio 320×. L0 dominates the wallclock because of
its massive input (16000 samples) and wide output channels (512).

| Layer     | Shape                  | Flex         | Candle       | Winner         |
| --------- | ---------------------- | ------------ | ------------ | -------------- |
| L0        | `1×1×16000`, k=10, s=5 | **511 µs**   | 3.60 ms      | **flex 7.04×** |
| L1        | `1×512×3199`, k=3, s=2 | 6.78 ms      | 6.51 ms      | candle 1.04×   |
| L2        | `1×512×1599`, k=3, s=2 | **3.32 ms**  | 3.59 ms      | **flex 1.08×** |
| L3        | `1×512×799`, k=3, s=2  | 2.46 ms      | 2.07 ms      | candle 1.19×   |
| L4        | `1×512×399`, k=3, s=2  | 1.70 ms      | 1.28 ms      | candle 1.33×   |
| L5        | `1×512×199`, k=2, s=2  | 974 µs       | 732 µs       | candle 1.33×   |
| L6        | `1×512×99`, k=2, s=2   | 777 µs       | 494 µs       | candle 1.57×   |
| **Total** |                        | **16.55 ms** | **18.27 ms** | **flex 1.10×** |

L0 is saved by [a weight-transpose fix](crates/burn-flex/src/ops/conv.rs) that rewrote the
`(c_in, K) → (K, c_in)` transpose as a tight `K`-inner loop that LLVM autovectorizes; for
wide-output conv that loop had been the dominant cost. L3-L6 is the remaining gap: candle has a
direct vec_dot-per-output path that beats tiled im2col+gemm on small spatial shapes. Overall, flex
wins the full feature extractor by ~10% because L0's 3 ms savings dwarfs the L3-L6 loss. Tracked at
[antimora/burn-flex#34](https://github.com/antimora/burn-flex/issues/34).

---

## wav2vec2-large per-forward-pass impact

Combining per-op numbers into a forward-pass estimate at 3s audio. Conv stack runs once; each
transformer layer runs softmax + 2× layer_norm × 24 layers.

| Component                                  | Flex         | Candle       | Delta per forward |
| ------------------------------------------ | ------------ | ------------ | ----------------- |
| Conv1d feature extractor (7 layers)        | 16.55 ms     | 18.27 ms     | **−1.72 ms**      |
| Softmax × 24 layers `[16, 150, 150]`       | 3.22 ms      | 4.66 ms      | **−1.44 ms**      |
| Layer_norm × 48 calls `[150, 1024]`        | 2.64 ms      | 4.22 ms      | **−1.58 ms**      |
| **Subtotal (softmax + layer_norm + conv)** | **22.41 ms** | **27.15 ms** | **−4.74 ms**      |

The rest of the forward pass (matmul, gelu, attention) is tied or a small flex win, so the 4.7 ms
wallclock savings from just these three op families drives the overall wav2vec2-large inference
advantage vs candle on the same hardware.

---

## Notes

- **No BLAS**: Both backends are pure-Rust. Candle-accelerate (Apple BLAS) beats this flex build on
  matmul; that comparison is not the point of this crate.
- **Numbers move**: Benchmark numbers shift with compiler versions, candle versions, and OS noise.
  Treat ratios as stable, absolute values as a specific-point-in-time snapshot.
- **No warmup caveat**: divan already does warmup/calibration per sample. The first few samples are
  excluded from the median.

See also:

- [`crates/burn-flex-bench-candle/README.md`](crates/burn-flex-bench-candle/README.md) for the crate
  layout and how to extend the benches.
- [`crates/burn-flex-bench-candle/UPSTREAM_ISSUE.md`](crates/burn-flex-bench-candle/UPSTREAM_ISSUE.md)
  for the proposal to add fused softmax/layer_norm hooks to burn-backend.
- [BENCHMARKS.md](BENCHMARKS.md) for the flex vs burn-ndarray comparison.

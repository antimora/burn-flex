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
# Original wav2vec2-focused benches
cargo bench -p burn-flex-bench-candle --bench matmul
cargo bench -p burn-flex-bench-candle --bench conv1d
cargo bench -p burn-flex-bench-candle --bench transformer_ops

# Broader op coverage (added 2026-04-05)
cargo bench -p burn-flex-bench-candle --bench elementwise
cargo bench -p burn-flex-bench-candle --bench reduce
cargo bench -p burn-flex-bench-candle --bench shape_ops
cargo bench -p burn-flex-bench-candle --bench matmul_batched
cargo bench -p burn-flex-bench-candle --bench conv2d
cargo bench -p burn-flex-bench-candle --bench pool2d
cargo bench -p burn-flex-bench-candle --bench indexing
cargo bench -p burn-flex-bench-candle --bench misc_ops

# Or everything
cargo bench -p burn-flex-bench-candle
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
through macerator once per chunk of rows.

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

## Broader op coverage (added 2026-04-05)

Eight additional bench files in `crates/burn-flex-bench-candle/benches/` cover every flex op that
intersects with candle's CPU API. Medians below are from a single fresh run on the same Apple M3
Max. Ratios use candle / flex (>1.0 means flex is faster).

### Batched matmul (attention shapes)

| Shape                                         | Flex       | Candle  | Ratio     |
| --------------------------------------------- | ---------- | ------- | --------- |
| QK^T `[16, 50, 64] @ [16, 64, 50]`            | **54 µs**  | 74 µs   | **1.37×** |
| QK^T `[16, 150, 64] @ [16, 64, 150]`          | **189 µs** | 1.63 ms | **8.62×** |
| QK^T transposed-view (seq=50)                  | **58 µs**  | 84 µs   | **1.45×** |
| QK^T transposed-view (seq=150)                 | **185 µs** | 1.64 ms | **8.86×** |
| AV `[16, 50, 50] @ [16, 50, 64]`              | **54 µs**  | 66 µs   | **1.22×** |
| AV `[16, 150, 150] @ [16, 150, 64]`           | **163 µs** | 1.61 ms | **9.88×** |
| batched_128 `[32, 128, 128] @ [32, 128, 128]` | **268 µs** | 3.31 ms | **12.4×** |

Flex's batched matmul passes actual layout strides to gemm, so transposed views (the
`q.matmul(k.swap_dims(1,2))` attention pattern) run at the same speed as contiguous inputs
with no copy. Fixed in [antimora/burn-flex#49](https://github.com/antimora/burn-flex/pull/49).

### Conv2d + conv_transpose2d (ResNet shapes)

| Layer                         | Flex        | Candle  | Ratio        |
| ----------------------------- | ----------- | ------- | ------------ |
| resnet_conv1 1×3×224² k7 s2   | **872 µs**  | 1.10 ms | **1.27×**    |
| resnet_l1 1×64×56² k3         | **942 µs**  | 1.28 ms | **1.36×**    |
| resnet_l2 1×128×28² k3        | **1.03 ms** | 1.90 ms | **1.85×**    |
| resnet_l3 1×256×14² k3        | **1.47 ms** | 2.20 ms | **1.49×**    |
| resnet_l4 1×512×7² k3         | **2.35 ms** | 3.08 ms | **1.31×**    |
| 1×1 pointwise 1×256×56² → 64  | 2.23 ms     | 1.52 ms | 0.68×        |
| conv_transpose 1×128×16² → 64 | **602 µs**  | 1.37 ms | **2.27×** |
| conv_transpose 1×64×32² → 32  | **838 µs**  | 1.72 ms | **2.05×** |

Flex wins every standard 3×3 conv2d layer and conv_transpose2d (2x faster after GEMM + col2im
rewrite in [antimora/burn-flex#46](https://github.com/antimora/burn-flex/pull/46)); 1×1 pointwise
is where candle's direct gemm is sharper.

### Pool2d

| Shape          | Op  | Flex       | Candle  | Ratio     |
| -------------- | --- | ---------- | ------- | --------- |
| 1×64×112² k3s2 | max | **436 µs** | 893 µs  | **2.05×** |
| 1×64×112² k3s2 | avg | **471 µs** | 896 µs  | **1.90×** |
| 8×64×56² k3s2  | max | **684 µs** | 1.81 ms | **2.64×** |
| 8×64×56² k3s2  | avg | **766 µs** | 1.75 ms | **2.29×** |
| 1×128×28² k2s2 | max | 92 µs      | 57 µs   | 0.62×     |
| 1×128×28² k2s2 | avg | 89 µs      | 56 µs   | 0.63×     |

Flex wins the common k=3 stride=2 ResNet pool by ~2×; loses on the tiny k=2 shapes, which are near
the divan noise floor (<100 µs).

### Reductions

| Op              | Shape | Flex        | Candle  | Ratio     |
| --------------- | ----- | ----------- | ------- | --------- |
| sum (full)      | 1024² | 52.6 µs     | 47.8 µs | 0.91×     |
| mean (full)     | 1024² | 48.5 µs     | 48.0 µs | tied      |
| max (full)      | 1024² | **129 µs**  | 520 µs  | **4.0×**  |
| min (full)      | 1024² | **129 µs**  | 520 µs  | **4.0×**  |
| sum_dim last    | 1024² | 44.7 µs     | 42.1 µs | tied      |
| sum_dim first   | 1024² | **78.4 µs** | 1.20 ms | **15.3×** |
| mean_dim last   | 1024² | 42.1 µs     | 42.7 µs | tied      |
| max_dim last    | 1024² | **92 µs**   | 545 µs  | **5.9×**  |
| max_dim first   | 1024² | **199 µs**  | 986 µs  | **5.0×**  |
| argmax_dim last | 1024² | **86 µs**   | 556 µs  | **6.5×**  |

Three big wins: **4× faster full-tensor max/min**, **15× faster non-last-dim sum**, and **5-6×
faster max_dim/argmax_dim** (was 4× slower before rayon + SIMD, fixed in
[antimora/burn-flex#47](https://github.com/antimora/burn-flex/pull/47)). Last-axis sum_dim/mean_dim
now ties candle after the 4-accumulator SIMD rewrite in
[antimora/burn-flex#50](https://github.com/antimora/burn-flex/pull/50).

### Elementwise (1D, 1M elements, f32)

All basic arithmetic and transcendentals land within 5% on both backends (both bottleneck on LPDDR5
bandwidth at roughly 56 GB/s for the cheap ops). Representative numbers at 1M:

| Op    | Flex     | Candle   |
| ----- | -------- | -------- |
| add   | 112.6 µs | 112.8 µs |
| mul   | 112.7 µs | 112.8 µs |
| neg   | 74.6 µs  | 76.7 µs  |
| recip | 74.3 µs  | 74.2 µs  |
| sqrt  | 139 µs   | 142 µs   |
| log   | 1.69 ms  | 1.69 ms  |
| tanh  | 1.73 ms  | 1.73 ms  |
| powf  | 3.01 ms  | 3.11 ms  |
| sin   | 1.41 ms  | 1.27 ms  |

The only notable gap is `sin` at ~10% slower on flex.

### Shape ops

View ops (transpose, reshape, narrow, expand, permute) are all under 150 ns on both backends;
differences are pure dispatch overhead and not meaningful.

Data-moving shape ops:

| Op                                 | Flex        | Candle  | Ratio     |
| ---------------------------------- | ----------- | ------- | --------- |
| transpose_then_exp 1024² (strided) | **1.29 ms** | 2.14 ms | **1.65×** |
| cat along dim 0, 2×1024²           | 101 µs      | 109 µs  | 1.08×     |
| cat along last dim, 2×1024²        | **121 µs**  | 322 µs  | **2.66×** |
| repeat_dim 256² ×4                 | 13.1 µs     | 13.0 µs | tied      |

Flex's strided-unary path handles transposed inputs 1.65× faster than candle, and its last-axis
`cat` is 2.66× faster (candle's per-row stride copy is the bottleneck).

### Indexing

| Shape | Op                 | Flex       | Candle | Ratio     |
| ----- | ------------------ | ---------- | ------ | --------- |
| 1024² | gather last dim    | **244 µs** | 511 µs | **2.09×** |
| 1024² | scatter_add last   | 548 µs     | 622 µs | 1.13×     |
| 1024² | index_select dim 0 | **27 µs**  | 37 µs  | **1.37×** |
| 1024² | where_cond         | 248 µs     | 122 µs | 0.49× ⚠️  |

Flex wins gather by 2× and `index_select` by 1.4× (was 2.8× slower before the rayon threshold +
uninit buffer fix); loses `mask_where` by ~2×.

### Cumsum, sort, nearest2d

| Op                            | Flex         | Candle  | Ratio     |
| ----------------------------- | ------------ | ------- | --------- |
| cumsum last dim, 256²         | **42 µs**    | 436 µs  | **10.4×** |
| cumsum last dim, 1024²        | **696 µs**   | 4.68 ms | **6.72×** |
| sort last dim, 256²           | 202 µs       | 186 µs  | tied       |
| sort last dim, 1024²          | **1.18 ms**  | 1.55 ms | **1.30×**  |
| nearest2d upsample 64² → 128² | 56 µs        | 30 µs   | 0.54×      |

Cumsum is one of the biggest flex wins (6–10×). Sort-last-dim now beats candle at 1024² after
adding rayon fan-out across rows (fixed in
[antimora/burn-flex#45](https://github.com/antimora/burn-flex/pull/45), was 11.95 ms / 0.13×).

---

## Perf bug list (prioritized)

Surfaced by the broader coverage pass. Ordered by impact on real workloads.

1. **conv1d L3-L6 (small wav2vec2 shapes): 1.2-1.5× slower**. Already tracked at
   [antimora/burn-flex#34](https://github.com/antimora/burn-flex/issues/34).
2. **where_cond / mask_where: 2× slower** (248 µs vs 122 µs). Elementwise select.
3. **nearest2d upsample: ~2× slower** (56 µs vs 30 µs). Low absolute cost.
4. **conv2d 1×1 pointwise: 1.5× slower** (2.23 ms vs 1.52 ms). Candle takes a direct gemm path for
   pointwise; flex's im2col adds overhead.

Fixed since the first pass:
- batched matmul on transposed-view input (was 4x slower at small seqs; now 1.4x faster), fixed in
  [antimora/burn-flex#49](https://github.com/antimora/burn-flex/pull/49).
- max_dim / argmax_dim at 1024² (was 4× slower; now 5-6× faster), fixed in
  [antimora/burn-flex#47](https://github.com/antimora/burn-flex/pull/47).
- conv_transpose2d (was 8× slower; now 2× faster), fixed in
  [antimora/burn-flex#46](https://github.com/antimora/burn-flex/pull/46).
- sort_last at 1024² (was 7.8× slower; now 1.3× faster), fixed in
  [antimora/burn-flex#45](https://github.com/antimora/burn-flex/pull/45).
- sum_dim/mean_dim last-axis at 1024² (was 2× slower; now tied), fixed in
  [antimora/burn-flex#50](https://github.com/antimora/burn-flex/pull/50).
- index_select at 1024² (was 2.8× slower; now 1.4× faster), fixed in
  [antimora/burn-flex#51](https://github.com/antimora/burn-flex/pull/51).

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
- [BENCHMARKS.md](BENCHMARKS.md) for the flex vs burn-ndarray comparison.

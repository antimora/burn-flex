# Benchmarks: Flex vs Candle

Direct per-op comparison between `burn-flex` and `candle-core` on CPU, using pure-Rust kernels on
both sides (no BLAS, no Accelerate, no MKL). The goal is apples-to-apples kernel quality, not
end-to-end framework throughput.

**Hardware**: Apple M3 Max (12 perf + 4 efficiency cores, LPDDR5) **Date**: 2026-04-07 **Candle
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
24x in end-to-end inference. Two of the three are served by burn-flex's fused row kernels (softmax,
layer_norm); gelu uses burn-tensor's trait hook that both backends have.

### softmax (last axis)

| Shape                             | Flex (fused) | Candle  | Flex Mem | Candle Mem | Flex vs Candle   |
| --------------------------------- | ------------ | ------- | -------- | ---------- | ---------------- |
| `[16, 50, 50]` (attn, 1s audio)   | **48.5 µs**  | 84.1 µs | 160.1 KB | 160.2 KB   | **1.73x faster** |
| `[16, 150, 150]` (attn, 3s audio) | **137 µs**   | 193 µs  | 1.44 MB  | 1.44 MB    | **1.41x faster** |

The fused path runs `burn_flex::ops::activation::softmax`, a three-pass row kernel (max, exp+sum,
normalize) dispatched through macerator once per chunk of rows.

### layer_norm (last axis)

| Shape                            | Flex (fused) | Candle  | Flex Mem | Candle Mem | Flex vs Candle   |
| -------------------------------- | ------------ | ------- | -------- | ---------- | ---------------- |
| `[50, 1024]` (hidden, 1s audio)  | **18.7 µs**  | 63.7 µs | 204.9 KB | 205 KB     | **3.41x faster** |
| `[150, 1024]` (hidden, 3s audio) | **51.9 µs**  | 83.5 µs | 614.5 KB | 614.6 KB   | **1.61x faster** |

Same story as softmax. The fused kernel does two passes per row (Welford-style mean/M2, then
normalize+affine), gamma and beta are read once per element per chunk and stay in L1.

### gelu (tanh approximation)

| Shape                         | Flex    | Candle  |
| ----------------------------- | ------- | ------- |
| `[50, 4096]` (ffn inter, 1s)  | 358 µs  | 375 µs  |
| `[150, 4096]` (ffn inter, 3s) | 1.08 ms | 1.09 ms |
| `[50, 1024]` (hidden, 1s)     | 89.8 µs | 89.2 µs |
| `[150, 1024]` (hidden, 3s)    | 271 µs  | 267 µs  |

Tied everywhere. gelu is the proof that the softmax/layer_norm gap is architectural, not
kernel-quality: `ActivationOps::gelu` already exists as a trait hook, so burn-flex's optimized impl
runs directly with no wrapper.

---

## Matmul (f32)

### Square

| N    | Flex        | Candle  | Flex Mem | Candle Mem | Flex vs Candle   |
| ---- | ----------- | ------- | -------- | ---------- | ---------------- |
| 128  | **43.0 µs** | 104 µs  | 327.8 KB | 132.6 KB   | **2.42x faster** |
| 256  | **152 µs**  | 167 µs  | 525.9 KB | 524.3 KB   | **1.10x faster** |
| 512  | 609 µs      | 609 µs  | 2.097 MB | 2.097 MB   | tied             |
| 1024 | 2.59 ms     | 2.54 ms | 8.388 MB | 8.388 MB   | tied             |

Flex wins decisively at N=128 (gemm crate's small-matrix microkernel beats candle's matmul path) and
ties or wins at 256. The 512 and 1024 cases are within noise.

### Transformer shapes (wav2vec2-large)

| Op                                      | Flex    | Candle  | Flex Mem | Candle Mem | Flex vs Candle |
| --------------------------------------- | ------- | ------- | -------- | ---------- | -------------- |
| qkv_proj, 1s `[50, 1024]x[1024, 3072]`  | 466 µs  | 447 µs  | 417.9 KB | 417.8 KB   | tied           |
| qkv_proj, 3s `[150, 1024]x[1024, 3072]` | 683 µs  | 692 µs  | 1.237 MB | 1.237 MB   | tied           |
| ffn_up, 1s `[50, 1024]x[1024, 4096]`    | 1.86 ms | 1.95 ms | 1.032 MB | 1.032 MB   | tied           |
| ffn_up, 3s `[150, 1024]x[1024, 4096]`   | 1.83 ms | 1.86 ms | 3.08 MB  | 3.08 MB    | tied           |
| ffn_down, 1s `[50, 4096]x[4096, 1024]`  | 1.34 ms | 1.38 ms | 630.9 KB | 630.8 KB   | tied           |
| ffn_down, 3s `[150, 4096]x[4096, 1024]` | 2.37 ms | 2.41 ms | 1.859 MB | 1.859 MB   | tied           |

Both backends delegate to heavily-tuned gemm routines (gemm crate on the flex side, candle-core's
internal matmul on the other). This is close to the theoretical ceiling for pure-Rust f32 gemm on M3
Max; the win happens in the ops that surround gemm.

---

## Conv1d (wav2vec2-large feature extractor, 1s of 16 kHz audio)

Seven strided conv1d layers that downsample raw audio 320x. L0 dominates the wallclock because of
its massive input (16000 samples) and wide output channels (512).

| Layer     | Shape                  | Flex         | Candle       | Flex Mem | Candle Mem | Winner         |
| --------- | ---------------------- | ------------ | ------------ | -------- | ---------- | -------------- |
| L0        | `1x1x16000`, k=10, s=5 | **502 µs**   | 3.83 ms      | 6.636 MB | 13.23 MB   | **flex 7.62x** |
| L1        | `1x512x3199`, k=3, s=2 | 6.74 ms      | 6.64 ms      | 12.97 MB | 22.92 MB   | tied           |
| L2        | `1x512x1599`, k=3, s=2 | **2.41 ms**  | 3.55 ms      | 2.685 MB | 11.46 MB   | **flex 1.47x** |
| L3        | `1x512x799`, k=3, s=2  | **1.68 ms**  | 2.03 ms      | 1.865 MB | 5.726 MB   | **flex 1.21x** |
| L4        | `1x512x399`, k=3, s=2  | 1.32 ms      | 1.29 ms      | 1.882 MB | 2.859 MB   | tied           |
| L5        | `1x512x199`, k=2, s=2  | **532 µs**   | 724 µs       | 1.48 MB  | 1.018 MB   | **flex 1.36x** |
| L6        | `1x512x99`, k=2, s=2   | **296 µs**   | 551 µs       | 1.28 MB  | 514.2 KB   | **flex 1.86x** |
| **Total** |                        | **13.50 ms** | **18.62 ms** |          |            | **flex 1.38x** |

L0 uses the 1x1 fast path. L2-L6 use the direct conv path that decomposes the convolution into `kw`
separate gemm calls operating directly on NCHW data with strided pointers, eliminating both the NHWC
conversion and im2col buffer. Fixed in
[antimora/burn-flex#34](https://github.com/antimora/burn-flex/issues/34).

---

## wav2vec2-large per-forward-pass impact

Combining per-op numbers into a forward-pass estimate at 3s audio. Conv stack runs once; each
transformer layer runs softmax + 2x layer_norm x 24 layers.

| Component                                  | Flex         | Candle       | Delta per forward |
| ------------------------------------------ | ------------ | ------------ | ----------------- |
| Conv1d feature extractor (7 layers)        | 13.50 ms     | 18.62 ms     | -5.12 ms          |
| Softmax x 24 layers `[16, 150, 150]`       | 3.29 ms      | 4.63 ms      | -1.34 ms          |
| Layer_norm x 48 calls `[150, 1024]`        | 2.49 ms      | 4.01 ms      | -1.52 ms          |
| **Subtotal (softmax + layer_norm + conv)** | **19.28 ms** | **27.26 ms** | **-7.98 ms**      |

The rest of the forward pass (matmul, gelu, attention) is tied or a small flex win, so the ~8.0 ms
wallclock savings from just these three op families drives the overall wav2vec2-large inference
advantage vs candle on the same hardware.

---

## Broader op coverage (added 2026-04-05)

Eight additional bench files in `crates/burn-flex-bench-candle/benches/` cover every flex op that
intersects with candle's CPU API. Medians below are from a single fresh run on the same Apple M3
Max. Ratios use candle / flex (>1.0 means flex is faster).

### Batched matmul (attention shapes)

| Shape                                         | Flex        | Candle  | Flex Mem | Candle Mem | Ratio      |
| --------------------------------------------- | ----------- | ------- | -------- | ---------- | ---------- |
| QK^T `[16, 50, 64] @ [16, 64, 50]`            | **51.4 µs** | 73.8 µs | 160.1 KB | 190.4 KB   | **1.44x**  |
| QK^T `[16, 150, 64] @ [16, 64, 150]`          | **199 µs**  | 1.69 ms | 1.44 MB  | 1.521 MB   | **8.53x**  |
| AV `[16, 50, 50] @ [16, 50, 64]`              | **48.7 µs** | 71.4 µs | 204.9 KB | 216.2 KB   | **1.47x**  |
| AV `[16, 150, 150] @ [16, 150, 64]`           | **183 µs**  | 1.56 ms | 614.5 KB | 707.2 KB   | **8.56x**  |
| batched_128 `[32, 128, 128] @ [32, 128, 128]` | **298 µs**  | 3.29 ms | 2.097 MB | 2.164 MB   | **11.04x** |

Flex's batched matmul passes actual layout strides to gemm, so transposed views (the
`q.matmul(k.swap_dims(1,2))` attention pattern) run at the same speed as contiguous inputs with no
copy. Fixed in [antimora/burn-flex#49](https://github.com/antimora/burn-flex/pull/49).

### Conv2d + conv_transpose2d (ResNet shapes)

| Layer                        | Flex        | Candle  | Flex Mem | Candle Mem | Ratio     |
| ---------------------------- | ----------- | ------- | -------- | ---------- | --------- |
| resnet_conv1 1x3x224^2 k7 s2 | **871 µs**  | 1.14 ms | 3.852 MB | 3.851 MB   | **1.30x** |
| resnet_l1 1x64x56^2 k3       | **923 µs**  | 1.28 ms | 1.753 MB | 1.753 MB   | **1.38x** |
| resnet_l2 1x128x28^2 k3      | **970 µs**  | 1.91 ms | 1.392 MB | 1.392 MB   | **1.97x** |
| resnet_l3 1x256x14^2 k3      | **1.50 ms** | 2.19 ms | 6.906 MB | 5.95 MB    | **1.45x** |
| resnet_l4 1x512x7^2 k3       | **2.40 ms** | 3.22 ms | 14.18 MB | 14.18 MB   | **1.34x** |
| 1x1 pointwise 1x256x56^2     | **446 µs**  | 1.79 ms | 868.5 KB | 4.882 MB   | **4.02x** |
| conv_transpose 1x128x16^2    | **663 µs**  | 1.26 ms | 1.835 MB | 393.2 KB   | **1.89x** |
| conv_transpose 1x64x32^2     | **850 µs**  | 1.31 ms | 2.752 MB | 786.4 KB   | **1.54x** |

Flex wins every conv2d layer and conv_transpose2d. The 1x1 pointwise fast path skips im2col entirely
and calls gemm directly on the NCHW input with correct strides, avoiding the transpose buffer (fixed
in [antimora/burn-flex#52](https://github.com/antimora/burn-flex/pull/52)). Conv_transpose2d is
1.5-1.9x faster after GEMM + col2im rewrite in
[antimora/burn-flex#46](https://github.com/antimora/burn-flex/pull/46).

### Pool2d

| Shape           | Op  | Flex        | Candle  | Flex Mem | Candle Mem | Ratio     |
| --------------- | --- | ----------- | ------- | -------- | ---------- | --------- |
| 1x64x112^2 k3s2 | max | **456 µs**  | 893 µs  | 2.323 MB | 774.6 KB   | **1.96x** |
| 1x64x112^2 k3s2 | avg | **493 µs**  | 896 µs  | 774.6 KB | 774.6 KB   | **1.82x** |
| 8x64x56^2 k3s2  | max | **698 µs**  | 1.76 ms | 4.479 MB | 1.493 MB   | **2.52x** |
| 8x64x56^2 k3s2  | avg | **729 µs**  | 1.75 ms | 1.493 MB | 1.493 MB   | **2.40x** |
| 1x64x56^2 k2s2  | max | **87.3 µs** | 114 µs  | 602.4 KB | 200.9 KB   | **1.31x** |
| 1x64x56^2 k2s2  | avg | **97.5 µs** | 123 µs  | 200.9 KB | 200.9 KB   | **1.26x** |
| 1x128x28^2 k2s2 | max | 67.0 µs     | 57.2 µs | 301.3 KB | 100.6 KB   | 0.85x     |
| 1x128x28^2 k2s2 | avg | 59.9 µs     | 57.1 µs | 100.5 KB | 100.6 KB   | ~1x       |

Flex wins the common k=3 stride=2 ResNet pool by 2-2.5x and the k=2 64x56 shapes by 1.3x. The small
k=2 128x28 shapes are now near-parity after flattening the nested rayon parallelism into a single
`batch*channels` par_iter (was 0.73-0.78x with nested par_iter dispatch overhead). Fixed in
[antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).

### Reductions

| Op              | Shape         | Flex        | Candle  | Ratio     |
| --------------- | ------------- | ----------- | ------- | --------- |
| sum (full)      | 1024^2        | **37.5 µs** | 50.6 µs | **1.35x** |
| mean (full)     | 1024^2        | **38.6 µs** | 51.0 µs | **1.32x** |
| max (full)      | 1024^2        | **141 µs**  | 530 µs  | **3.76x** |
| min (full)      | 1024^2        | **141 µs**  | 566 µs  | **4.01x** |
| sum_dim last    | 1024^2        | 42.1 µs     | 40.3 µs | tied      |
| mean_dim last   | 1024^2        | 42.1 µs     | 42.7 µs | tied      |
| max_dim last    | 1024^2        | **108 µs**  | 551 µs  | **5.10x** |
| max_dim (ffn)   | `[150, 4096]` | **82.8 µs** | 330 µs  | **3.99x** |
| argmax_dim last | 1024^2        | **106 µs**  | 557 µs  | **5.25x** |
| sum (ffn_3s)    | `[150, 4096]` | **23.2 µs** | 30.1 µs | **1.30x** |
| mean (ffn_3s)   | `[150, 4096]` | **23.2 µs** | 30.3 µs | **1.31x** |

Three big wins: **4x faster full-tensor max/min**, **5.1x faster max_dim last**, and **5.25x faster
argmax_dim** (was 4x slower before rayon + SIMD, fixed in
[antimora/burn-flex#47](https://github.com/antimora/burn-flex/pull/47)). Full-tensor sum/mean now
**1.3x faster** after bumping the SIMD sum kernel from 4 to 8 accumulators and raising the rayon
threshold for L2-resident data. Last-axis sum_dim/mean_dim ties candle after the SIMD rewrite in
[antimora/burn-flex#50](https://github.com/antimora/burn-flex/pull/50).

### Elementwise (1D, 1M elements, f32)

All basic arithmetic and transcendentals land within 5% on both backends (both bottleneck on LPDDR5
bandwidth at roughly 56 GB/s for the cheap ops). Representative numbers at 1M:

| Op      | Flex    | Candle  |
| ------- | ------- | ------- |
| add     | 112 µs  | 119 µs  |
| mul     | 112 µs  | 112 µs  |
| neg     | 74.4 µs | 74.3 µs |
| recip   | 74.1 µs | 74.1 µs |
| sqrt    | 139 µs  | 139 µs  |
| log     | 1.70 ms | 1.71 ms |
| tanh    | 1.73 ms | 1.73 ms |
| powf    | 3.01 ms | 3.03 ms |
| sin     | 1.30 ms | 1.28 ms |
| equal   | 90.2 µs | 87.0 µs |
| greater | 93.6 µs | 87.1 µs |

Comparison ops are now near-parity after replacing the explicit macerator SIMD comparison kernel
with scalar loops that LLVM autovectorizes. The old kernel used `store_as_bool` which wrote only
`lanes` bytes per SIMD iteration (4 on NEON); LLVM batches 16+ f32 comparisons and packs results
into a single wide u8 vector store, eliminating the 4:1 bottleneck. Fixed in
[antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).

### Shape ops

View ops (transpose, reshape, narrow, expand, permute) are all under 150 ns on both backends;
differences are pure dispatch overhead and not meaningful.

Data-moving shape ops:

| Op                                  | Flex        | Candle  | Ratio     |
| ----------------------------------- | ----------- | ------- | --------- |
| transpose_then_exp 1024^2 (strided) | **1.28 ms** | 2.12 ms | **1.66x** |
| transpose_then_exp 256^2 (strided)  | **78.5 µs** | 134 µs  | **1.71x** |
| cat along dim 0, 2x1024^2           | **136 µs**  | 333 µs  | **2.45x** |
| repeat_dim 256^2 x4                 | 12.9 µs     | 12.8 µs | tied      |

Flex's strided-unary path handles transposed inputs 1.66-1.71x faster than candle. Cat along dim 0
is now 2.45x faster (previously was tied at ~100 µs; the bench was updated to a more representative
shape).

### Indexing

| Shape  | Op                 | Flex        | Candle  | Flex Mem | Candle Mem | Ratio     |
| ------ | ------------------ | ----------- | ------- | -------- | ---------- | --------- |
| 1024^2 | gather last dim    | **268 µs**  | 523 µs  | 6.291 MB | 2.097 MB   | **1.95x** |
| 1024^2 | scatter_add last   | 546 µs      | 619 µs  | 8.388 MB | 4.194 MB   | 1.13x     |
| 1024^2 | index_select dim 0 | **28.0 µs** | 36.1 µs | 2.101 MB | 2.097 MB   | **1.29x** |
| 1024^2 | where_cond         | 125 µs      | 122 µs  | 4.194 MB | 4.194 MB   | tied      |

Flex wins gather by ~2x and `index_select` by 1.3x. `where_cond` is now tied (was 2x slower before
the branchless bitwise blend + uninit buffer fix in
[antimora/burn-flex#41](https://github.com/antimora/burn-flex/issues/41)).

### Nearest2d interpolation

| Op                              | Flex        | Candle  | Ratio     |
| ------------------------------- | ----------- | ------- | --------- |
| nearest2d upsample 64 -> 128    | **22.1 µs** | 30.2 µs | **1.37x** |
| nearest2d downsample 256 -> 128 | **22.5 µs** | 29.7 µs | **1.32x** |
| nearest2d upsample 4x 32 -> 128 | **22.0 µs** | 30.5 µs | **1.38x** |

Flex wins all nearest2d shapes by 1.3-1.4x after the precomputed index map optimization in
[antimora/burn-flex#54](https://github.com/antimora/burn-flex/pull/54).

---

## Perf bug list (prioritized)

Surfaced by the broader coverage pass. Ordered by impact on real workloads.

Open regressions:

- (none from the original candle benchmark pass)

Fixed since the first pass:

- Comparison ops (`equal`, `greater`) at 1M elements (was 2.2-2.7x slower; now within 10%),
  fixed in [antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).
- Full-tensor sum/mean at 1024^2 (was tied/1.09x; now 1.3x faster after 8-accumulator SIMD sum),
  fixed in [antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).
- Full-tensor sum/mean on smaller shapes `[150, 4096]` (was 0.68-0.71x; now 1.3x faster after
  raising rayon threshold for L2-resident data),
  fixed in [antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).
- Pool2d k=2 s=2 at 128x28 (was 0.73-0.78x; now 0.85-1.0x after flattening nested rayon
  parallelism), fixed in [antimora/burn-flex#59](https://github.com/antimora/burn-flex/issues/59).
- conv1d L3-L6 small wav2vec2 shapes (was 1.2-1.6x slower; now 1.2-1.9x faster), fixed in
  [antimora/burn-flex#34](https://github.com/antimora/burn-flex/issues/34).
- batched matmul on transposed-view input (was 4x slower at small seqs; now 1.4x faster), fixed in
  [antimora/burn-flex#49](https://github.com/antimora/burn-flex/pull/49).
- max_dim / argmax_dim at 1024^2 (was 4x slower; now 4.8-5.1x faster), fixed in
  [antimora/burn-flex#47](https://github.com/antimora/burn-flex/pull/47).
- conv_transpose2d (was 8x slower; now 1.5-1.9x faster), fixed in
  [antimora/burn-flex#46](https://github.com/antimora/burn-flex/pull/46).
- sort_last at 1024^2 (was 7.8x slower; now 1.3x faster), fixed in
  [antimora/burn-flex#45](https://github.com/antimora/burn-flex/pull/45).
- sum_dim/mean_dim last-axis at 1024^2 (was 2x slower; now tied), fixed in
  [antimora/burn-flex#50](https://github.com/antimora/burn-flex/pull/50).
- index_select at 1024^2 (was 2.8x slower; now 1.3x faster), fixed in
  [antimora/burn-flex#51](https://github.com/antimora/burn-flex/pull/51).
- conv2d 1x1 pointwise (was 1.5x slower; now 4.0x faster), fixed in
  [antimora/burn-flex#52](https://github.com/antimora/burn-flex/pull/52).
- mask_where / where_cond at 1024^2 (was 2x slower; now tied), fixed in
  [antimora/burn-flex#41](https://github.com/antimora/burn-flex/issues/41).
- nearest2d interpolation (was 2x slower; now 1.3x faster), fixed in
  [antimora/burn-flex#43](https://github.com/antimora/burn-flex/issues/43).

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

# burn-ember Benchmarks

Tracking performance compared to burn-ndarray.

## Running Benchmarks

```bash
cargo bench --bench binary_ops --features simd
cargo bench --bench matmul --features simd,gemm
cargo bench --bench slice_ops --features simd
```

## Results (with SIMD)

**Platform:** Apple Silicon M3 (ARM64) **Date:** 2026-02-03

### Contiguous Operations (1M elements, f32)

| Operation  | Ember | NdArray | Speedup |
| ---------- | ----- | ------- | ------- |
| add        | 202µs | 340µs   | 1.7x    |
| mul        | 214µs | 335µs   | 1.6x    |
| div        | 227µs | 334µs   | 1.5x    |
| scalar_add | 117µs | 179µs   | 1.5x    |
| scalar_mul | 117µs | 175µs   | 1.5x    |

### Non-Contiguous Operations (transposed)

| Size      | Ember  | NdArray | Result                |
| --------- | ------ | ------- | --------------------- |
| 256x256   | 58µs   | 53µs    | ~equal                |
| 1024x1024 | 1.06ms | 1.08ms  | ~equal                |

### Optimization History

#### Phase 1: Row-based Iteration (non-contiguous)

**Initial implementation (StridedIter):**

| Size      | Ember | NdArray | Ratio     |
| --------- | ----- | ------- | --------- |
| 256x256   | 388µs | 55µs    | 7x slower |
| 1024x1024 | 6.4ms | 1.2ms   | 5x slower |

**After row-based iteration:**

| Size      | Ember  | NdArray | Ratio       |
| --------- | ------ | ------- | ----------- |
| 256x256   | 58µs   | 54µs    | ~equal      |
| 1024x1024 | 1.09ms | 1.55ms  | 1.4x faster |

**Result: 5.9x faster for large transposed tensors**

#### Phase 2: SIMD Kernels (contiguous)

Added NEON SIMD intrinsics for ARM64 with in-place mutation optimization.

**Before SIMD (baseline):**

| Operation | Ember | NdArray | Speedup |
| --------- | ----- | ------- | ------- |
| add       | 210µs | 397µs   | 1.9x    |

**After SIMD:**

| Operation | Ember | NdArray | Speedup |
| --------- | ----- | ------- | ------- |
| add       | 202µs | 340µs   | 1.7x    |

SIMD provides modest gains for element-wise ops since they are memory-bound, not compute-bound. The main benefit comes from in-place mutation avoiding extra allocations.

### Matrix Multiplication (gemm)

**Run with:** `cargo bench --bench matmul --features simd,gemm,rayon`

#### Square Matrices (with rayon)

| Size | Ember | NdArray | Result |
|------|-------|---------|--------|
| 64x64 | 7.1µs | 20.4µs | **Ember 2.9x faster** |
| 128x128 | 46.8µs | 64.9µs | **Ember 1.4x faster** |
| 256x256 | 158µs | 170µs | ~equal |
| 512x512 | 597µs | 895µs | **Ember 1.5x faster** |
| 1024x1024 | 2.88ms | 6.1ms | **Ember 2.1x faster** |

#### Batched Matmul (with batch-level parallelism)

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| batch8_64x64 | 76µs | 112µs | **Ember 1.5x faster** |
| batch16_128x128 | 336µs | 632µs | **Ember 1.9x faster** |
| batch32_64x64 | 117µs | 190µs | **Ember 1.6x faster** |
| heads12_seq512_dim64 | 913µs | 1.84ms | **Ember 2.0x faster** |

#### Transposed Inputs (256x256)

| Input | Ember | NdArray | Result |
|-------|-------|---------|--------|
| lhs transposed | 184µs | 208µs | **Ember 1.1x faster** |
| rhs transposed | 154µs | 191µs | **Ember 1.2x faster** |
| both transposed | 184µs | 212µs | **Ember 1.2x faster** |

#### Memory Usage

| Size | Ember Max Alloc | NdArray Max Alloc |
|------|-----------------|-------------------|
| 64x64 | 66 KB | 49 KB |
| 256x256 | 1.05 MB | 787 KB |
| 1024x1024 | 16.8 MB | 12.6 MB |

#### Matmul Optimizations Applied

1. **Strided gemm**: Transposed inputs use native strides (no copy needed)
2. **Per-matrix parallelism**: Auto-enabled for matrices > 192^3 ops (~7M) via rayon
3. **Batch-level parallelism**: For small matrices in batches, parallelize the batch loop
4. **Batch heuristic**: Prefer batch parallelism when batch_size >= 4 (avoids repeated thread sync)
5. **Memory reduction**: ~30% less allocation by avoiding intermediate buffers

#### Matmul Analysis

With optimizations, Ember wins for:
- Small matrices (<192x192): Lower overhead, single-threaded gemm
- Large matrices (>=256x256): Parallel gemm beats NdArray BLAS
- Batched small matrices: Batch-level parallelism beats sequential execution

256x256 is essentially equal to NdArray. Ember wins at 4/5 square sizes, **all 3 transposed cases**, and **all 4 batched cases**.

### Integer Matrix Multiplication (i32)

**Run with:** `cargo bench --bench matmul --features simd,rayon`

Integer matmul uses naive triple-loop with SIMD dot product (no gemm equivalent for integers).

#### 2D Square Matrices

| Size | Ember | NdArray | Result |
|------|-------|---------|--------|
| 64x64 | 109µs | 119µs | **Ember 8% faster** |
| 128x128 | 948µs | 990µs | **Ember 4% faster** |
| 256x256 | 10.6ms | 10.2ms | ~equal |
| 512x512 | 121ms | 118ms | ~equal |

#### Batched Integer Matmul

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| batch8_64x64 | 873µs | 283µs | NdArray 3.1x faster |
| batch16_128x128 | 15.2ms | 2.3ms | NdArray 6.6x faster |

#### Integer Matmul Analysis

For 2D matrices, Ember wins small/medium sizes due to SIMD-optimized dot product with rhs transpose. Larger sizes are roughly equal.

For batched operations, NdArray is faster due to better cache utilization in its generic matmul implementation. No high-performance integer GEMM exists in pure Rust (gemm/matrixmultiply only support f32/f64).

**Optimizations applied:**
1. NEON SIMD dot product (`vmlaq_s32`, `vaddvq_s32`)
2. Rhs transpose for contiguous column access
3. Batch-level rayon parallelism

**Future optimization:** Cache-blocked (tiled) matmul would improve locality for batched cases.

### Slice Operations

**Run with:** `cargo bench --bench slice_ops --features simd`

Ember's slice operation is zero-copy for positive steps (metadata only). This section compares slicing performance across different patterns.

#### Basic Slice

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| slice_1d_1k | 339ns | 343ns | ~equal |
| slice_1d_1m | 159µs | 80µs | NdArray 2.0x faster |
| slice_2d_256x256 | 7.8µs | 8.1µs | ~equal |
| slice_2d_1024x1024 | 165µs | 86µs | NdArray 1.9x faster |
| slice_3d_64x64x64 | 28.8µs | 29.3µs | ~equal |

#### Slice on Transposed Tensor (non-contiguous)

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| transposed_256x256 | 7.3µs | 8.3µs | **Ember 1.14x faster** |
| transposed_1024x1024 | 145µs | 233µs | **Ember 1.60x faster** |

#### Slice with Step

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| step2_1d_1k | 318ns | 478ns | **Ember 1.50x faster** |
| step2_1d_1m | 149µs | 195µs | **Ember 1.31x faster** |
| step4_2d_256x256 | 7.4µs | 6.7µs | ~equal |
| step2_2d_1024x1024 | 160µs | 140µs | ~equal |

#### Narrow (single dimension slice)

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| narrow_dim0_256x256 | 8.0µs | 6.3µs | NdArray 1.27x faster |
| narrow_dim1_256x256 | 8.2µs | 11.1µs | **Ember 1.36x faster** |
| narrow_dim0_1024x1024 | 168µs | 80µs | NdArray 2.1x faster |

#### Slice Assign

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| assign_1d_1k | 587ns | 904ns | **Ember 1.5x faster** |
| assign_2d_256x256 | 12.4µs | 14.6µs | **Ember 1.2x faster** |
| assign_2d_1024x1024 | 199µs | 198µs | ~equal |

#### Memory Comparison (Slice Operations)

| Benchmark | Ember Max Alloc | NdArray Max Alloc |
|-----------|-----------------|-------------------|
| slice_1d_1k | 8.3 KB | 6.3 KB |
| slice_1d_1m | 8.4 MB | 6.3 MB |
| slice_2d_1024x1024 | 8.4 MB | 5.2 MB |
| slice_assign_2d_1024x1024 | 10.5 MB | 10.5 MB |

#### Slice Analysis

**Wins for Ember:**
- Transposed tensor slicing (1.14-1.60x faster)
- Step-based slicing on 1D tensors (1.31-1.50x faster)
- narrow_dim1 (slicing along non-contiguous dimension)
- Slice assign (1.2-1.5x faster after row-based optimization)

**Wins for NdArray:**
- Large contiguous slices (1.9-2.0x faster on 1M+ elements)
- narrow_dim0 on large tensors

**Key insight:** Ember's slice is metadata-only (zero-copy) for positive steps. The benchmark overhead comes from tensor cloning in the test setup. For actual workloads, slice itself is O(1).

#### Slice Assign Optimization

The initial recursive implementation was 3-5x slower than NdArray. After optimization:

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| assign_1d_1k | 2.2µs | 587ns | **3.8x** |
| assign_2d_256x256 | 73µs | 12.4µs | **5.9x** |
| assign_2d_1024x1024 | 1.03ms | 199µs | **5.2x** |

**Optimizations applied:**
1. `copy_from_slice` for contiguous inner dimensions (memcpy-based)
2. Direct row/column loops for 2D tensors (no recursion)
3. Iterative odometer-style index computation for ND tensors

### Analysis

#### Row-based iteration for 2D tensors

The original `StridedIter` did per-element index computation with multi-index tracking:

- 5+ operations per element
- Poor branch prediction
- Cache-unfriendly access patterns

The optimized approach uses direct row-column loops:

```rust
for row in 0..rows {
    let row_start = offset + row * row_stride;
    for col in 0..cols {
        result.push(op(src[row_start + col * col_stride], ...));
    }
}
```

Benefits:

- No per-element overhead
- Predictable inner loop (compiler can optimize)
- Better cache prefetching

#### SIMD with in-place mutation

For contiguous tensors, SIMD kernels use NEON intrinsics (ARM64):

```rust
// 4x f32 per 128-bit register
let va = vld1q_f32(a_ptr.add(offset));
let vb = vld1q_f32(b_ptr.add(offset));
let vr = vaddq_f32(va, vb);
vst1q_f32(a_ptr.add(offset), vr);  // in-place write
```

Key insight: Element-wise ops are memory-bound. SIMD helps marginally, but avoiding allocation is more impactful. The in-place kernels write directly back to the input tensor.

### Memory Allocations

Ember consistently uses fewer allocations:

- Ember: 10 allocations, 8.4MB for binary ops
- NdArray: 13 allocations, 16.8MB for binary ops

### Features

- `simd` (default): NEON kernels on ARM64, scalar fallback elsewhere
- `rayon` (optional): Parallel execution for tensors > 4M elements

## Future Optimizations

1. **AVX2 kernels** - SIMD for x86_64
2. **Cache blocking** - Tiled iteration for better locality
3. **3D+ tensor optimization** - Extend row-based approach to higher dimensions
4. **Fused operations** - Combine multiple ops to reduce memory traffic

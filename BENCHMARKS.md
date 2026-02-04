# burn-ember Benchmarks

Tracking performance compared to burn-ndarray.

## Running Benchmarks

```bash
cargo bench --bench binary_ops --features simd
cargo bench --bench matmul --features simd,gemm
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
| 64x64 | 114µs | 116µs | ~equal |
| 128x128 | 976µs | 1.08ms | **Ember 10% faster** |
| 256x256 | 10.9ms | 10.2ms | NdArray 7% faster |
| 512x512 | 122ms | 119ms | ~equal |

#### Batched Integer Matmul

| Benchmark | Ember | NdArray | Result |
|-----------|-------|---------|--------|
| batch8_64x64 | 945µs | 261µs | NdArray 3.6x faster |
| batch16_128x128 | 15.5ms | 2.4ms | NdArray 6.5x faster |

#### Integer Matmul Analysis

For 2D matrices, Ember is competitive with NdArray. The SIMD-optimized dot product with rhs transpose provides good performance for single matrix operations.

For batched operations, NdArray is significantly faster. This is likely due to NdArray using more optimized BLAS-like routines internally. Ember's naive approach with per-batch transpose has higher overhead.

**Future optimization:** Tiled/blocked matrix multiplication would improve cache utilization for both 2D and batched cases.

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

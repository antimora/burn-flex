# burn-ember Benchmarks

Tracking performance compared to burn-ndarray.

## Running Benchmarks

```bash
cargo bench --bench binary_ops
```

## Results

**Platform:** Apple Silicon M3 (ARM64)
**Date:** 2026-02-03
**Commit:** Initial binary ops implementation

### Contiguous Operations (1M elements, f32)

| Operation | Ember | NdArray | Speedup |
|-----------|-------|---------|---------|
| add | 219µs | 364µs | 1.7x |
| mul | 224µs | 343µs | 1.5x |
| div | 225µs | 365µs | 1.6x |
| scalar_add | 131µs | 197µs | 1.5x |
| scalar_mul | 124µs | 201µs | 1.6x |

### Non-Contiguous Operations (transposed)

| Size | Ember | NdArray | Ratio |
|------|-------|---------|-------|
| 256x256 | 388µs | 55µs | 7x slower |
| 1024x1024 | 6.4ms | 1.2ms | 5x slower |

### Analysis

**Strengths:**
- Contiguous tensor operations are 1.5-1.7x faster than NdArray
- Simple loop-based implementation with good cache locality
- Lower allocation overhead (fewer intermediate allocations)

**Areas for Improvement:**
- Non-contiguous (strided) iteration is significantly slower
- `StridedIter` needs optimization or replacement with block-based iteration
- Consider adopting ndarray's strided access patterns for non-contiguous cases

### Memory Allocations

Ember uses fewer allocations for contiguous operations:
- Ember: 10 allocations, 8.4MB for binary ops
- NdArray: 13 allocations, 16.8MB for binary ops

## Future Optimizations

1. **SIMD kernels** - NEON for ARM64, AVX2 for x86_64
2. **Parallel execution** - rayon for large tensors
3. **Better strided iteration** - Block-based or tiled access patterns
4. **In-place optimization** - Arc-based storage with `is_unique()` check

# burn-ember Architecture

A pure-Rust CPU backend for [Burn](https://github.com/tracel-ai/burn).

## Goals

From README:

- Fast, memory-efficient CPU backend
- Multi-threading, SIMD, faer-rs acceleration
- Runs on std, no_std, and WebAssembly
- Supports f16/bf16
- Zero-copy data loading
- Thread-safe by design

## Target Platform

**Primary: Apple Silicon M3 (ARM64 + NEON)**

- 128-bit SIMD registers (4x f32, 8x f16)
- Unified memory architecture
- Native f16 support in hardware

**Secondary: x86_64 with AVX2/AVX-512** (via conditional compilation)

---

## Design Principles

1. **Leverage Burn** - Use `burn-backend` types and `burn-std` utilities wherever possible
2. **Portability first** - No platform-specific dependencies; std, no_std, WASM
3. **Zero C dependencies** - Pure Rust only (faer-rs for linalg)
4. **Simple and direct** - Eager execution, no lazy graphs, no fusion (use `burn-fusion` if needed)
5. **Memory reuse** - Minimize allocations through in-place ops and buffer reuse

---

## Memory Strategy

Minimize allocations wherever possible:

### In-Place Operations

When the input tensor is uniquely owned (`Bytes::is_unique()`), mutate in place:

```rust
fn neg_inplace(mut tensor: EmberTensor) -> EmberTensor {
    if tensor.data.is_unique() && tensor.is_contiguous() {
        let slice: &mut [f32] = tensor.data.as_mut_slice();
        for x in slice.iter_mut() {
            *x = -*x;
        }
        tensor
    } else {
        // COW: allocate new buffer
        neg_copy(&tensor)
    }
}
```

### Output Buffer Reuse

For binary ops, reuse input buffer when possible:

```rust
fn add(lhs: EmberTensor, rhs: &EmberTensor) -> EmberTensor {
    if lhs.data.is_unique() && lhs.is_contiguous() && lhs.shape() == rhs.shape() {
        // Reuse lhs buffer
        add_into(lhs, rhs)
    } else {
        add_alloc(&lhs, rhs)
    }
}
```

### When to Allocate

Only allocate when necessary:

- Shape changes (broadcast, concat, reshape of non-contiguous)
- Multiple references to input (`!is_unique()`)
- Non-contiguous input that must become contiguous

---

## Burn Infrastructure We Use

From `burn-backend`:

- `Shape` - tensor dimensions
- `TensorData` - serialized tensor format
- `DType` - runtime dtype enum
- `Element` trait - compile-time element types
- `Backend` trait - the interface we implement
- `*TensorOps` traits - operation interfaces

From `burn-std`:

- `Bytes` - aligned byte storage with COW semantics (our tensor backing store)
- `is_contiguous()` - stride validation
- Platform abstractions for no_std

---

## Core Types

### Layout

Metadata for interpreting storage as an N-dimensional tensor:

```rust
use burn_backend::Shape;

pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    start_offset: usize,
}
```

Many operations are zero-copy (metadata changes only):

- `transpose()` - swap strides
- `narrow()` - adjust offset
- `reshape()` - recompute strides if contiguous
- `broadcast()` - set stride to 0

### Tensor

Uses `Bytes` from burn-std directly (aligned, zero-copy capable):

```rust
use burn_std::Bytes;
use burn_backend::DType;

pub struct EmberTensor {
    data: Bytes,
    layout: Layout,
    dtype: DType,
}

impl EmberTensor {
    /// Cast to typed slice for operations (zero-cost, just pointer reinterpret)
    pub fn as_slice<T: Element>(&self) -> &[T] {
        self.data.as_slice()
    }

    pub fn as_slice_mut<T: Element>(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}
```

Operations dispatch on `dtype` and cast once at the boundary:

```rust
fn add(a: &EmberTensor, b: &EmberTensor) -> EmberTensor {
    match a.dtype {
        DType::F32 => add_impl(a.as_slice::<f32>(), b.as_slice::<f32>()),
        DType::F16 => add_impl(a.as_slice::<f16>(), b.as_slice::<f16>()),
        // ...
    }
}
```

---

## Backend Implementation

```rust
use burn_backend::{Backend, DType};

#[derive(Clone, Copy, Debug, Default)]
pub struct Ember;

impl Backend for Ember {
    type Device = EmberDevice;
    type FloatTensorPrimitive = EmberTensor;
    type IntTensorPrimitive = EmberTensor;
    type BoolTensorPrimitive = EmberTensor;
    type QuantizedTensorPrimitive = EmberQTensor;

    fn name() -> String { "ember".into() }

    fn float_supported_dtypes() -> Vec<DType> {
        vec![DType::F64, DType::F32, DType::F16, DType::BF16]
    }

    fn int_supported_dtypes() -> Vec<DType> {
        vec![DType::I64, DType::I32, DType::I16, DType::I8,
             DType::U64, DType::U32, DType::U16, DType::U8]
    }
}
```

---

## Execution Strategy

### Contiguous Fast Path

Most tensors are contiguous. Detect and use direct slice operations:

```rust
fn unary_op<T, F>(storage: &[T], layout: &Layout, f: F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    if let Some((start, end)) = layout.contiguous_offsets() {
        storage[start..end].iter().map(|&x| f(x)).collect()
    } else {
        StridedIter::new(layout).map(|i| f(storage[i])).collect()
    }
}
```

### SIMD Kernels

NEON for ARM64, with scalar fallback:

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    const LANES: usize = 4;
    let chunks = a.len() / LANES;

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(off));
        let vb = vld1q_f32(b.as_ptr().add(off));
        vst1q_f32(out.as_mut_ptr().add(off), vaddq_f32(va, vb));
    }

    for i in (chunks * LANES)..a.len() {
        out[i] = a[i] + b[i];
    }
}
```

### Parallel Execution

Via rayon for large tensors:

```rust
use rayon::prelude::*;

fn parallel_unary<T, F>(src: &[T], f: F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    src.par_iter().map(|&x| f(x)).collect()
}
```

### Linear Algebra

faer-rs for matrix operations (pure Rust, WASM-compatible):

```rust
use faer::{MatRef, MatMut};

pub fn matmul<T: faer::RealField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    out: MatMut<'_, T>,
) {
    faer::linalg::matmul::matmul(
        out, a, b, None, T::one(),
        faer::Parallelism::Rayon(0)
    );
}
```

---

## Zero-Copy Loading

`Bytes` from burn-std supports zero-copy scenarios (mmap, external buffers). `EmberTensor` inherits
this directly.

## Thread Safety

`Bytes` handles ownership semantics (clone-on-write). No additional machinery needed.

---

## Implementation Phases

### Phase 1: Minimum Viable Backend

- `Layout`, `EmberTensor` (using `Bytes`)
- `Backend` trait impl
- Basic `FloatTensorOps`: from_data, into_data, shape, dtype

### Phase 2: Core Operations

- Arithmetic: add, sub, mul, div, neg
- Comparisons: equal, greater, less
- Shape: reshape, transpose, slice, concat
- Reductions: sum, mean, max, min
- Matmul via faer-rs

### Phase 3: Module Operations

- Conv2d, conv_transpose
- Pooling (max, avg)
- Full `ModuleOps` trait

### Phase 4: Optimization

- NEON SIMD kernels
- Parallel execution via rayon
- Contiguous detection and fast paths

### Phase 5: Advanced

- Quantization (Q8, Q4)
- f16/bf16 compute paths
- no_std support
- WASM validation

---

## File Structure

```
src/
├── lib.rs
├── backend.rs          # Backend trait impl
├── tensor.rs           # EmberTensor (Bytes + Layout + DType)
├── layout.rs           # Layout (shape + strides)
├── ops/
│   ├── mod.rs
│   ├── unary.rs
│   ├── binary.rs
│   ├── reduce.rs
│   ├── shape.rs
│   ├── matmul.rs
│   ├── conv.rs
│   └── pool.rs
└── simd/
    ├── mod.rs
    └── neon.rs
```

---

## Dependencies

| Crate        | Purpose                    |
| ------------ | -------------------------- |
| burn-backend | Core types, Backend trait  |
| burn-std     | Bytes, utilities           |
| faer         | Linear algebra (pure Rust) |
| half         | f16/bf16 types             |
| rayon        | Parallelism (optional)     |

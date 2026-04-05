# Proposal: add `softmax` and `layer_norm` hooks to backend traits

## TL;DR

`burn_tensor::activation::softmax` and `burn::nn::LayerNorm::forward` both
decompose into 5-6 primitive tensor ops with intermediate allocations. Every
burn CPU backend inherits this decomposition because there is no trait hook
for backends to override. On wav2vec2-large transformer inference, this makes
burn up to **29x slower per softmax call** and **7x slower per layer_norm
call** than a single-pass fused implementation, even against candle, which
does expose fused versions through its own ops layer.

This proposes adding `softmax` / `log_softmax` / `softmin` methods to the
existing `ActivationOps` trait and a layer_norm hook somewhere in backend
ops, each with a default implementation that matches today's decomposed
behavior (zero breaking change). Backends that want to fuse can override;
backends that don't just get the current behavior via the default.

Prototype implementations in `burn-flex` (a CPU backend in development)
show the expected wins:

| Op | Shape | Decomposed (status quo) | Fused (this proposal) | Candle (reference) |
|---|---|---|---|---|
| softmax | [16, 50, 50], last axis | 612 µs | **49 µs** (12.5x faster) | 92 µs |
| softmax | [16, 150, 150], last axis | 5.72 ms | **131 µs** (43.7x faster) | 194 µs |
| layer_norm | [50, 1024] | 136 µs | **18.8 µs** (7.2x faster) | 71 µs |
| layer_norm | [150, 1024] | 387 µs | **59 µs** (6.5x faster) | 94 µs |

Apple M3 Max, pure Rust, no BLAS on either side. Full per-op bench setup at
[crates/burn-flex-bench-candle](./) in the burn-flex repo.

For wav2vec2-large inference specifically (24 transformer layers, softmax +
2× layer_norm per layer), the per-forward-pass impact is ~38 ms at 1s audio
and ~190 ms at 3s audio, enough to flip burn from ~2x slower to *faster*
than candle on the same workload.

## The problem

### softmax

`burn_tensor::activation::softmax` is a backend-agnostic free function:

```rust
// burn-tensor/src/tensor/activation/base.rs
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));
    let tensor = tensor.clone() - tensor.detach().max_dim(dim);  // max_dim + sub
    let tensor = tensor.exp();                                    // exp
    let tensor_tmp = tensor.clone().sum_dim(dim);                 // sum_dim
    tensor.div(tensor_tmp)                                        // div
}
```

This calls five trait methods on the backend (`max_dim`, `sub`, `exp`,
`sum_dim`, `div`), each a separate full-tensor pass with intermediate
allocations. For an attention-scores tensor `[16, 150, 150]` that's ~7 MB of
memory traffic plus 4 malloc/free pairs per call.

The backend trait has no `softmax` method, so there is no way for a backend
to replace this with a fused kernel. `burn-ndarray` inherits this code path.
So does `burn-flex`. Any future CPU backend will too.

Crucially, `ActivationOps<B>` already exposes trait hooks for
`leaky_relu`/`relu`/`gelu`/`sigmoid`/`hard_sigmoid`/`log_sigmoid`, each with
a decomposed default impl. Backends that care about performance override
them. `softmax` just happens to be missing from that list: an asymmetric
gap in an otherwise consistent design.

### layer_norm

`burn::nn::LayerNorm::forward` is in the `burn-nn` module crate:

```rust
// burn-nn/src/modules/norm/layer.rs
pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
    let (var, mean) = input.clone().var_mean_bias(D - 1);
    let input_normalized = input.sub(mean).div(var.add_scalar(self.epsilon).sqrt());
    let output = input_normalized.mul(self.gamma.val().unsqueeze());
    match &self.beta {
        Some(beta) => output.add(beta.val().unsqueeze()),
        None => output,
    }
}
```

Same story: six primitive tensor ops, each a full pass. No trait hook
anywhere for a backend to intercept. `rms_norm` and `group_norm` have the
same shape.

## Evidence

See `crates/burn-flex-bench-candle/` in the burn-flex repo for a runnable
bench crate that compares burn-flex against candle-core 0.9 on CPU (pure
Rust, no BLAS/Accelerate/MKL). The relevant benches are:

```sh
cargo bench -p burn-flex-bench-candle --bench transformer_ops
```

which exercises softmax, layer_norm, and gelu at wav2vec2-large shapes.

Key observations from that bench:

1. **gelu is tied** between burn-flex and candle. This is because
   `ActivationOps::gelu` exists as a trait method, burn-flex overrides it
   with a fused single-pass impl, and the optimized path runs.

2. **softmax is 6-29x slower in burn-flex than in candle**. This is because
   no such hook exists; burn-flex is stuck with the decomposed form.

3. **layer_norm is 2-4x slower** for the same reason.

The delta between (1) and (2)/(3) is almost entirely the
presence/absence of a trait hook. It is *not* a difference in kernel
quality. burn-flex's primitive ops (`max_dim`, `exp`, `sum_dim`, `mean`,
`var`, `sqrt`, broadcast arithmetic) are already SIMD-optimized via
`macerator`. The overhead is architectural: repeated full-tensor sweeps
and intermediate heap allocations that cannot be elided by the caller.

## Proposal

### 1. Add softmax-family methods to `ActivationOps`

```rust
// burn-backend/src/backend/ops/activation.rs
pub trait ActivationOps<B: Backend> {
    // ... existing methods (leaky_relu, relu, gelu, sigmoid, ...) ...

    /// Applies the softmax activation function along `dim`.
    ///
    /// Default implementation matches the previous behavior of
    /// `burn_tensor::activation::softmax`: a 5-op decomposition that
    /// any backend can override with a fused single-pass kernel.
    fn softmax(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        let max = B::float_max_dim(tensor.clone(), dim);
        let shifted = B::float_sub(tensor, max);
        let exp = B::float_exp(shifted);
        let sum = B::float_sum_dim(exp.clone(), dim);
        B::float_div(exp, sum)
    }

    /// Applies the log-softmax activation function along `dim`.
    fn log_softmax(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        // log(softmax(x)) via the log-sum-exp trick
        let max = B::float_max_dim(tensor.clone(), dim);
        let shifted = B::float_sub(tensor, max.clone());
        let exp = B::float_exp(shifted.clone());
        let sum = B::float_sum_dim(exp, dim);
        let log_sum = B::float_log(sum);
        B::float_sub(shifted, log_sum)
    }

    /// Applies the softmin activation function along `dim`.
    /// Equivalent to `softmax(-x, dim)`.
    fn softmin(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        Self::softmax(B::float_neg(tensor), dim)
    }
}
```

`burn_tensor::activation::softmax` (and `log_softmax`, `softmin`) becomes a
one-liner forwarder:

```rust
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));
    Tensor::from_primitive(TensorPrimitive::Float(
        B::softmax(tensor.into_primitive().tensor(), dim)
    ))
}
```

This mirrors exactly how `activation::relu`, `activation::gelu`,
`activation::sigmoid`, etc. already forward through the backend trait.

### 2. Add a layer_norm hook

Two options here, and the burn maintainers are better placed to pick:

**Option A:** extend `ActivationOps` (it already contains `log_sigmoid`
which does the log-sum-exp trick with multi-op expansions, and layer_norm is
similar):

```rust
pub trait ActivationOps<B: Backend> {
    // ...

    /// Applies layer normalization along the last axis.
    ///
    /// `y = ((x - mean) / sqrt(var + epsilon)) * gamma + beta`, where
    /// `mean` and `var` are computed per row along the last axis.
    fn layer_norm(
        tensor: FloatTensor<B>,
        gamma: FloatTensor<B>,
        beta: Option<FloatTensor<B>>,
        epsilon: f64,
    ) -> FloatTensor<B> {
        // Default impl matches burn::nn::LayerNorm::forward:
        let rank = tensor.shape().num_dims();
        let axis = rank - 1;
        let (var, mean) = /* var_mean_bias equivalent */;
        let shifted = B::float_sub(tensor, mean);
        let inv_std = B::float_recip(B::float_sqrt(
            B::float_add_scalar(var, epsilon.into()),
        ));
        let normalized = B::float_mul(shifted, inv_std);
        let scaled = B::float_mul(normalized, gamma /* unsqueezed */);
        match beta {
            Some(b) => B::float_add(scaled, b /* unsqueezed */),
            None => scaled,
        }
    }
}
```

**Option B:** a new `NormOps<B: Backend>` supertrait for normalization
primitives (layer_norm, rms_norm, group_norm, batch_norm) with default
impls. Cleaner organization if more norm ops are expected, but requires a
new trait plus the `Backend: NormOps<Self>` bound.

Either way, `burn::nn::LayerNorm::forward` becomes a forwarder through the
backend trait, preserving current semantics.

### 3. Zero breaking change

Both proposals are additive:

- New trait methods, all with default implementations that are
  byte-equivalent to the current decomposed behavior.
- Existing backends compile unchanged; they inherit the defaults.
- Backends that want to opt into fused kernels add a single trait method
  implementation. burn-flex's implementations are ready to port in (see
  reference impls below).
- User-facing API (`burn_tensor::activation::softmax`, `burn::nn::LayerNorm`)
  is unchanged.

## Reference implementation

burn-flex has working fused implementations of both softmax and layer_norm
that achieve the numbers in the TL;DR table. Both use
`#[macerator::with_simd]` at chunk-of-rows granularity so SIMD dispatch is
amortized, `rayon` for row-level parallelism, and a two-pass row kernel
(mean/sum+sumsq pass, then normalize+affine pass for layer_norm; max pass,
then exp+sum pass, then normalize pass for softmax).

Source:

- `crates/burn-flex/src/ops/activation.rs`: fused `softmax` and
  `layer_norm` functions, ~500 lines total including comments and dtype
  branching.
- `crates/burn-flex/src/ops/activation.rs` test module: cross-check tests
  against manually computed references and against
  `burn_tensor::activation::softmax`.

If the hooks land in burn-backend, burn-flex will move these function
bodies into `impl ActivationOps<Flex> for Flex { fn softmax(...) { ... } }`
as a one-line change.

## Scope and non-goals

- **In scope:** `softmax`, `log_softmax`, `softmin`, `layer_norm`. These
  are the highest-impact ops for transformer workloads on CPU.
- **Adjacent but deferred:** `rms_norm` (LLaMA-family), `group_norm`
  (vision). Same pattern, separate PR if interest.
- **Out of scope:** fused attention (already has `ModuleOps::scaled_dot_product_attention`).
- **Not proposed:** changing how `max_dim`/`sum_dim`/etc. work. The
  primitive ops stay as-is; we're adding higher-level hooks alongside them.

## Follow-up once this lands

With the hooks in place, backends can progressively adopt fused versions:

- **burn-ndarray:** could add a simple single-pass scalar fused softmax
  using `ndarray::Zip::from(row).for_each(...)`. Probably 2-5x speedup on
  the same shapes.
- **burn-flex (CPU):** ports the existing implementations. Numbers above.
- **burn-cuda, burn-wgpu:** fused softmax is a single kernel launch
  instead of four. Less memory traffic, fewer kernel launches, meaningful
  latency wins at small batch sizes.
- **burn-candle (if it exists):** trivially delegates to candle-nn's
  fused versions.

## Questions for maintainers

1. Is `ActivationOps` the right home for `layer_norm`, or should it live
   in a new `NormOps` trait?
2. Should the softmax hook take `dim: usize` or a more structured
   descriptor (e.g. supporting negative indexing)?
3. For `layer_norm`, what's the preferred shape of the weight tensor: a
   1-D `[d_model]` aligned with the last axis (as burn-nn's `LayerNorm`
   currently does), or something more flexible?
4. Is there a preferred naming: `float_softmax` / `float_layer_norm`
   (matching the `float_*` prefix on `FloatTensorOps` methods), or the
   unprefixed `softmax` / `layer_norm` (matching the existing
   `ActivationOps` methods like `gelu`, `sigmoid`, `relu`)?

Happy to open a draft PR against any/all of these once the shape is
confirmed.

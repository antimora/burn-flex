# Running burn-backend-tests for Ember

## Setup

We use a burn worktree at `/Users/dilshod/Projects/burn-ember-worktree` with ember configured.

The worktree has these modifications in `crates/burn-backend-tests/`:

**Cargo.toml:**

```toml
[dependencies]
burn-ember = { path = "/Users/dilshod/Projects/burn-ember/crates/burn-ember", optional = true, default-features = false }

[features]
ember = ["burn-ember"]
```

**tests/common/backend.rs:**

```rust
#[cfg(feature = "ember")]
pub type TestBackend = burn_ember::Ember;
```

## Running Tests

```bash
cd /Users/dilshod/Projects/burn-ember-worktree/crates/burn-backend-tests

# Run all tensor tests
cargo test --no-default-features --features ember --test tensor

# Run all autodiff tests
cargo test --no-default-features --features ember --test autodiff

# Run specific test modules
cargo test --no-default-features --features ember --test tensor trig
cargo test --no-default-features --features ember --test tensor matmul
cargo test --no-default-features --features ember --test tensor slice
```

## Current Status

**Tensor tests:** 355 passed, 716 failed (many ops not yet implemented)

**Implemented ops that pass:**

- Arithmetic: add, sub, mul, div (tensor-tensor and tensor-scalar)
- Math: exp, log, log1p, sqrt, abs, recip, erf
- Trig: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh (12/13, atan2 todo)
- Reduction: sum, sum_dim, mean_dim, argmax, argmin
- Shape: reshape, transpose, slice, slice_assign
- Linear algebra: matmul (2D and batched)

**Not yet implemented:**

- float_random, float_into_int, float_cast
- float_remainder, float_powf, float_atan2
- float_permute, float_flip, float_expand
- float_gather, float_scatter_add, float_select
- float_mask_where, float_mask_fill
- Comparison ops (equal, greater, lower, etc.)
- Cumulative ops (cumsum, cumprod, cummin, cummax)
- Int tensor ops
- Bool tensor ops

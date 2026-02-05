# TODO

## Code Quality
- [ ] Make tests consistent; some places use different approximate functions
- [ ] Remove duplications via macros
- [ ] Break down large functions into smaller ones (conv.rs)
- [ ] Move code from simd/mod.rs
- [ ] Update ARCHITECTURE.md with Arc ref counting
- [ ] Test without feature flags
- [ ] Licenses
- [ ] Give credit to Ndarray and Candle
- [ ] Proper testing of strides and storage
- [ ] Zero rank tensors
- [ ] Non contiguous tensors operations

## Bugs
- [ ] Chained tensor op bugs
- [ ] Reference counting for highly parallelized operations
- [ ] int_matmul batch dimension mismatch

## Unimplemented Operations

All tensor operations are now implemented.

### Float Ops
- [x] `float_into_int`
- [x] `float_permute`, `float_flip`
- [x] `float_mask_where`, `float_mask_fill`
- [x] `float_remainder`, `float_remainder_scalar`
- [x] `float_cast`
- [x] `float_gather`, `float_scatter_add`, `float_select`, `float_select_add`
- [x] `float_cumsum`, `float_cumprod`, `float_cummin`, `float_cummax`
- [x] `float_powf`, `float_atan2`, `float_cross`, `float_unfold`
- [x] `float_random`

### Int Ops
- [x] `int_add`, `int_sub`, `int_mul`, `int_div` (and scalar variants)
- [x] `int_into_float`
- [x] `int_permute`, `int_flip`
- [x] `int_mask_where`, `int_mask_fill`
- [x] `int_remainder`
- [x] `int_gather`, `int_scatter_add`, `int_select`, `int_select_add`

### Bool Ops
- [x] `bool_into_int`, `bool_into_float`
- [x] `bool_permute`, `bool_flip`
- [x] `bool_mask_where`, `bool_mask_fill`
- [x] `bool_ones`
- [x] `bool_gather`, `bool_scatter_or`, `bool_unfold`




Categorized failures:
Category: Bool ops
Count: 21
Issues: cat, stack, repeat, logical (and/or/xor/not), tri_mask, gather/scatter, unfold
────────────────────────────────────────
Category: Float module
Count: 22
Issues: deform_conv2d (not impl), linear (not impl), avgpool ceil_mode, adaptive_pool bigger_output, conv groups
────────────────────────────────────────
Category: Float ops
Count: 35
Issues: fmod, remainder, all/any, prod, grid_sample, slice negative steps, matmul broadcast 4D
────────────────────────────────────────
Category: Int ops
Count: 31
Issues: abs, bitwise, cast, not_equal, remainder, random, slice with steps, unfold
────────────────────────────────────────
Category: Other
Count: 4
Issues: matvec broadcast, vector_norm, clone_invariance
Not implemented (todo!):
- deform_conv2d
- linear (module)
- grid_sample
- tri_mask
- Bool logical ops (and, or, xor, not)
- fmod, remainder
- all, any, prod
- Int abs, bitwise ops
- Slice with negative steps

Bugs to fix:
- conv groups > 1
- avgpool ceil_mode
- adaptive_pool bigger_output
- matmul 4D broadcast


Progress: 999 passed, 72 failed

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

## Bugs
- [ ] Chained tensor op bugs
- [ ] Reference counting for highly parallelized operations
- [ ] int_matmul batch dimension mismatch

## Unimplemented Operations

**Test status: 755 passed, 323 failed** (tensor tests only)

### Critical (blocks many tests)
- [x] `float_random` - implemented with no_std support (+63 tests now passing)

### Float Ops (11 missing)
- [x] `float_into_int`
- [x] `float_permute`, `float_flip`
- [x] `float_mask_where`, `float_mask_fill`
- [x] `float_remainder`, `float_remainder_scalar`
- [ ] `float_cast`
- [ ] `float_gather`, `float_scatter_add`, `float_select`, `float_select_add`
- [ ] `float_cumsum`, `float_cumprod`, `float_cummin`, `float_cummax`
- [ ] `float_powf`, `float_atan2`, `float_cross`, `float_unfold`

### Int Ops (2 missing)
- [x] `int_add`, `int_sub`, `int_mul`, `int_div` (and scalar variants)
- [x] `int_into_float`
- [x] `int_permute`, `int_flip`
- [x] `int_mask_where`, `int_mask_fill`
- [x] `int_remainder`
- [ ] `int_gather`, `int_scatter_add`, `int_select`, `int_select_add`

### Bool Ops (3 missing)
- [x] `bool_into_int`, `bool_into_float`
- [x] `bool_permute`, `bool_flip`
- [x] `bool_mask_where`, `bool_mask_fill`
- [x] `bool_ones`
- [ ] `bool_gather`, `bool_scatter_or`, `bool_unfold`

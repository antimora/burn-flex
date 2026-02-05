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

## Unimplemented Operations

**Test status: 614 passed, 464 failed** (tensor tests only)

### Critical (blocks many tests)
- [x] `float_random` - implemented with no_std support (+63 tests now passing)

### Float Ops (16 missing)
- [x] `float_into_int`
- [ ] `float_cast`
- [ ] `float_remainder`, `float_remainder_scalar`
- [ ] `float_permute`, `float_flip`
- [ ] `float_gather`, `float_scatter_add`, `float_select`, `float_select_add`
- [ ] `float_mask_where`, `float_mask_fill`
- [ ] `float_cumsum`, `float_cumprod`, `float_cummin`, `float_cummax`
- [ ] `float_powf`, `float_atan2`, `float_cross`, `float_unfold`

### Int Ops (7 missing)
- [x] `int_add`, `int_sub`, `int_mul`, `int_div` (and scalar variants)
- [x] `int_into_float`
- [ ] `int_remainder`, `int_permute`
- [ ] `int_mask_where`, `int_mask_fill`
- [ ] `int_gather`, `int_scatter_add`, `int_select`, `int_select_add`

### Bool Ops (8 missing)
- [x] `bool_into_int`, `bool_into_float`
- [ ] `bool_permute`, `bool_flip`, `bool_ones`
- [ ] `bool_mask_where`, `bool_mask_fill`
- [ ] `bool_gather`, `bool_scatter_or`, `bool_unfold`

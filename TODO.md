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
- [x] Proper testing of strides and storage
- [ ] Zero rank tensors
- [x] Non contiguous tensors operations
- [x] Scan for default backend implementations and make sure we are using Embers

## Bugs
- [ ] Chained tensor op bugs
- [ ] Reference counting for highly parallelized operations
- [x] int_matmul batch dimension mismatch


All 1493 burn tests + 293 local tests pass with zero failures.

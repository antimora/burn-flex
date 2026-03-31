# burn-flex Project Instructions

Consult `ARCHITECTURE.md` for design decisions, memory strategy, and implementation patterns.

## Git Workflow

1. Complete work on current branch
2. Submit PR and wait for merge
3. Only after merge, create a new branch for the next phase

Do not start new feature work without confirming the PR/merge status first.

## Burn's Bytes Type

`Bytes` is from burn-std (re-exported from cubecl-common). Key methods:

- `Bytes::from_elems<E>(Vec<E>)` - create from typed vector
- `Bytes::from_bytes_vec(Vec<u8>)` - create from raw bytes
- `Deref<Target=[u8]>` - access as byte slice
- `DerefMut` - mutable byte access
- `try_into_vec::<E>()` - zero-copy conversion back to typed vector

## Coding Patterns

- Encapsulate bytemuck casting inside `FlexTensor` methods (`storage()`, `storage_mut()`)
- Use `tensor.storage::<f32>()` not `bytemuck::cast_slice(tensor.data())`
- Check `layout().contiguous_offsets()` for fast-path decisions
- For in-place ops, check `Some((0, end))` pattern (contiguous at offset 0)
- Index-producing ops must respect `out_dtype`/`indices_dtype` parameters. Use `isize` +
  `INDEX_DTYPE` internally, then `int_cast` to requested dtype. Never hardcode `i64`.

## Testing

Write generous tests. Cover:
- Contiguous and non-contiguous tensors (transposed, sliced)
- Multiple dtypes (f32, f64, f16 when applicable)
- Edge cases: empty tensors, single element, large tensors
- Both tensor-tensor and tensor-scalar variants
- Verify correctness through round-trip with `into_data()`

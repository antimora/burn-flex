Here are all the unfinished items / known limitations in the burn-flex crate as it stands in the PR:

  Unfinished items in burn-flex

  1. Bool(U32) storage not implemented

  Files: backend.rs:144, tensor.rs:302, comparison.rs:393

  burn-flex stores bools as 1 byte per element (Native and U8 only). Bool(U32) would require 4-byte-per-element storage plumbed
  through ~30 sites: dtype_size, to_contiguous, storage::<E>() cast, copy_contiguous, every bool ops reader in bool.rs, slice.rs
  dispatch, cat.rs dispatch, mask.rs, gather_scatter.rs, and the SIMD bool ops in portable.rs. Currently declared unsupported via
  dtype_usage and guarded with panics.

  No tracking issue yet. I should file one.

  2. Phantom-generic Flex not truly generic over element types

  File: backend.rs:70-76 (links to https://github.com/tracel-ai/burn/issues/4762)

  Flex<E = f32, I = i32> has phantom generics to slot into burn-dispatch's $Backend<f32> macro pattern, but the Backend impl is
  only provided for the default Flex<f32, i32>. Making it truly generic requires rewriting ~9 ops trait impl blocks plus ~50
  Flex::method() to Self::method() substitutions. Tracked in #4762.

  3. grid_sample bicubic mode not supported

  File: ops/grid_sample.rs:27

  grid_sample_2d panics on bicubic interpolation mode. Supports bilinear and nearest. This is pre-existing from the standalone
  repo, not introduced in the migration.
  
  

crates/burn-core's TestBackend (still ndarray; unrelated optimization)
crates/burn-no-std-tests (still ndarray; flex's own no_std path is validated by its own test suite)
burn-book and contributor-book docs (doc update as follow-up once the backend lands)



/Users/dilshod/Projects/burn-flex/.github/workflows/test.yml

update the documentation

- Note that flex is NOT in default, but ndarray IS — this mirrors burn-ndarray's treatment (default
  backend stays ndarray) while keeping flex opt-in. We can flip this when burn-flex replaces
  ndarray, but for this PR flex is additive.

autodiff

cargo test-flex 0.98s user 0.19s system 86% cpu 1.349 total [burn-backend-tests]% time cargo
test-flex

cargo test-ndarray 1.30s user 0.23s system 107% cpu 1.414 total [burn-backend-tests]% time cargo
test-ndarray

Worth flagging to upstream reviewers:

1. The phantom-generic change to Flex is the most unusual piece. I documented it in a doc-comment
   explaining why it exists (dispatch compat) and what its limitation is (impl Backend only for the
   default instantiation). If reviewers want "true" generic elem support in burn-flex, that's a
   follow-up requiring ~9 trait impl block rewrites and ~50 Flex:: → Self:: substitutions inside
   impl bodies.
2. The float_predicate fix in burn-flex is a genuine latent bug — worth a mention in the PR
   description so it doesn't look like an unrelated change.
3. The BackendId::Flex = 8 discriminant I chose appends to the enum rather than slotting in
   alphabetically. This keeps the diff minimal but means BackendId ordering no longer follows
   backend_list order. Harmless (pub(crate) enum, runtime-only use) but reviewers might ask.
4. xtask adds flex to GithubRunner alongside ndarray, which roughly doubles the backend-tests
   portion of CI for that runner. Maintainers may prefer to swap ndarray out once they're confident
   flex is equivalent.
5. I did not touch the burn-book / contributor-book docs, crates/burn-no-std-tests, or
   crates/burn-core's TestBackend, per the earlier triage. Those remain as follow-up work.

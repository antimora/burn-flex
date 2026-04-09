#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use burn_backend::{DType, Element, TensorData, TensorMetadata};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::layout::Layout;

/// CPU tensor primitive for the Flex backend.
///
/// Uses type-erased byte storage with runtime dtype and Arc-based sharing.
/// Clone is O(1) (refcount increment). Copy-on-write for mutations.
#[derive(Clone)]
pub struct FlexTensor {
    /// Shared byte storage. Clone increments refcount.
    data: Arc<Bytes>,
    /// Layout describing shape, strides, and offset.
    layout: Layout,
    /// Runtime data type.
    dtype: DType,
}

impl fmt::Debug for FlexTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlexTensor")
            .field("shape", self.layout.shape())
            .field("dtype", &self.dtype)
            .field("contiguous", &self.layout.is_contiguous())
            .field("unique", &self.is_unique())
            .finish()
    }
}

impl FlexTensor {
    /// Create a new tensor from bytes, layout, and dtype.
    pub fn new(data: Bytes, layout: Layout, dtype: DType) -> Self {
        Self {
            data: Arc::new(data),
            layout,
            dtype,
        }
    }

    /// Create a tensor from TensorData.
    pub fn from_data(data: TensorData) -> Self {
        let shape = data.shape.clone();
        let layout = Layout::contiguous(shape);
        let dtype = data.dtype;
        Self {
            data: Arc::new(data.bytes),
            layout,
            dtype,
        }
    }

    /// Convert tensor to TensorData.
    ///
    /// If non-contiguous or shared, this will copy data.
    pub fn into_data(self) -> TensorData {
        if self.layout.is_contiguous() && self.layout.start_offset() == 0 {
            let expected_bytes = self.layout.num_elements() * dtype_size(self.dtype);
            debug_assert!(
                expected_bytes <= self.data.len(),
                "into_data: buffer ({} bytes) too small for {} elements of {:?}",
                self.data.len(),
                self.layout.num_elements(),
                self.dtype
            );
            if self.data.len() == expected_bytes {
                // Buffer exactly matches logical size; try zero-copy unwrap
                match Arc::try_unwrap(self.data) {
                    Ok(bytes) => TensorData {
                        bytes,
                        shape: self.layout.shape().clone(),
                        dtype: self.dtype,
                    },
                    Err(arc) => {
                        let bytes = Bytes::from_bytes_vec((*arc)[..expected_bytes].to_vec());
                        TensorData {
                            bytes,
                            shape: self.layout.shape().clone(),
                            dtype: self.dtype,
                        }
                    }
                }
            } else {
                // Contiguous at offset 0 but buffer is oversized (e.g., narrowed view).
                // Truncate to exact logical size.
                let bytes = Bytes::from_bytes_vec(self.data[..expected_bytes].to_vec());
                TensorData {
                    bytes,
                    shape: self.layout.shape().clone(),
                    dtype: self.dtype,
                }
            }
        } else {
            // Non-contiguous or non-zero offset: copy to contiguous layout
            self.to_contiguous().into_data()
        }
    }

    /// Check if this tensor has exclusive ownership of its data.
    ///
    /// When true, in-place mutations are safe without copying.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Get the layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Create a new tensor with a different layout but sharing the same data.
    ///
    /// This is a zero-copy operation used for operations like flip, transpose, etc.
    pub fn with_layout(self, layout: Layout) -> Self {
        Self {
            data: self.data,
            layout,
            dtype: self.dtype,
        }
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Check if tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Get the raw bytes (read-only).
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get a clone of the Arc for sharing data with a new layout.
    ///
    /// Use this for zero-copy view operations (reshape, transpose, slice).
    pub fn data_arc(&self) -> Arc<Bytes> {
        Arc::clone(&self.data)
    }

    /// Create a tensor from shared data, layout, and dtype.
    ///
    /// Use this for zero-copy view operations.
    pub fn from_arc(data: Arc<Bytes>, layout: Layout, dtype: DType) -> Self {
        Self {
            data,
            layout,
            dtype,
        }
    }

    /// Zero-copy typed view of the full storage buffer.
    ///
    /// Use with `StridedIter` for non-contiguous access, or with
    /// `layout().contiguous_offsets()` for the contiguous fast path.
    ///
    /// # Panics
    /// Debug-asserts if `E::dtype()` doesn't match the tensor's dtype.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn storage<E: Element + bytemuck::Pod>(&self) -> &[E] {
        debug_assert!(
            E::dtype() == self.dtype
                || (matches!(self.dtype, DType::Bool(_)) && E::dtype() == DType::U8),
            "storage: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        bytemuck::cast_slice(&self.data)
    }

    /// Mutable typed view with copy-on-write semantics.
    ///
    /// If the tensor is shared (refcount > 1), this will copy the data first.
    /// For in-place operations, prefer `try_storage_mut()` which returns None
    /// if shared, allowing you to choose an alternative strategy.
    ///
    /// # Panics
    /// Debug-asserts if `E::dtype()` doesn't match the tensor's dtype.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn storage_mut<E: Element + bytemuck::Pod>(&mut self) -> &mut [E] {
        debug_assert!(
            E::dtype() == self.dtype
                || (matches!(self.dtype, DType::Bool(_)) && E::dtype() == DType::U8),
            "storage_mut: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        // COW: clone data if shared
        let bytes = Arc::make_mut(&mut self.data);
        bytemuck::cast_slice_mut(bytes)
    }

    /// Try to get mutable storage without copying.
    ///
    /// Returns `Some` if tensor is uniquely owned, `None` if shared.
    /// Use this when you want to avoid the implicit copy in `storage_mut()`.
    /// Note: Bool tensors are stored as u8, so both Bool and U8 dtypes accept u8 access.
    pub fn try_storage_mut<E: Element + bytemuck::Pod>(&mut self) -> Option<&mut [E]> {
        debug_assert!(
            E::dtype() == self.dtype
                || (matches!(self.dtype, DType::Bool(_)) && E::dtype() == DType::U8),
            "try_storage_mut: dtype mismatch (expected {:?}, got {:?})",
            self.dtype,
            E::dtype()
        );
        if self.is_unique() {
            // Safe: we're the only owner
            let bytes = Arc::get_mut(&mut self.data)?;
            Some(bytemuck::cast_slice_mut(bytes))
        } else {
            None
        }
    }

    /// Get typed slice view (zero-cost if contiguous and offset is 0).
    ///
    /// Returns None if dtype doesn't match E or tensor is non-contiguous.
    pub fn as_slice<E: Element + bytemuck::Pod>(&self) -> Option<&[E]> {
        if E::dtype() != self.dtype {
            return None;
        }
        let storage: &[E] = self.storage();
        self.layout
            .contiguous_offsets()
            .map(|(start, end)| &storage[start..end])
    }

    /// Create an empty tensor with given shape and dtype.
    pub fn empty(shape: Shape, dtype: DType) -> Self {
        let num_elements = shape.num_elements();
        let elem_size = dtype_size(dtype);
        let bytes = Bytes::from_bytes_vec(alloc::vec![0u8; num_elements * elem_size]);
        let layout = Layout::contiguous(shape);
        Self {
            data: Arc::new(bytes),
            layout,
            dtype,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        Self::empty(shape, dtype)
    }

    /// Create a tensor filled with `n` copies of a typed value.
    pub fn filled_typed<E: bytemuck::Pod + Send + Sync>(
        shape: Shape,
        dtype: DType,
        value: E,
    ) -> Self {
        debug_assert_eq!(
            dtype_size(dtype),
            core::mem::size_of::<E>(),
            "filled_typed: dtype size mismatch"
        );
        let n = shape.num_elements();
        let data = alloc::vec![value; n];
        let bytes = Bytes::from_elems(data);
        Self {
            data: Arc::new(bytes),
            layout: Layout::contiguous(shape),
            dtype,
        }
    }

    /// Copy to contiguous layout if needed.
    //
    // Note: `collapse_for_copy` is a free function below this impl block.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() && self.layout.start_offset() == 0 {
            return self.clone();
        }

        // Copy data to new contiguous buffer
        match self.dtype {
            DType::F64 => self.copy_contiguous::<f64>(),
            DType::F32 => self.copy_contiguous::<f32>(),
            DType::F16 => self.copy_contiguous::<f16>(),
            DType::BF16 => self.copy_contiguous::<bf16>(),
            DType::I64 => self.copy_contiguous::<i64>(),
            DType::I32 => self.copy_contiguous::<i32>(),
            DType::I16 => self.copy_contiguous::<i16>(),
            DType::I8 => self.copy_contiguous::<i8>(),
            DType::U64 => self.copy_contiguous::<u64>(),
            DType::U32 => self.copy_contiguous::<u32>(),
            DType::U16 => self.copy_contiguous::<u16>(),
            DType::U8 => self.copy_contiguous::<u8>(),
            DType::Bool(_) => self.copy_contiguous::<u8>(), // bool as u8
            _ => panic!("Unsupported dtype for contiguous copy: {:?}", self.dtype),
        }
    }

    fn copy_contiguous<E: Element + bytemuck::Pod>(&self) -> Self {
        let src: &[E] = bytemuck::cast_slice(&self.data);
        let n = self.layout.num_elements();
        let mut dst = Vec::with_capacity(n);

        // Try to reduce the layout to a 2D transpose first, then fall
        // back to the generic ND strided iterator only if collapse
        // can't simplify it further. Collapsing squeezes size-1 dims
        // and merges adjacent dims whose strides line up, which turns
        // e.g. a permuted `[N, H, W, C]` ConvNeXt layer-norm input
        // into a simple 2D `[H*W, C]` transpose. Without this the 4D
        // ND fallback scalar-walks the whole tensor; see issue #64
        // item 2.
        let shape_vec = self.layout.shape().to_vec();
        let (collapsed_shape, collapsed_strides) =
            collapse_for_copy(&shape_vec, self.layout.strides());
        let offset = self.layout.start_offset() as isize;
        let all_positive = collapsed_strides.iter().all(|&s| s >= 0);

        if collapsed_shape.len() <= 1 && all_positive {
            // 0-D or 1-D after collapse: either a scalar or a
            // contiguous run with a uniform stride.
            debug_assert_eq!(n, collapsed_shape.iter().product::<usize>().max(1));
            // SAFETY: capacity is n; we fill every position.
            unsafe { dst.set_len(n) };
            if collapsed_shape.is_empty() {
                if n > 0 {
                    dst[0] = src[offset as usize];
                }
            } else {
                let len = collapsed_shape[0];
                let stride = collapsed_strides[0];
                if stride == 1 {
                    dst[..len].copy_from_slice(&src[offset as usize..offset as usize + len]);
                } else {
                    for (i, slot) in dst.iter_mut().take(len).enumerate() {
                        let idx = (offset + i as isize * stride) as usize;
                        *slot = src[idx];
                    }
                }
            }
        } else if collapsed_shape.len() == 2 && all_positive {
            // 2D positive-stride (transpose-like): tile both dims to
            // keep reads in cache. This is the hot path for permuted
            // ConvNeXt inputs and transposed matmul operands alike.
            let (rows, cols) = (collapsed_shape[0], collapsed_shape[1]);
            let (row_stride, col_stride) = (collapsed_strides[0], collapsed_strides[1]);
            const TILE: usize = 16;

            debug_assert_eq!(rows * cols, n, "2D strides must cover all elements");
            // SAFETY: capacity is n. The tiled loops visit every
            // (row, col) pair exactly once, writing all n positions.
            unsafe { dst.set_len(n) };

            // Pick the loop nesting so that the innermost read walks
            // the *smaller* source stride. Otherwise we'd read with a
            // large stride on the hot inner loop and trash the cache.
            // For a [N,C,H,W].permute([0,2,3,1]) ConvNeXt layer norm
            // the collapsed strides are [1, 54656] (row_stride << col),
            // so the existing col-inner nesting did ~54k-apart reads;
            // we swap to row-inner here to read contiguously.
            if row_stride <= col_stride {
                for col_tile in (0..cols).step_by(TILE) {
                    let col_end = (col_tile + TILE).min(cols);
                    for row_tile in (0..rows).step_by(TILE) {
                        let row_end = (row_tile + TILE).min(rows);
                        for col in col_tile..col_end {
                            let col_base = offset + col as isize * col_stride;
                            for row in row_tile..row_end {
                                let idx = (col_base + row as isize * row_stride) as usize;
                                unsafe {
                                    *dst.get_unchecked_mut(row * cols + col) = src[idx];
                                }
                            }
                        }
                    }
                }
            } else {
                for row_tile in (0..rows).step_by(TILE) {
                    let row_end = (row_tile + TILE).min(rows);
                    for col_tile in (0..cols).step_by(TILE) {
                        let col_end = (col_tile + TILE).min(cols);
                        for row in row_tile..row_end {
                            let row_base =
                                offset + row as isize * row_stride + col_tile as isize * col_stride;
                            let dst_base = row * cols + col_tile;
                            for c in 0..(col_end - col_tile) {
                                let idx = (row_base + c as isize * col_stride) as usize;
                                unsafe {
                                    *dst.get_unchecked_mut(dst_base + c) = src[idx];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // General fallback using strided iterator. Covers negative
            // strides (flipped tensors) and ND layouts that can't be
            // collapsed to 2D.
            for idx in crate::strided_index::StridedIter::new(&self.layout) {
                dst.push(src[idx]);
            }
        }

        let bytes = Bytes::from_elems(dst);
        let layout = Layout::contiguous(self.layout.shape().clone());
        Self {
            data: Arc::new(bytes),
            layout,
            dtype: self.dtype,
        }
    }

    /// Reshape tensor. Zero-copy if contiguous.
    pub fn reshape(&self, new_shape: Shape) -> Self {
        debug_assert_eq!(
            self.layout.num_elements(),
            new_shape.num_elements(),
            "reshape must preserve total elements"
        );

        if let Some(new_layout) = self.layout.reshape(new_shape.clone()) {
            Self {
                data: Arc::clone(&self.data),
                layout: new_layout,
                dtype: self.dtype,
            }
        } else {
            // Non-contiguous: copy first
            self.to_contiguous().reshape(new_shape)
        }
    }

    /// Transpose two dimensions. Zero-copy (metadata only).
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.transpose(dim1, dim2),
            dtype: self.dtype,
        }
    }

    /// Narrow/slice along a dimension. Zero-copy (metadata only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.narrow(dim, start, len),
            dtype: self.dtype,
        }
    }

    /// Permute dimensions according to axes. Zero-copy (metadata only).
    pub fn permute(&self, axes: &[usize]) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.permute(axes),
            dtype: self.dtype,
        }
    }
}

impl TensorMetadata for FlexTensor {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.layout.shape().clone()
    }

    fn rank(&self) -> usize {
        self.layout.num_dims()
    }
}

/// Collapse a shape/stride pair into the minimum-rank equivalent layout
/// for a contiguous copy. Specifically:
///
/// 1. Squeezes size-1 dims (their stride is irrelevant since the loop
///    never advances past index 0).
/// 2. Merges adjacent dims `(i, i+1)` when `stride[i] == stride[i+1] *
///    shape[i+1]`, meaning the two dims form a single logical run
///    through memory and can be treated as one.
///
/// Canonical example: a 4D ConvNeXt input `[1, 244, 224, 48]` with
/// strides `[2623488, 224, 1, 54656]` (the result of
/// `[N,C,H,W].permute([0, 2, 3, 1])`) collapses to a 2D `[54656, 48]`
/// with strides `[1, 54656]` — a plain 2D transpose that the tiled 2D
/// fast path handles at near-memcpy speed.
///
/// Collapsing preserves the iteration order of a contiguous destination
/// walk, so the resulting `(shape, strides)` pair can be substituted
/// directly into a row-major copy loop without changing the visit
/// order.
fn collapse_for_copy(shape: &[usize], strides: &[isize]) -> (Vec<usize>, Vec<isize>) {
    let mut s: Vec<usize> = Vec::with_capacity(shape.len());
    let mut st: Vec<isize> = Vec::with_capacity(strides.len());

    // Step 1: squeeze size-1 dims.
    for i in 0..shape.len() {
        if shape[i] != 1 {
            s.push(shape[i]);
            st.push(strides[i]);
        }
    }

    // Step 2: merge adjacent dims that form one contiguous run. Walk
    // backwards so merging doesn't invalidate indices ahead of us.
    let mut i = s.len();
    while i >= 2 {
        let lo = i - 2;
        let hi = i - 1;
        if st[lo] == st[hi] * s[hi] as isize {
            s[lo] *= s[hi];
            st[lo] = st[hi];
            s.remove(hi);
            st.remove(hi);
        }
        i -= 1;
    }

    (s, st)
}

/// Get the size in bytes for a dtype element.
pub(crate) fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        DType::I8 | DType::U8 | DType::Bool(_) => 1,
        _ => panic!("Unsupported dtype: {:?}", dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_from_data_roundtrip() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor = FlexTensor::from_data(data.clone());
        let result = tensor.into_data();
        assert_eq!(data.shape, result.shape);
        assert_eq!(data.dtype, result.dtype);
    }

    #[test]
    fn test_collapse_for_copy_squeezes_size1_and_merges_contig() {
        // Case from issue #64 item 2: permuted ConvNeXt input.
        // Shape [1, 244, 224, 48], strides from .permute([0,2,3,1])
        // on an [N,C,H,W]=[1,48,244,224] contiguous tensor.
        let shape = vec![1, 244, 224, 48];
        let strides = vec![2_623_488_isize, 224, 1, 54656];
        let (s, st) = collapse_for_copy(&shape, &strides);
        // After squeezing dim 0 and merging dims 1+2 (which are
        // row-contiguous), we get a 2D transpose of [54656, 48] with
        // the original big stride on the trailing dim.
        assert_eq!(s, vec![54656, 48]);
        assert_eq!(st, vec![1, 54656]);
    }

    #[test]
    fn test_collapse_for_copy_already_contiguous_3d() {
        // Contiguous [2, 3, 4] strides [12, 4, 1] should collapse all
        // the way to a single 1D run because each pair is
        // stride-contiguous.
        let (s, st) = collapse_for_copy(&[2, 3, 4], &[12, 4, 1]);
        assert_eq!(s, vec![24]);
        assert_eq!(st, vec![1]);
    }

    #[test]
    fn test_collapse_for_copy_transpose_2d() {
        // Plain transpose of [3, 5] -> [5, 3] with strides swapped.
        // Cannot collapse further.
        let (s, st) = collapse_for_copy(&[5, 3], &[1, 5]);
        assert_eq!(s, vec![5, 3]);
        assert_eq!(st, vec![1, 5]);
    }

    #[test]
    fn test_collapse_for_copy_all_size1() {
        // All dims size 1: collapses to empty (a scalar).
        let (s, st) = collapse_for_copy(&[1, 1, 1], &[0, 0, 0]);
        assert!(s.is_empty());
        assert!(st.is_empty());
    }

    #[test]
    fn test_to_contiguous_4d_permuted_matches_naive() {
        // Matches the hot shape: [1, 48, 244, 224] permuted to
        // [1, 244, 224, 48]. Verify the collapsed + tiled copy path
        // produces the same bytes as a naive strided iteration.
        let dims = [1, 48, 4, 5];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = FlexTensor::from_data(TensorData::new(data.clone(), dims.to_vec()));
        let permuted = t.permute(&[0, 2, 3, 1]);
        assert!(!permuted.is_contiguous());

        let contig = permuted.to_contiguous();
        assert!(contig.is_contiguous());
        assert_eq!(contig.shape().to_vec(), vec![1, 4, 5, 48]);

        // Build expected via manual index walk.
        let mut expected = Vec::with_capacity(n);
        for h in 0..4 {
            for w in 0..5 {
                for c in 0..48 {
                    // Original [1, 48, 4, 5] linear index:
                    // c * (4*5) + h * 5 + w
                    let idx = c * 20 + h * 5 + w;
                    expected.push(data[idx]);
                }
            }
        }

        let result_data = contig.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        assert_eq!(values, expected.as_slice());
    }

    #[test]
    fn test_reshape() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        let reshaped = tensor.reshape(Shape::from(vec![3, 2]));
        assert_eq!(reshaped.shape().to_vec(), vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        let transposed = tensor.transpose(0, 1);
        assert_eq!(transposed.shape().to_vec(), vec![3, 2]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_clone_is_cheap() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = FlexTensor::from_data(data);

        // Before clone, tensor is unique
        assert!(tensor.is_unique());

        // Clone shares data
        let cloned = tensor.clone();
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Both point to same data
        assert!(core::ptr::eq(
            tensor.bytes().as_ptr(),
            cloned.bytes().as_ptr()
        ));
    }

    #[test]
    fn test_cow_on_mutation() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = FlexTensor::from_data(data);
        let mut cloned = tensor.clone();

        // Both share data
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Mutate cloned - triggers COW
        let storage: &mut [f32] = cloned.storage_mut();
        storage[0] = 99.0;

        // Now cloned has its own copy, tensor is unique again
        assert!(tensor.is_unique());
        assert!(cloned.is_unique());

        // Data is different
        assert_ne!(tensor.bytes().as_ptr(), cloned.bytes().as_ptr());
        assert_eq!(tensor.storage::<f32>()[0], 1.0);
        assert_eq!(cloned.storage::<f32>()[0], 99.0);
    }

    #[test]
    fn test_into_data_narrowed_at_offset_zero() {
        // [1, 2, 3, 4, 5, 6] shape [2, 3]
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = FlexTensor::from_data(data);
        // narrow to first row: shape [1, 3], offset 0, contiguous
        let narrowed = tensor.narrow(0, 0, 1);
        assert!(narrowed.is_contiguous());
        assert_eq!(narrowed.layout().start_offset(), 0);

        let result = narrowed.into_data();
        assert_eq!(result.shape.to_vec(), vec![1, 3]);
        // Must have exactly 3 f32s = 12 bytes, not 24
        assert_eq!(result.bytes.len(), 3 * core::mem::size_of::<f32>());
        let values: Vec<f32> = result.to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}

use alloc::vec::Vec;
use core::fmt;

use burn_backend::{DType, Element, TensorData, TensorMetadata};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::layout::Layout;

/// CPU tensor primitive for the Ember backend.
///
/// Uses type-erased byte storage with runtime dtype. Operations cast to typed
/// slices at the boundary, avoiding enum proliferation while maintaining performance.
#[derive(Clone)]
pub struct EmberTensor {
    /// Raw byte storage (aligned, COW semantics via Arc).
    data: Bytes,
    /// Layout describing shape, strides, and offset.
    layout: Layout,
    /// Runtime data type.
    dtype: DType,
}

impl fmt::Debug for EmberTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmberTensor")
            .field("shape", &self.layout.shape().dims)
            .field("dtype", &self.dtype)
            .field("contiguous", &self.layout.is_contiguous())
            .finish()
    }
}

impl EmberTensor {
    /// Create a new tensor from bytes, layout, and dtype.
    pub fn new(data: Bytes, layout: Layout, dtype: DType) -> Self {
        Self {
            data,
            layout,
            dtype,
        }
    }

    /// Create a tensor from TensorData.
    pub fn from_data(data: TensorData) -> Self {
        let shape = Shape::from(data.shape.clone());
        let layout = Layout::contiguous(shape);
        let dtype = data.dtype;
        Self {
            data: data.bytes,
            layout,
            dtype,
        }
    }

    /// Convert tensor to TensorData.
    ///
    /// If non-contiguous, this will copy data to make it contiguous.
    pub fn into_data(self) -> TensorData {
        if self.layout.is_contiguous() && self.layout.start_offset() == 0 {
            TensorData {
                bytes: self.data,
                shape: self.layout.shape().dims.clone(),
                dtype: self.dtype,
            }
        } else {
            // Non-contiguous: need to copy to contiguous layout
            self.to_contiguous().into_data()
        }
    }

    /// Get the layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Check if tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Get typed slice view (zero-cost if contiguous and offset is 0).
    ///
    /// Returns None if dtype doesn't match E.
    pub fn as_slice<E: Element + bytemuck::Pod>(&self) -> Option<&[E]> {
        if E::dtype() != self.dtype {
            return None;
        }

        // Cast bytes to typed slice
        let all_elements: &[E] = bytemuck::cast_slice(&self.data);

        if let Some((start, end)) = self.layout.contiguous_offsets() {
            Some(&all_elements[start..end])
        } else {
            // For non-contiguous, caller should use iterator or copy first
            None
        }
    }

    /// Create an empty tensor with given shape and dtype.
    pub fn empty(shape: Shape, dtype: DType) -> Self {
        let num_elements = shape.num_elements();
        let elem_size = dtype_size(dtype);
        let bytes = Bytes::from_bytes_vec(alloc::vec![0u8; num_elements * elem_size]);
        let layout = Layout::contiguous(shape);
        Self {
            data: bytes,
            layout,
            dtype,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        Self::empty(shape, dtype)
    }

    /// Copy to contiguous layout if needed.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() && self.layout.start_offset() == 0 {
            return self.clone();
        }

        // Copy data to new contiguous buffer
        match self.dtype {
            DType::F64 => self.copy_contiguous::<f64>(),
            DType::F32 => self.copy_contiguous::<f32>(),
            DType::F16 => self.copy_contiguous_f16(),
            DType::BF16 => self.copy_contiguous_bf16(),
            DType::I64 => self.copy_contiguous::<i64>(),
            DType::I32 => self.copy_contiguous::<i32>(),
            DType::I16 => self.copy_contiguous::<i16>(),
            DType::I8 => self.copy_contiguous::<i8>(),
            DType::U64 => self.copy_contiguous::<u64>(),
            DType::U32 => self.copy_contiguous::<u32>(),
            DType::U16 => self.copy_contiguous::<u16>(),
            DType::U8 => self.copy_contiguous::<u8>(),
            DType::Bool => self.copy_contiguous::<u8>(), // bool as u8
            _ => panic!("Unsupported dtype for contiguous copy: {:?}", self.dtype),
        }
    }

    fn copy_contiguous<E: Element + bytemuck::Pod>(&self) -> Self {
        let src: &[E] = bytemuck::cast_slice(&self.data);
        let mut dst = Vec::with_capacity(self.layout.num_elements());

        for idx in crate::strided_index::StridedIter::new(&self.layout) {
            dst.push(src[idx]);
        }

        let bytes = Bytes::from_elems(dst);
        let layout = Layout::contiguous(self.layout.shape().clone());
        Self {
            data: bytes,
            layout,
            dtype: self.dtype,
        }
    }

    fn copy_contiguous_f16(&self) -> Self {
        let src: &[f16] = bytemuck::cast_slice(&self.data);
        let mut dst = Vec::with_capacity(self.layout.num_elements());

        for idx in crate::strided_index::StridedIter::new(&self.layout) {
            dst.push(src[idx]);
        }

        let bytes = Bytes::from_elems(dst);
        let layout = Layout::contiguous(self.layout.shape().clone());
        Self {
            data: bytes,
            layout,
            dtype: self.dtype,
        }
    }

    fn copy_contiguous_bf16(&self) -> Self {
        let src: &[bf16] = bytemuck::cast_slice(&self.data);
        let mut dst = Vec::with_capacity(self.layout.num_elements());

        for idx in crate::strided_index::StridedIter::new(&self.layout) {
            dst.push(src[idx]);
        }

        let bytes = Bytes::from_elems(dst);
        let layout = Layout::contiguous(self.layout.shape().clone());
        Self {
            data: bytes,
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
                data: self.data.clone(),
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
            data: self.data.clone(),
            layout: self.layout.transpose(dim1, dim2),
            dtype: self.dtype,
        }
    }

    /// Narrow/slice along a dimension. Zero-copy (metadata only).
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        Self {
            data: self.data.clone(),
            layout: self.layout.narrow(dim, start, len),
            dtype: self.dtype,
        }
    }
}

impl TensorMetadata for EmberTensor {
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

/// Get the size in bytes for a dtype element.
fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        DType::I8 | DType::U8 | DType::Bool => 1,
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
        let tensor = EmberTensor::from_data(data.clone());
        let result = tensor.into_data();
        assert_eq!(data.shape, result.shape);
        assert_eq!(data.dtype, result.dtype);
    }

    #[test]
    fn test_reshape() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = EmberTensor::from_data(data);
        let reshaped = tensor.reshape(Shape::from(vec![3, 2]));
        assert_eq!(reshaped.shape().dims, vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = EmberTensor::from_data(data);
        let transposed = tensor.transpose(0, 1);
        assert_eq!(transposed.shape().dims, vec![3, 2]);
        assert!(!transposed.is_contiguous());
    }
}

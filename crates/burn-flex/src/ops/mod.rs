//! Backend operations implementations.

/// Wrapper for raw mutable pointers that can be sent across rayon threads.
///
/// # Safety
///
/// The caller must ensure:
/// - The pointer remains valid for the lifetime of all uses
/// - No two threads write to the same offset
/// - No references to the underlying data exist during writes
#[cfg(feature = "rayon")]
pub(crate) struct SendMutPtr<T>(*mut T);

#[cfg(feature = "rayon")]
unsafe impl<T> Send for SendMutPtr<T> {}
#[cfg(feature = "rayon")]
unsafe impl<T> Sync for SendMutPtr<T> {}

#[cfg(feature = "rayon")]
impl<T> SendMutPtr<T> {
    pub(crate) fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Write `val` at the given element offset.
    ///
    /// # Safety
    /// Offset must be in bounds and no other thread may write to the same offset.
    pub(crate) unsafe fn write(&self, offset: usize, val: T) {
        unsafe { self.0.add(offset).write(val) }
    }

    /// Returns the raw pointer offset by `offset` elements.
    ///
    /// # Safety
    /// Offset must be in bounds.
    pub(crate) unsafe fn ptr_add(&self, offset: usize) -> *mut T {
        unsafe { self.0.add(offset) }
    }
}

mod activation;
pub mod attention;
pub mod binary;
mod bool;
pub mod cat;
pub mod comparison;
pub mod conv;
pub mod cumulative;
pub mod deform_conv;
pub mod expand;
pub mod flip;
mod float;
pub mod gather_scatter;
pub mod grid_sample;
mod int;
pub mod interpolate;
pub mod mask;
pub mod matmul;
mod module;
pub mod pool;
mod qtensor;
pub mod reduce;
pub mod slice;
mod transaction;
pub mod unary;
pub mod unfold;

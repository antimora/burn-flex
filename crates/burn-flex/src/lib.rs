#![cfg_attr(not(feature = "std"), no_std)]

//! # burn-flex
//!
//! A fast, portable CPU backend for [Burn](https://github.com/tracel-ai/burn).
//!
//! ## Features
//!
//! - Pure Rust (no C dependencies)
//! - f16/bf16 support
//! - SIMD acceleration (NEON on ARM, future AVX on x86)
//! - Zero-copy tensor views
//! - Thread-safe by design
//!
//! ## Usage
//!
//! ```ignore
//! use burn_flex::Flex;
//! use burn::tensor::Tensor;
//!
//! let tensor: Tensor<Flex, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());
//! ```

extern crate alloc;

mod backend;
mod layout;
mod qtensor;
mod strided_index;
mod tensor;

pub mod ops;

pub mod simd;

pub use backend::{Flex, FlexDevice};
pub use layout::Layout;
pub use qtensor::FlexQTensor;
pub use tensor::FlexTensor;

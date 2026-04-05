//! Convolution operations using tiled im2col + gemm approach.
//!
//! All convolutions (1D, 2D, 3D) use a unified 3D implementation:
//! - conv1d: adds two size-1 dimensions, calls conv3d, squeezes output
//! - conv2d: adds one size-1 dimension, calls conv3d, squeezes output
//! - conv3d: native implementation
//!
//! Optimizations:
//! - Tiled im2col: Process output in tiles for better cache usage and parallelism
//! - NHWC layout: Convert to channels-last for cache-friendly access
//! - Nested parallelism: Batch and tile dimensions run in parallel via rayon
//! - 1x1 fast path: Skip im2col for pointwise convolutions
//!
//! Supported dtypes: f32, f64, f16 (native gemm), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::conv::{calculate_conv_output_size, calculate_conv_transpose_output_size};
use burn_backend::ops::{ConvOptions, ConvTransposeOptions};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{FlexTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates a convNd function that delegates to a conv3d function via expand/squeeze.
macro_rules! conv_nd_via_3d {
    ($fn_name:ident, $conv3d_fn:ident, $expand_fn:ident, $squeeze_fn:ident, $dim:literal, $Options:ident) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &$Options<$dim>,
        ) -> FlexTensor {
            let (x_3d, weight_3d, options_3d) = $expand_fn(&x, &weight, options);
            let result_3d = $conv3d_fn(x_3d, weight_3d, bias, &options_3d);
            $squeeze_fn(result_3d)
        }
    };
}

/// Generates a bf16 function that converts to f32, calls the f32 variant, converts back.
macro_rules! bf16_via_f32 {
    ($bf16_fn:ident, $f32_fn:ident, $dim:literal, $Options:ident) => {
        pub fn $bf16_fn(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &$Options<$dim>,
        ) -> FlexTensor {
            let x_f32 = convert_bf16_to_f32(&x);
            let weight_f32 = convert_bf16_to_f32(&weight);
            let bias_f32 = bias.map(|b| convert_bf16_to_f32(&b));
            let result_f32 = $f32_fn(x_f32, weight_f32, bias_f32, options);
            convert_f32_to_bf16(&result_f32)
        }
    };
}

/// Generates a conv3d_1x1 function that uses the optimized gemm fast path.
macro_rules! conv3d_1x1_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr) => {
        fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            conv3d_1x1_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

/// Generates a conv3d typed function with shape-aware dispatch:
/// 1x1 fast path → direct (no-im2col) path → tiled im2col+gemm path.
macro_rules! conv3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr, $fn_1x1:ident) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            let w_shape = weight.layout().shape();
            let x_shape = x.layout().shape();
            if is_1x1_conv(w_shape[2], w_shape[3], w_shape[4], options) {
                return $fn_1x1(x, weight, bias, options);
            }
            // Dispatch to direct path for small spatial_out, where gemm's
            // packing overhead dominates compute and per-c_out parallelism
            // beats gemm's internal parallelism.
            let out_d = calculate_conv_output_size(
                w_shape[2], options.stride[0], options.padding[0], options.dilation[0], x_shape[2],
            );
            let out_h = calculate_conv_output_size(
                w_shape[3], options.stride[1], options.padding[1], options.dilation[1], x_shape[3],
            );
            let out_w = calculate_conv_output_size(
                w_shape[4], options.stride[2], options.padding[2], options.dilation[2], x_shape[4],
            );
            let spatial_out = out_d * out_h * out_w;
            if should_use_direct_conv3d(
                w_shape[1], w_shape[2], w_shape[3], w_shape[4], spatial_out, options,
            ) {
                return conv3d_direct_impl::<$T>(
                    x, weight, bias, options, $dtype, $add_fn,
                );
            }
            conv3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

/// Generates a conv_transpose3d typed function.
macro_rules! conv_transpose3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvTransposeOptions<3>,
        ) -> FlexTensor {
            conv_transpose3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $add_fn)
        }
    };
}

// ============================================================================
// Conv1d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv1d_f32,
    conv3d_f32,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f64,
    conv3d_f64,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f16,
    conv3d_f16,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
bf16_via_f32!(conv1d_bf16, conv1d_f32, 1, ConvOptions);

fn expand_1d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvOptions<1>,
) -> (FlexTensor, FlexTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![x_shape[0], x_shape[1], 1, 1, x_shape[2]]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![w_shape[0], w_shape[1], 1, 1, w_shape[2]]));

    let options_3d = ConvOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

fn squeeze_3d_to_1d(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(vec![shape[0], shape[1], shape[4]]))
}

// ============================================================================
// Conv2d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv2d_f32,
    conv3d_f32,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f64,
    conv3d_f64,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f16,
    conv3d_f16,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
bf16_via_f32!(conv2d_bf16, conv2d_f32, 2, ConvOptions);

fn expand_2d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvOptions<2>,
) -> (FlexTensor, FlexTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3],
    ]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape[0], w_shape[1], 1, w_shape[2], w_shape[3],
    ]));

    let options_3d = ConvOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

fn squeeze_3d_to_2d(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(vec![shape[0], shape[1], shape[3], shape[4]]))
}

// ============================================================================
// Conv3d - native implementations
// ============================================================================

conv3d_typed!(
    conv3d_f32,
    f32,
    DType::F32,
    0.0f32,
    gemm_f32,
    |a, b| a + b,
    conv3d_1x1_f32
);
conv3d_typed!(
    conv3d_f64,
    f64,
    DType::F64,
    0.0f64,
    gemm_f64,
    |a, b| a + b,
    conv3d_1x1_f64
);
conv3d_typed!(
    conv3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    conv3d_1x1_f16
);
bf16_via_f32!(conv3d_bf16, conv3d_f32, 3, ConvOptions);

/// Generic 3D convolution implementation using tiled im2col.
///
/// Tiled approach processes output in TILE_SIZE chunks:
/// - Reduces memory usage (smaller im2col buffer per tile)
/// - Enables tile-level parallelism
/// - Improves cache utilization
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_impl<T: bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let out_d = calculate_conv_output_size(kernel_d, stride_d, pad_d, options.dilation[0], in_d);
    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, options.dilation[1], in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, options.dilation[2], in_w);

    // Validate sizes won't overflow index calculations
    let _total = [batch_size, channels_out, out_d, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: output tensor dimensions would overflow index calculations");
    let _col_total = [channels_per_group, kernel_d, kernel_h, kernel_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: kernel dimensions would overflow index calculations");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let col_len = channels_per_group * kernel_d * kernel_h * kernel_w;
    let spatial_out = out_d * out_h * out_w;

    let [dilation_d, dilation_h, dilation_w] = options.dilation;

    // Tile size for processing output pixels. Larger = better GEMM utilization,
    // smaller = more parallelism and better cache usage. 512 is a good balance.
    const TILE_SIZE: usize = 512;
    let num_tiles = spatial_out.div_ceil(TILE_SIZE);

    // Flatten kernel to [c_out, k_d * k_h * k_w * c_in] for GEMM.
    //
    // This is a 2D transpose of shape (channels_per_group, k_spatial) per
    // c_out slice: the source weight is [c_out, c_in, kd, kh, kw] row-major
    // and the target col layout is [c_out, kd, kh, kw, c_in] row-major. The
    // inner (c_in, k_spatial) submatrix needs to be transposed.
    //
    // The old formulation used five nested loops with a 10-term index
    // expression in the inner body, which LLVM cannot autovectorize. This
    // form exposes a simple per-slice 2D transpose with a small k_spatial
    // inner loop that the compiler can unroll and vectorize. For small
    // kernels (which are typical: k=2, k=3, k=1 on non-spatial dims in
    // conv1d/2d), the k_spatial loop gets fully unrolled.
    //
    // For small shapes (e.g. wav2vec2 conv1d late layers) this transpose
    // was previously the dominant cost in conv3d_impl, larger than the
    // actual GEMM compute. See crates/burn-flex-bench-candle for baseline
    // numbers.
    let k_spatial = kernel_d * kernel_h * kernel_w;
    let mut w_flat = vec![zero; channels_out * col_len];
    for c_out in 0..channels_out {
        let src_base = c_out * channels_per_group * k_spatial;
        let dst_base = c_out * col_len;
        for c_in in 0..channels_per_group {
            let src_row = src_base + c_in * k_spatial;
            for k in 0..k_spatial {
                w_flat[dst_base + k * channels_per_group + c_in] = w_data[src_row + k];
            }
        }
    }

    // Convert input to NHWC layout for cache-friendly access in im2col.
    //
    // NOTE: the inner-loop structure here (c innermost, spatial outer)
    // looks non-obvious but is empirically the fastest on aarch64 M3 Max.
    // An alternative formulation with w innermost (contiguous load, strided
    // store) was benchmarked on wav2vec2 shapes and ran 10-35% slower at
    // every layer — modern ARM cores have better load prefetchers than
    // store write-buffers, so strided *loads* + contiguous *stores* wins.
    let nhwc_stride = (
        in_d * in_h * in_w * channels_in,
        in_h * in_w * channels_in,
        in_w * channels_in,
        channels_in,
        1,
    );
    let mut x_nhwc = vec![zero; batch_size * in_d * in_h * in_w * channels_in];
    for b in 0..batch_size {
        for d in 0..in_d {
            for h in 0..in_h {
                for w in 0..in_w {
                    for c in 0..channels_in {
                        let src_idx = b * channels_in * in_d * in_h * in_w
                            + c * in_d * in_h * in_w
                            + d * in_h * in_w
                            + h * in_w
                            + w;
                        let dst_idx = b * nhwc_stride.0
                            + d * nhwc_stride.1
                            + h * nhwc_stride.2
                            + w * nhwc_stride.3
                            + c;
                        x_nhwc[dst_idx] = x_data[src_idx];
                    }
                }
            }
        }
    }

    // Use tiled parallel execution
    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut dst = vec![zero; batch_size * channels_out * spatial_out];
            let dst_ptr = crate::ops::SendMutPtr::new(dst.as_mut_ptr());

            // Process batches and tiles in parallel (nested parallelism)
            (0..batch_size).into_par_iter().for_each(|b| {
                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        // Build im2col for this tile and group
                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM: w_group[out_c_per_group, col_len] @ col_tile[tile_size, col_len]^T
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                unsafe {
                                    debug_assert!(
                                        dst_idx < batch_size * channels_out * spatial_out
                                    );
                                    dst_ptr.write(dst_idx, result[res_idx]);
                                }
                            }
                        }
                    }
                });
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            // Sequential path with tiling
            let mut output = vec![zero; batch_size * channels_out * spatial_out];

            for b in 0..batch_size {
                for tile_idx in 0..num_tiles {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM for this group
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                output[dst_idx] = result[res_idx];
                            }
                        }
                    }
                }
            }
            output
        }
    };

    if let Some(bias) = bias {
        let mut output = output;
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial_out,
            add_fn,
        );
        let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    } else {
        let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    }
}

/// Decide whether to use the direct (no-im2col) path or the tiled+gemm path.
///
/// Currently returns `false` unconditionally — the direct path
/// (`conv3d_direct_impl` + `ConvElement::conv_direct_sweep`) is fully
/// implemented and produces correct results on every conv test, but on
/// M3 Max it loses to the tiled+gemm path by 1.2-2x at small shapes where
/// it was intended to win (wav2vec2 feature extractor L3-L6).
///
/// Root cause of the gap: gemm's microkernel uses register blocking across
/// multiple output columns, reusing loaded weight vectors across several
/// dot products before advancing. Our direct sweep computes one output at
/// a time, reloading weights into registers for each. Closing the gap
/// requires writing a similar register-blocked microkernel in the sweep
/// helper (e.g. compute 4 or 8 outputs in parallel per iteration of the
/// c_in loop). That is substantial work and is tracked separately.
///
/// The infrastructure (trait `ConvElement`, `#[macerator::with_simd]`
/// sweep helpers, dispatcher hook) is all in place so that future work is
/// a localized change to `conv_direct_sweep_{f32,f64}_simd` and flipping
/// this function's return value once the microkernel matches gemm.
fn should_use_direct_conv3d(
    _channels_per_group: usize,
    _kernel_d: usize,
    _kernel_h: usize,
    _kernel_w: usize,
    _spatial_out: usize,
    _options: &ConvOptions<3>,
) -> bool {
    false
}

/// Direct convolution path: no im2col materialization, no gemm call.
///
/// For each output element we sum `k_spatial` `vec_dot`s between a
/// `c_in`-contiguous weight slice and a `c_in`-contiguous input patch slice
/// from the NHWC-laid-out input. Parallelism is over the flattened
/// `(batch, c_out)` index — much finer-grained than the tiled path, which
/// matters at small `spatial_out` where gemm's internal parallelism cannot
/// saturate multiple cores.
///
/// This path is unified across conv1d/conv2d/conv3d via the same 5D shape
/// convention (conv1d and conv2d expand to singleton dims) — every
/// improvement here benefits all three.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_direct_impl<T: ConvElement>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let [dilation_d, dilation_h, dilation_w] = options.dilation;

    let out_d = calculate_conv_output_size(kernel_d, stride_d, pad_d, dilation_d, in_d);
    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, in_w);
    let spatial_out = out_d * out_h * out_w;

    let k_spatial = kernel_d * kernel_h * kernel_w;
    let col_len = channels_per_group * k_spatial;

    // Dispatcher guarantees groups == 1 here.
    debug_assert_eq!(options.groups, 1);

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();
    let zero = T::zero();

    // Preprocess weight from [c_out, c_in, kd, kh, kw] row-major into
    // [c_out, kd*kh*kw, c_in] row-major. This exposes a `c_in`-contiguous
    // slice per `(c_out, k)` pair, which is exactly what `conv_direct_sweep`
    // needs for its SIMD vload pattern.
    let mut w_reshaped = vec![zero; channels_out * col_len];
    for c_out in 0..channels_out {
        let src_base = c_out * channels_per_group * k_spatial;
        let dst_base = c_out * col_len;
        for c_in in 0..channels_per_group {
            let src_row = src_base + c_in * k_spatial;
            for k in 0..k_spatial {
                w_reshaped[dst_base + k * channels_per_group + c_in] = w_data[src_row + k];
            }
        }
    }

    // Preprocess input: NCHW -> NHWC, giving us a `c_in`-contiguous slice per
    // `(b, d, h, w)` position.
    let in_spatial = in_d * in_h * in_w;
    let mut x_nhwc = vec![zero; batch_size * in_spatial * channels_in];
    for b in 0..batch_size {
        for c in 0..channels_in {
            let src_c_base = (b * channels_in + c) * in_spatial;
            for d in 0..in_d {
                for h in 0..in_h {
                    for w_ in 0..in_w {
                        let src_idx = src_c_base + (d * in_h + h) * in_w + w_;
                        let dst_idx = ((b * in_d + d) * in_h + h) * in_w * channels_in
                            + w_ * channels_in
                            + c;
                        x_nhwc[dst_idx] = x_data[src_idx];
                    }
                }
            }
        }
    }

    // Any padding? If not, the sweep's bounds check branches are never
    // taken and the compiler can elide them via branch prediction.
    let has_padding = pad_d != 0 || pad_h != 0 || pad_w != 0;

    // Loop order: k_spatial OUTER SERIAL, c_out PARALLEL INNER.
    //
    //   * Per outer k iteration, all c_out threads read the *same* slice of
    //     input (the positions for that k). L2/L3 serves these reads once
    //     and the data is reused across the c_out parallel sweep.
    //   * The output buffer is accumulated in place across k iterations, so
    //     each output cell stays warm in cache across its k_spatial updates.
    //
    // Inside each (b, kd, kh, kw, c_out) sweep, `T::conv_direct_sweep` is
    // called once and processes all `spatial_out` output positions. That is
    // where macerator's SIMD dispatch happens — amortized over a full sweep.
    let mut dst = vec![zero; batch_size * channels_out * spatial_out];

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let dst_ptr = crate::ops::SendMutPtr::new(dst.as_mut_ptr());
        for b in 0..batch_size {
            let x_b_slice_start = b * in_spatial * channels_in;
            let x_b_slice_end = x_b_slice_start + in_spatial * channels_in;
            let dst_b_base = b * channels_out * spatial_out;

            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let k_idx = (kd * kernel_h + kh) * kernel_w + kw;
                        let k_offset_c = k_idx * channels_per_group;
                        let params = SweepParams {
                            out_h,
                            out_w,
                            in_d,
                            in_h,
                            in_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            kd,
                            kh,
                            kw,
                            pad_d,
                            pad_h,
                            pad_w,
                            has_padding,
                        };

                        let x_batch = &x_nhwc[x_b_slice_start..x_b_slice_end];
                        let w_reshaped_ref = &w_reshaped;

                        (0..channels_out).into_par_iter().for_each(|c_out| {
                            let w_start = c_out * col_len + k_offset_c;
                            let w_slice =
                                &w_reshaped_ref[w_start..w_start + channels_per_group];
                            let dst_c_base = dst_b_base + c_out * spatial_out;
                            // Safety: each c_out owns a disjoint
                            // `spatial_out`-length range of the output buffer,
                            // and the k loop is serial OUTSIDE this par_iter,
                            // so no two threads ever touch the same position.
                            let dst_row: &mut [T] = unsafe {
                                core::slice::from_raw_parts_mut(
                                    dst_ptr.ptr_add(dst_c_base),
                                    spatial_out,
                                )
                            };
                            T::conv_direct_sweep(
                                w_slice,
                                x_batch,
                                dst_row,
                                &params,
                                channels_per_group,
                            );
                        });
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        for b in 0..batch_size {
            let x_b_slice_start = b * in_spatial * channels_in;
            let x_b_slice_end = x_b_slice_start + in_spatial * channels_in;
            let dst_b_base = b * channels_out * spatial_out;
            let x_batch = &x_nhwc[x_b_slice_start..x_b_slice_end];

            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let k_idx = (kd * kernel_h + kh) * kernel_w + kw;
                        let k_offset_c = k_idx * channels_per_group;
                        let params = SweepParams {
                            out_h,
                            out_w,
                            in_d,
                            in_h,
                            in_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            kd,
                            kh,
                            kw,
                            pad_d,
                            pad_h,
                            pad_w,
                            has_padding,
                        };

                        for c_out in 0..channels_out {
                            let w_start = c_out * col_len + k_offset_c;
                            let w_slice =
                                &w_reshaped[w_start..w_start + channels_per_group];
                            let dst_c_base = dst_b_base + c_out * spatial_out;
                            let dst_row =
                                &mut dst[dst_c_base..dst_c_base + spatial_out];
                            T::conv_direct_sweep(
                                w_slice,
                                x_batch,
                                dst_row,
                                &params,
                                channels_per_group,
                            );
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut dst,
            bias_data,
            batch_size,
            channels_out,
            spatial_out,
            add_fn,
        );
    }
    FlexTensor::new(Bytes::from_elems(dst), Layout::contiguous(out_shape), dtype)
}

/// Build im2col tile for a range of output positions.
///
/// Fills `col_tile` with shape [tile_size, col_len] where each row is a flattened
/// patch from the NHWC input for one output position.
#[allow(clippy::too_many_arguments)]
fn im2col_3d_tile<T: bytemuck::Pod + Copy>(
    col_tile: &mut [T],
    x_nhwc: &[T],
    tile_start: usize,
    tile_end: usize,
    out_h: usize,
    out_w: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    channels_per_group: usize,
    col_len: usize,
    b: usize,
    in_c_start: usize,
    nhwc_stride: (usize, usize, usize, usize, usize),
) {
    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
        let out_d_idx = global_idx / (out_h * out_w);
        let rem = global_idx % (out_h * out_w);
        let out_h_idx = rem / out_w;
        let out_w_idx = rem % out_w;

        let mut col_offset = 0;
        for kd in 0..kernel_d {
            let id = (out_d_idx * stride_d + kd * dilation_d) as isize - pad_d as isize;
            for kh in 0..kernel_h {
                let ih = (out_h_idx * stride_h + kh * dilation_h) as isize - pad_h as isize;
                for kw in 0..kernel_w {
                    let iw = (out_w_idx * stride_w + kw * dilation_w) as isize - pad_w as isize;

                    if id >= 0
                        && id < in_d as isize
                        && ih >= 0
                        && ih < in_h as isize
                        && iw >= 0
                        && iw < in_w as isize
                    {
                        let id = id as usize;
                        let ih = ih as usize;
                        let iw = iw as usize;
                        let inp_base = b * nhwc_stride.0
                            + id * nhwc_stride.1
                            + ih * nhwc_stride.2
                            + iw * nhwc_stride.3
                            + in_c_start;
                        for c in 0..channels_per_group {
                            col_tile[local_idx * col_len + col_offset] = x_nhwc[inp_base + c];
                            col_offset += 1;
                        }
                    } else {
                        // Padding: skip channels_per_group positions (already zero)
                        col_offset += channels_per_group;
                    }
                }
            }
        }
    }
}

/// Check if this is a 1x1 convolution that can use the fast path.
fn is_1x1_conv(
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    options: &ConvOptions<3>,
) -> bool {
    kernel_d == 1
        && kernel_h == 1
        && kernel_w == 1
        && options.stride == [1, 1, 1]
        && options.padding == [0, 0, 0]
}

/// Optimized 1x1 convolution: skip im2col, use gemm directly.
///
/// For 1x1 conv, im2col just transposes input to [spatial, channels] layout.
/// We do the same transpose but avoid the full im2col kernel iteration overhead.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_1x1_impl<T: bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let spatial = x_shape[2] * x_shape[3] * x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let mut dst = vec![zero; batch_size * channels_out * spatial];
            let dst_ptr = crate::ops::SendMutPtr::new(dst.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    // Build X transposed: [spatial, channels_per_group]
                    let mut x_t = vec![zero; spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    // W[out_channels_per_group, channels_per_group], X_T[spatial, channels_per_group]
                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_fn(
                        w_slice,
                        &x_t,
                        out_channels_per_group,
                        channels_per_group,
                        spatial,
                    );

                    // Write result to output
                    let out_offset = b * channels_out * spatial + out_c_start * spatial;
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            result.as_ptr(),
                            dst_ptr.ptr_add(out_offset),
                            out_channels_per_group * spatial,
                        );
                    }
                }
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![zero; batch_size * channels_out * spatial];
            for b in 0..batch_size {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![zero; spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_fn(
                        w_slice,
                        &x_t,
                        out_channels_per_group,
                        channels_per_group,
                        spatial,
                    );

                    let out_offset = b * channels_out * spatial + out_c_start * spatial;
                    output[out_offset..out_offset + out_channels_per_group * spatial]
                        .copy_from_slice(&result);
                }
            }
            output
        }
    };

    if let Some(bias) = bias {
        let mut output = output;
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial,
            add_fn,
        );
        let out_shape = Shape::from(vec![
            batch_size,
            channels_out,
            x_shape[2],
            x_shape[3],
            x_shape[4],
        ]);
        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    } else {
        let out_shape = Shape::from(vec![
            batch_size,
            channels_out,
            x_shape[2],
            x_shape[3],
            x_shape[4],
        ]);
        FlexTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    }
}

conv3d_1x1_typed!(conv3d_1x1_f32, f32, DType::F32, 0.0f32, gemm_f32, |a, b| a
    + b);
conv3d_1x1_typed!(conv3d_1x1_f64, f64, DType::F64, 0.0f64, gemm_f64, |a, b| a
    + b);
conv3d_1x1_typed!(
    conv3d_1x1_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);

// ============================================================================
// Bias addition
// ============================================================================

#[allow(clippy::needless_range_loop)]
fn add_bias<T: Copy>(
    output: &mut [T],
    bias: &[T],
    batch: usize,
    channels: usize,
    spatial: usize,
    add_fn: fn(T, T) -> T,
) {
    for b in 0..batch {
        for c in 0..channels {
            let offset = b * channels * spatial + c * spatial;
            let bias_val = bias[c];
            for i in 0..spatial {
                output[offset + i] = add_fn(output[offset + i], bias_val);
            }
        }
    }
}

// ============================================================================
// Conv Transpose 1d - delegates to conv_transpose3d
// ============================================================================

conv_nd_via_3d!(
    conv_transpose1d_f32,
    conv_transpose3d_f32,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose1d_f64,
    conv_transpose3d_f64,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose1d_f16,
    conv_transpose3d_f16,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
bf16_via_f32!(
    conv_transpose1d_bf16,
    conv_transpose1d_f32,
    1,
    ConvTransposeOptions
);

fn expand_transpose_1d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvTransposeOptions<1>,
) -> (FlexTensor, FlexTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, L] -> [N, C_in, 1, 1, L]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![x_shape[0], x_shape[1], 1, 1, x_shape[2]]));

    // weight: [C_in, C_out, K] -> [C_in, C_out, 1, 1, K]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![w_shape[0], w_shape[1], 1, 1, w_shape[2]]));

    let options_3d = ConvTransposeOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [0, 0, options.padding_out[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv Transpose 2d - delegates to conv_transpose3d
// ============================================================================

conv_nd_via_3d!(
    conv_transpose2d_f32,
    conv_transpose3d_f32,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose2d_f64,
    conv_transpose3d_f64,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose2d_f16,
    conv_transpose3d_f16,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
bf16_via_f32!(
    conv_transpose2d_bf16,
    conv_transpose2d_f32,
    2,
    ConvTransposeOptions
);

fn expand_transpose_2d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvTransposeOptions<2>,
) -> (FlexTensor, FlexTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, H, W] -> [N, C_in, 1, H, W]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3],
    ]));

    // weight: [C_in, C_out, Kh, Kw] -> [C_in, C_out, 1, Kh, Kw]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape[0], w_shape[1], 1, w_shape[2], w_shape[3],
    ]));

    let options_3d = ConvTransposeOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [0, options.padding_out[0], options.padding_out[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv Transpose 3d - core implementation
// ============================================================================

conv_transpose3d_typed!(conv_transpose3d_f32, f32, DType::F32, 0.0f32, |a, b| a + b);
conv_transpose3d_typed!(conv_transpose3d_f64, f64, DType::F64, 0.0f64, |a, b| a + b);
conv_transpose3d_typed!(
    conv_transpose3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);
bf16_via_f32!(
    conv_transpose3d_bf16,
    conv_transpose3d_f32,
    3,
    ConvTransposeOptions
);

/// Sequential scatter loop for transposed convolution.
///
/// Iterates over (batch, output_channel) pairs and accumulates each input*weight
/// product into the output at the appropriate scattered position.
#[allow(clippy::too_many_arguments)]
fn conv_transpose3d_scatter<T: bytemuck::Pod + Copy>(
    output: &mut [T],
    x_data: &[T],
    w_data: &[T],
    batch_size: usize,
    in_channels: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    out_channels_per_group: usize,
    in_channels_per_group: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_d: usize,
    out_h: usize,
    out_w: usize,
    add_fn: fn(T, T) -> T,
) {
    for b in 0..batch_size {
        for oc in 0..out_channels {
            let g = oc / out_channels_per_group;
            let oc_local = oc % out_channels_per_group;

            let ic_start = g * in_channels_per_group;
            let ic_end = ic_start + in_channels_per_group;

            for ic in ic_start..ic_end {
                for id in 0..in_d {
                    for ih in 0..in_h {
                        for iw in 0..in_w {
                            let x_val = x_data[b * in_channels * in_d * in_h * in_w
                                + ic * in_d * in_h * in_w
                                + id * in_h * in_w
                                + ih * in_w
                                + iw];

                            for kd in 0..kernel_d {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let od_raw = id * stride_d + kd * dilation_d;
                                        let oh_raw = ih * stride_h + kh * dilation_h;
                                        let ow_raw = iw * stride_w + kw * dilation_w;

                                        if od_raw < pad_d || oh_raw < pad_h || ow_raw < pad_w {
                                            continue;
                                        }

                                        let od = od_raw - pad_d;
                                        let oh = oh_raw - pad_h;
                                        let ow = ow_raw - pad_w;

                                        if od >= out_d || oh >= out_h || ow >= out_w {
                                            continue;
                                        }

                                        let w_idx = ic
                                            * out_channels_per_group
                                            * kernel_d
                                            * kernel_h
                                            * kernel_w
                                            + oc_local * kernel_d * kernel_h * kernel_w
                                            + kd * kernel_h * kernel_w
                                            + kh * kernel_w
                                            + kw;

                                        let w_val = w_data[w_idx];

                                        let out_idx = b * out_channels * out_d * out_h * out_w
                                            + oc * out_d * out_h * out_w
                                            + od * out_h * out_w
                                            + oh * out_w
                                            + ow;

                                        output[out_idx] =
                                            add_fn(output[out_idx], mul_generic(x_val, w_val));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Generic 3D transposed convolution implementation.
///
/// Transposed convolution "scatters" input values to output positions:
/// For each input position (id, ih, iw), write to output at:
///   od = id * stride_d + kd * dilation_d - padding_d
///   oh = ih * stride_h + kh * dilation_h - padding_h
///   ow = iw * stride_w + kw * dilation_w - padding_w
#[allow(clippy::too_many_arguments)]
fn conv_transpose3d_impl<T: bytemuck::Pod + Clone + Copy + Send + Sync + burn_backend::Element>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvTransposeOptions<3>,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let in_channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    // Weight shape for transpose: [in_channels, out_channels_per_group, kd, kh, kw]
    let out_channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let [pad_out_d, pad_out_h, pad_out_w] = options.padding_out;
    let [dilation_d, dilation_h, dilation_w] = options.dilation;
    let groups = options.groups;

    let out_channels = out_channels_per_group * groups;
    let in_channels_per_group = in_channels / groups;

    let out_d = calculate_conv_transpose_output_size(
        kernel_d, stride_d, pad_d, pad_out_d, dilation_d, in_d,
    );
    let out_h = calculate_conv_transpose_output_size(
        kernel_h, stride_h, pad_h, pad_out_h, dilation_h, in_h,
    );
    let out_w = calculate_conv_transpose_output_size(
        kernel_w, stride_w, pad_w, pad_out_w, dilation_w, in_w,
    );

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let output_size = batch_size * out_channels * out_d * out_h * out_w;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU32, Ordering};

            // For f32, we can use atomic adds for thread-safe accumulation
            // For other types, we fall back to per-(batch, out_channel) parallelism
            if dtype == DType::F32 {
                // Atomic approach for f32: each thread can write to any output position
                let atomic_output: Vec<AtomicU32> =
                    (0..output_size).map(|_| AtomicU32::new(0)).collect();

                (0..batch_size * out_channels)
                    .into_par_iter()
                    .for_each(|k| {
                        let b = k / out_channels;
                        let oc = k % out_channels;
                        let g = oc / out_channels_per_group;
                        let oc_local = oc % out_channels_per_group;

                        let ic_start = g * in_channels_per_group;
                        let ic_end = ic_start + in_channels_per_group;

                        for ic in ic_start..ic_end {
                            for id in 0..in_d {
                                for ih in 0..in_h {
                                    for iw in 0..in_w {
                                        let x_val = x_data[b * in_channels * in_d * in_h * in_w
                                            + ic * in_d * in_h * in_w
                                            + id * in_h * in_w
                                            + ih * in_w
                                            + iw];

                                        for kd in 0..kernel_d {
                                            for kh in 0..kernel_h {
                                                for kw in 0..kernel_w {
                                                    let od_raw = id * stride_d + kd * dilation_d;
                                                    let oh_raw = ih * stride_h + kh * dilation_h;
                                                    let ow_raw = iw * stride_w + kw * dilation_w;

                                                    // Check bounds with padding
                                                    if od_raw < pad_d
                                                        || oh_raw < pad_h
                                                        || ow_raw < pad_w
                                                    {
                                                        continue;
                                                    }

                                                    let od = od_raw - pad_d;
                                                    let oh = oh_raw - pad_h;
                                                    let ow = ow_raw - pad_w;

                                                    if od >= out_d || oh >= out_h || ow >= out_w {
                                                        continue;
                                                    }

                                                    let w_idx = ic
                                                        * out_channels_per_group
                                                        * kernel_d
                                                        * kernel_h
                                                        * kernel_w
                                                        + oc_local * kernel_d * kernel_h * kernel_w
                                                        + kd * kernel_h * kernel_w
                                                        + kh * kernel_w
                                                        + kw;

                                                    let w_val = w_data[w_idx];

                                                    // Multiply and accumulate
                                                    let x_f32: f32 = bytemuck::cast(x_val);
                                                    let w_f32: f32 = bytemuck::cast(w_val);
                                                    let prod = x_f32 * w_f32;

                                                    let out_idx =
                                                        b * out_channels * out_d * out_h * out_w
                                                            + oc * out_d * out_h * out_w
                                                            + od * out_h * out_w
                                                            + oh * out_w
                                                            + ow;

                                                    // Atomic add using compare-exchange
                                                    let atomic = &atomic_output[out_idx];
                                                    loop {
                                                        let old_bits =
                                                            atomic.load(Ordering::Relaxed);
                                                        let old_f32 = f32::from_bits(old_bits);
                                                        let new_f32 = old_f32 + prod;
                                                        let new_bits = new_f32.to_bits();
                                                        if atomic
                                                            .compare_exchange_weak(
                                                                old_bits,
                                                                new_bits,
                                                                Ordering::Relaxed,
                                                                Ordering::Relaxed,
                                                            )
                                                            .is_ok()
                                                        {
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                // Convert atomics back to f32
                atomic_output
                    .into_iter()
                    .map(|a| {
                        let bits = a.load(Ordering::Relaxed);
                        let f = f32::from_bits(bits);
                        bytemuck::cast::<f32, T>(f)
                    })
                    .collect::<Vec<T>>()
            } else {
                // Non-f32: sequential scatter (no atomics needed)
                let mut output = vec![zero; output_size];
                conv_transpose3d_scatter(
                    &mut output,
                    x_data,
                    w_data,
                    batch_size,
                    in_channels,
                    in_d,
                    in_h,
                    in_w,
                    out_channels,
                    out_channels_per_group,
                    in_channels_per_group,
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    stride_d,
                    stride_h,
                    stride_w,
                    pad_d,
                    pad_h,
                    pad_w,
                    dilation_d,
                    dilation_h,
                    dilation_w,
                    out_d,
                    out_h,
                    out_w,
                    add_fn,
                );
                output
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![zero; output_size];
            conv_transpose3d_scatter(
                &mut output,
                x_data,
                w_data,
                batch_size,
                in_channels,
                in_d,
                in_h,
                in_w,
                out_channels,
                out_channels_per_group,
                in_channels_per_group,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                pad_d,
                pad_h,
                pad_w,
                dilation_d,
                dilation_h,
                dilation_w,
                out_d,
                out_h,
                out_w,
                add_fn,
            );
            output
        }
    };

    // Add bias if present
    let output = if let Some(bias) = bias {
        let mut output = output;
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        let spatial = out_d * out_h * out_w;
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            out_channels,
            spatial,
            add_fn,
        );
        output
    } else {
        output
    };

    let out_shape = Shape::from(vec![batch_size, out_channels, out_d, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Generic multiplication for Pod types (used for non-f32 types).
fn mul_generic<T: bytemuck::Pod + Copy>(a: T, b: T) -> T {
    // This is a workaround since we can't require T: Mul<Output=T>
    // We rely on the fact that for f64, f16 we're only using this
    // and the caller passes in the appropriate add_fn
    let a_bytes = bytemuck::bytes_of(&a);
    let b_bytes = bytemuck::bytes_of(&b);

    match a_bytes.len() {
        4 => {
            // f32
            let a_f32 = f32::from_ne_bytes(a_bytes.try_into().unwrap());
            let b_f32 = f32::from_ne_bytes(b_bytes.try_into().unwrap());
            let result = a_f32 * b_f32;
            bytemuck::cast(result)
        }
        8 => {
            // f64
            let a_f64 = f64::from_ne_bytes(a_bytes.try_into().unwrap());
            let b_f64 = f64::from_ne_bytes(b_bytes.try_into().unwrap());
            let result = a_f64 * b_f64;
            bytemuck::cast(result)
        }
        2 => {
            // f16
            let a_f16 = f16::from_ne_bytes(a_bytes.try_into().unwrap());
            let b_f16 = f16::from_ne_bytes(b_bytes.try_into().unwrap());
            let result = f16::from_f32(a_f16.to_f32() * b_f16.to_f32());
            bytemuck::cast(result)
        }
        _ => panic!("unsupported type size for mul_generic"),
    }
}

// ============================================================================
// bf16 conversion helpers
// ============================================================================

fn convert_bf16_to_f32(tensor: &FlexTensor) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
    FlexTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(tensor: &FlexTensor) -> FlexTensor {
    let data: &[f32] = tensor.storage();
    let bf16_data: Vec<bf16> = data.iter().map(|x| bf16::from_f32(*x)).collect();
    FlexTensor::new(
        Bytes::from_elems(bf16_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::BF16,
    )
}

// ============================================================================
// gemm implementations
// ============================================================================

fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    #[cfg(feature = "rayon")]
    let parallelism = if m * n * k >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            parallelism,
        );
    }
    c
}

fn gemm_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    #[cfg(feature = "rayon")]
    let parallelism = if m * n * k >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            0.0f64,
            1.0f64,
            false,
            false,
            false,
            parallelism,
        );
    }
    c
}

fn gemm_f16(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Vec<f16> {
    let mut c = vec![f16::from_f32(0.0); m * n];
    #[cfg(feature = "rayon")]
    let parallelism = if m * n * k >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            k as isize,
            1,
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            false,
            false,
            false,
            parallelism,
        );
    }
    c
}

// ============================================================================
// Direct conv SIMD helpers (via macerator)
// ============================================================================
//
// These are the core SIMD primitives used by `conv3d_direct_impl` via the
// `ConvElement` trait.
//
// Key design decision: dispatch granularity. The naive approach of making
// `vec_dot` a `#[with_simd]` function and calling it per output element
// incurs macerator's dispatch shim on every call — tens of thousands of
// times per conv, dominating runtime. Instead we make the entire *sweep*
// over all output positions for one `(c_out, k_offset)` pair a single
// `#[with_simd]` call. One dispatch per sweep, amortized over
// `spatial_out * c_in` FMAs.
//
// This mirrors the pattern used in `crates/burn-flex/src/simd/portable.rs`
// for element-wise ops: the SIMD-specialized function contains the slice
// loop, not just the inner op.

/// Parameters describing one (c_out, k_offset) sweep for the direct conv.
///
/// Grouped so the `conv_direct_sweep` trait method and its `#[with_simd]`
/// helpers don't drown in positional arguments.
#[derive(Clone, Copy)]
struct SweepParams {
    out_h: usize,
    out_w: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    has_padding: bool,
}

/// The dtype-specialization surface for the direct conv path.
///
/// `conv3d_direct_impl` is generic over `T: ConvElement`, so each impl gets
/// monomorphized per dtype. Inside the impl, the SIMD work lives in a
/// `#[macerator::with_simd]` helper that's called once per sweep (one
/// (c_out, k_offset) pair), amortizing macerator's dispatch over
/// `spatial_out * c_in` FMAs.
///
/// This structure intentionally avoids `Fn` closure generics for the inner
/// hot path — in an earlier revision, a `VD: Fn(&[T], &[T]) -> T` parameter
/// prevented LLVM from inlining through the Fn::call boundary and the code
/// ran at scalar FMA speed despite the intrinsics being correct. Trait
/// methods with concrete monomorphized impls inline cleanly where Fn
/// closures do not.
trait ConvElement: bytemuck::Pod + Copy + Send + Sync + burn_backend::Element + 'static {
    fn zero() -> Self;

    /// Process all `spatial_out` output positions for one (c_out, k_offset)
    /// pair, accumulating `w_slice · x_patch` into `dst_row`.
    ///
    /// `w_slice` has length `channels_per_group` (the c_in weights for this
    /// (c_out, k_offset)). `x_nhwc` is the full NHWC input for this batch.
    /// `dst_row` has length `spatial_out` and accumulates in place.
    fn conv_direct_sweep(
        w_slice: &[Self],
        x_nhwc: &[Self],
        dst_row: &mut [Self],
        params: &SweepParams,
        channels_in: usize,
    );
}

// -- f32 --

#[cfg(feature = "simd")]
#[macerator::with_simd]
fn conv_direct_sweep_f32_simd<S: macerator::Simd>(
    w_slice: &[f32],
    x_nhwc: &[f32],
    dst_row: &mut [f32],
    params: &SweepParams,
    channels_in: usize,
) {
    use macerator::{Scalar, vload_unaligned};
    let lanes = <f32 as Scalar>::lanes::<S>();
    let unroll = 4 * lanes;
    let unroll_len = channels_in / unroll * unroll;
    let simd_len = channels_in / lanes * lanes;

    let SweepParams {
        out_h,
        out_w,
        in_d,
        in_h,
        in_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        kd,
        kh,
        kw,
        pad_d,
        pad_h,
        pad_w,
        has_padding,
        ..
    } = *params;

    // Derive out_d from dst_row.len() and the other spatial dims. dst_row
    // has exactly `out_d * out_h * out_w` elements for this (c_out, k).
    let out_d = dst_row.len() / (out_h * out_w);

    // Nested loops over (od, oh_, ow_) with a running `oidx` counter. This
    // avoids per-output integer divisions — a previous revision flattened
    // the output into a single `for oidx in 0..spatial_out` loop with
    // `oidx / (out_h * out_w)` and friends inside, and LLVM could not
    // strength-reduce the runtime divisions, burning ~4 udiv instructions
    // per output position. At L3 that was milliseconds of pure division
    // cost per conv.
    let mut oidx: usize = 0;
    for od in 0..out_d {
        let base_id_s = (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
        let id_in_range = !has_padding || (base_id_s >= 0 && base_id_s < in_d as isize);
        let id = if has_padding {
            if !id_in_range {
                // Entire (od, *, *) block is out of bounds — skip all
                // output positions for this od.
                oidx += out_h * out_w;
                continue;
            }
            base_id_s as usize
        } else {
            base_id_s as usize
        };

        for oh_ in 0..out_h {
            let base_ih_s = (oh_ * stride_h + kh * dilation_h) as isize - pad_h as isize;
            let ih_in_range =
                !has_padding || (base_ih_s >= 0 && base_ih_s < in_h as isize);
            let ih = if has_padding {
                if !ih_in_range {
                    oidx += out_w;
                    continue;
                }
                base_ih_s as usize
            } else {
                base_ih_s as usize
            };

            for ow_ in 0..out_w {
                let base_iw_s = (ow_ * stride_w + kw * dilation_w) as isize - pad_w as isize;
                if has_padding && (base_iw_s < 0 || base_iw_s >= in_w as isize) {
                    oidx += 1;
                    continue;
                }
                let iw = base_iw_s as usize;

                let x_start = ((id * in_h + ih) * in_w + iw) * channels_in;
                let x_slice = &x_nhwc[x_start..x_start + channels_in];

                // Four independent accumulators for ILP through the FMA pipeline.
                let mut acc0 = (0.0f32).splat::<S>();
                let mut acc1 = (0.0f32).splat::<S>();
                let mut acc2 = (0.0f32).splat::<S>();
                let mut acc3 = (0.0f32).splat::<S>();
                let mut i = 0;
                while i < unroll_len {
                    unsafe {
                        let w0 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i));
                        let x0 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i));
                        acc0 = acc0.mul_add(w0, x0);
                        let w1 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + lanes));
                        let x1 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + lanes));
                        acc1 = acc1.mul_add(w1, x1);
                        let w2 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + 2 * lanes));
                        let x2 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + 2 * lanes));
                        acc2 = acc2.mul_add(w2, x2);
                        let w3 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + 3 * lanes));
                        let x3 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + 3 * lanes));
                        acc3 = acc3.mul_add(w3, x3);
                    }
                    i += unroll;
                }
                while i < simd_len {
                    unsafe {
                        let wv = vload_unaligned::<S, _>(w_slice.as_ptr().add(i));
                        let xv = vload_unaligned::<S, _>(x_slice.as_ptr().add(i));
                        acc0 = acc0.mul_add(wv, xv);
                    }
                    i += lanes;
                }
                let partial = (acc0 + acc1 + acc2 + acc3).reduce_add();
                let mut tail = 0.0f32;
                while i < channels_in {
                    tail += w_slice[i] * x_slice[i];
                    i += 1;
                }
                dst_row[oidx] += partial + tail;
                oidx += 1;
            }
        }
    }
}

/// Scalar fallback for targets without the `simd` feature. LLVM
/// autovectorizes the unrolled 4-accumulator loop on most modern targets.
#[cfg(not(feature = "simd"))]
fn conv_direct_sweep_f32_scalar(
    w_slice: &[f32],
    x_nhwc: &[f32],
    dst_row: &mut [f32],
    params: &SweepParams,
    channels_in: usize,
) {
    let SweepParams {
        out_h,
        out_w,
        in_d,
        in_h,
        in_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        kd,
        kh,
        kw,
        pad_d,
        pad_h,
        pad_w,
        has_padding,
        ..
    } = *params;
    for oidx in 0..dst_row.len() {
        let od = oidx / (out_h * out_w);
        let rem = oidx % (out_h * out_w);
        let oh_ = rem / out_w;
        let ow_ = rem % out_w;
        let (id, ih, iw) = if has_padding {
            let id_s = (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
            let ih_s = (oh_ * stride_h + kh * dilation_h) as isize - pad_h as isize;
            let iw_s = (ow_ * stride_w + kw * dilation_w) as isize - pad_w as isize;
            if id_s < 0
                || id_s >= in_d as isize
                || ih_s < 0
                || ih_s >= in_h as isize
                || iw_s < 0
                || iw_s >= in_w as isize
            {
                continue;
            }
            (id_s as usize, ih_s as usize, iw_s as usize)
        } else {
            (
                od * stride_d + kd * dilation_d,
                oh_ * stride_h + kh * dilation_h,
                ow_ * stride_w + kw * dilation_w,
            )
        };
        let x_start = ((id * in_h + ih) * in_w + iw) * channels_in;
        let x_slice = &x_nhwc[x_start..x_start + channels_in];
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0, 0.0, 0.0);
        let mut i = 0;
        while i + 4 <= channels_in {
            s0 += w_slice[i] * x_slice[i];
            s1 += w_slice[i + 1] * x_slice[i + 1];
            s2 += w_slice[i + 2] * x_slice[i + 2];
            s3 += w_slice[i + 3] * x_slice[i + 3];
            i += 4;
        }
        let mut s = s0 + s1 + s2 + s3;
        while i < channels_in {
            s += w_slice[i] * x_slice[i];
            i += 1;
        }
        dst_row[oidx] += s;
    }
}

impl ConvElement for f32 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn conv_direct_sweep(
        w_slice: &[f32],
        x_nhwc: &[f32],
        dst_row: &mut [f32],
        params: &SweepParams,
        channels_in: usize,
    ) {
        #[cfg(feature = "simd")]
        conv_direct_sweep_f32_simd(w_slice, x_nhwc, dst_row, params, channels_in);
        #[cfg(not(feature = "simd"))]
        conv_direct_sweep_f32_scalar(w_slice, x_nhwc, dst_row, params, channels_in);
    }
}

// -- f64 --

#[cfg(feature = "simd")]
#[macerator::with_simd]
fn conv_direct_sweep_f64_simd<S: macerator::Simd>(
    w_slice: &[f64],
    x_nhwc: &[f64],
    dst_row: &mut [f64],
    params: &SweepParams,
    channels_in: usize,
) {
    use macerator::{Scalar, vload_unaligned};
    let lanes = <f64 as Scalar>::lanes::<S>();
    let unroll = 4 * lanes;
    let unroll_len = channels_in / unroll * unroll;
    let simd_len = channels_in / lanes * lanes;

    let SweepParams {
        out_h,
        out_w,
        in_d,
        in_h,
        in_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        kd,
        kh,
        kw,
        pad_d,
        pad_h,
        pad_w,
        has_padding,
        ..
    } = *params;

    for oidx in 0..dst_row.len() {
        let od = oidx / (out_h * out_w);
        let rem = oidx % (out_h * out_w);
        let oh_ = rem / out_w;
        let ow_ = rem % out_w;

        let (id, ih, iw) = if has_padding {
            let id_s = (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
            let ih_s = (oh_ * stride_h + kh * dilation_h) as isize - pad_h as isize;
            let iw_s = (ow_ * stride_w + kw * dilation_w) as isize - pad_w as isize;
            if id_s < 0
                || id_s >= in_d as isize
                || ih_s < 0
                || ih_s >= in_h as isize
                || iw_s < 0
                || iw_s >= in_w as isize
            {
                continue;
            }
            (id_s as usize, ih_s as usize, iw_s as usize)
        } else {
            (
                od * stride_d + kd * dilation_d,
                oh_ * stride_h + kh * dilation_h,
                ow_ * stride_w + kw * dilation_w,
            )
        };

        let x_start = ((id * in_h + ih) * in_w + iw) * channels_in;
        let x_slice = &x_nhwc[x_start..x_start + channels_in];

        let mut acc0 = (0.0f64).splat::<S>();
        let mut acc1 = (0.0f64).splat::<S>();
        let mut acc2 = (0.0f64).splat::<S>();
        let mut acc3 = (0.0f64).splat::<S>();
        let mut i = 0;
        while i < unroll_len {
            unsafe {
                let w0 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i));
                let x0 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i));
                acc0 = acc0.mul_add(w0, x0);
                let w1 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + lanes));
                let x1 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + lanes));
                acc1 = acc1.mul_add(w1, x1);
                let w2 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + 2 * lanes));
                let x2 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + 2 * lanes));
                acc2 = acc2.mul_add(w2, x2);
                let w3 = vload_unaligned::<S, _>(w_slice.as_ptr().add(i + 3 * lanes));
                let x3 = vload_unaligned::<S, _>(x_slice.as_ptr().add(i + 3 * lanes));
                acc3 = acc3.mul_add(w3, x3);
            }
            i += unroll;
        }
        while i < simd_len {
            unsafe {
                let wv = vload_unaligned::<S, _>(w_slice.as_ptr().add(i));
                let xv = vload_unaligned::<S, _>(x_slice.as_ptr().add(i));
                acc0 = acc0.mul_add(wv, xv);
            }
            i += lanes;
        }
        let partial = (acc0 + acc1 + acc2 + acc3).reduce_add();
        let mut tail = 0.0f64;
        while i < channels_in {
            tail += w_slice[i] * x_slice[i];
            i += 1;
        }
        dst_row[oidx] += partial + tail;
    }
}

#[cfg(not(feature = "simd"))]
fn conv_direct_sweep_f64_scalar(
    w_slice: &[f64],
    x_nhwc: &[f64],
    dst_row: &mut [f64],
    params: &SweepParams,
    channels_in: usize,
) {
    let SweepParams {
        out_h,
        out_w,
        in_d,
        in_h,
        in_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        kd,
        kh,
        kw,
        pad_d,
        pad_h,
        pad_w,
        has_padding,
        ..
    } = *params;
    for oidx in 0..dst_row.len() {
        let od = oidx / (out_h * out_w);
        let rem = oidx % (out_h * out_w);
        let oh_ = rem / out_w;
        let ow_ = rem % out_w;
        let (id, ih, iw) = if has_padding {
            let id_s = (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
            let ih_s = (oh_ * stride_h + kh * dilation_h) as isize - pad_h as isize;
            let iw_s = (ow_ * stride_w + kw * dilation_w) as isize - pad_w as isize;
            if id_s < 0
                || id_s >= in_d as isize
                || ih_s < 0
                || ih_s >= in_h as isize
                || iw_s < 0
                || iw_s >= in_w as isize
            {
                continue;
            }
            (id_s as usize, ih_s as usize, iw_s as usize)
        } else {
            (
                od * stride_d + kd * dilation_d,
                oh_ * stride_h + kh * dilation_h,
                ow_ * stride_w + kw * dilation_w,
            )
        };
        let x_start = ((id * in_h + ih) * in_w + iw) * channels_in;
        let x_slice = &x_nhwc[x_start..x_start + channels_in];
        let mut s = 0.0f64;
        for i in 0..channels_in {
            s += w_slice[i] * x_slice[i];
        }
        dst_row[oidx] += s;
    }
}

impl ConvElement for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn conv_direct_sweep(
        w_slice: &[f64],
        x_nhwc: &[f64],
        dst_row: &mut [f64],
        params: &SweepParams,
        channels_in: usize,
    ) {
        #[cfg(feature = "simd")]
        conv_direct_sweep_f64_simd(w_slice, x_nhwc, dst_row, params, channels_in);
        #[cfg(not(feature = "simd"))]
        conv_direct_sweep_f64_scalar(w_slice, x_nhwc, dst_row, params, channels_in);
    }
}

// -- f16 --

impl ConvElement for f16 {
    #[inline(always)]
    fn zero() -> Self {
        f16::from_f32(0.0)
    }
    fn conv_direct_sweep(
        w_slice: &[f16],
        x_nhwc: &[f16],
        dst_row: &mut [f16],
        params: &SweepParams,
        channels_in: usize,
    ) {
        // f16 accumulates in f32 for precision. Most hardware prefers this
        // over native f16 FMA anyway (width and dynamic range). A dedicated
        // f16 SIMD path could be added later; for now f16 conv is rare.
        let SweepParams {
            out_h,
            out_w,
            in_d,
            in_h,
            in_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            kd,
            kh,
            kw,
            pad_d,
            pad_h,
            pad_w,
            has_padding,
            ..
        } = *params;
        for oidx in 0..dst_row.len() {
            let od = oidx / (out_h * out_w);
            let rem = oidx % (out_h * out_w);
            let oh_ = rem / out_w;
            let ow_ = rem % out_w;
            let (id, ih, iw) = if has_padding {
                let id_s = (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
                let ih_s = (oh_ * stride_h + kh * dilation_h) as isize - pad_h as isize;
                let iw_s = (ow_ * stride_w + kw * dilation_w) as isize - pad_w as isize;
                if id_s < 0
                    || id_s >= in_d as isize
                    || ih_s < 0
                    || ih_s >= in_h as isize
                    || iw_s < 0
                    || iw_s >= in_w as isize
                {
                    continue;
                }
                (id_s as usize, ih_s as usize, iw_s as usize)
            } else {
                (
                    od * stride_d + kd * dilation_d,
                    oh_ * stride_h + kh * dilation_h,
                    ow_ * stride_w + kw * dilation_w,
                )
            };
            let x_start = ((id * in_h + ih) * in_w + iw) * channels_in;
            let x_slice = &x_nhwc[x_start..x_start + channels_in];
            let mut sum = 0.0f32;
            for i in 0..channels_in {
                sum += w_slice[i].to_f32() * x_slice[i].to_f32();
            }
            let prev = dst_row[oidx].to_f32();
            dst_row[oidx] = f16::from_f32(prev + sum);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_conv2d_1x1() {
        // 1x1 convolution uses the optimized fast path (no im2col)
        // Input: [1, 4, 3, 3] (batch=1, channels=4, 3x3 spatial)
        // Weight: [8, 4, 1, 1] (8 output channels, 4 input channels, 1x1 kernel)
        // Output: [1, 8, 3, 3]
        let x_data: Vec<f32> = (0..36).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 4, 3, 3]));

        // Weight: each output channel sums specific input channels
        // Simple weight: first output channel = sum of all input channels
        let mut w_data = vec![0.0f32; 32]; // 8 * 4 = 32
        for i in 0..4 {
            w_data[i] = 1.0; // First output channel: sum all inputs
        }
        w_data[4] = 1.0; // Second output channel: just first input channel
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![8, 4, 1, 1]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 8, 3, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // First output channel should be sum across all 4 input channels at each position
        // Position (0,0): channels 0-3 at (0,0) = 0 + 9 + 18 + 27 = 54
        assert!((out[0] - 54.0).abs() < 1e-5, "got {}", out[0]);

        // Second output channel should be just the first input channel
        // Position (0,0): channel 0 at (0,0) = 0
        let second_ch_start = 9; // 3*3 = 9 elements per channel
        assert!(
            (out[second_ch_start] - 0.0).abs() < 1e-5,
            "got {}",
            out[second_ch_start]
        );
    }

    #[test]
    fn test_conv2d_1x1_with_bias() {
        // 1x1 conv with bias
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 4, 2, 2]));
        let w_data: Vec<f32> = (0..8).map(|_| 0.5f32).collect(); // 2 output channels, 4 input
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![2, 4, 1, 1]));
        let bias = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0f32], vec![2]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, Some(bias), &options);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Each output = 4 * 1.0 * 0.5 + bias = 2.0 + bias
        assert!((out[0] - 12.0).abs() < 1e-5); // First channel: 2.0 + 10.0
        assert!((out[4] - 22.0).abs() < 1e-5); // Second channel: 2.0 + 20.0
    }

    #[test]
    fn test_conv1d_simple() {
        let x_data: Vec<f32> = (1..=5).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 5]));
        let w_data = vec![1.0f32, 1.0, 1.0];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 3]));
        let options = ConvOptions::new([1], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_conv2d_simple() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f32; 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 3, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_with_padding() {
        let x_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));
        let w_data = vec![1.0f32; 9];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out[4], 45.0); // center element sums all
    }

    #[test]
    fn test_conv2d_with_bias() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 1, 4, 4]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 4], vec![1, 1, 2, 2]));
        let bias = FlexTensor::from_data(TensorData::new(vec![10.0f32], vec![1]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, Some(bias), &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 14.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_groups() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 36], vec![1, 4, 3, 3]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 32], vec![4, 2, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_conv3d_simple() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 18], vec![1, 1, 2, 3, 3]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 8], vec![1, 1, 2, 2, 2]));
        let options = ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        let result = conv3d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 1, 2, 2]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 8.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f64; 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_f16() {
        let x_data: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f16(x, weight, None, &options);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }

    #[test]
    fn test_conv2d_bf16() {
        let x_data: Vec<bf16> = (1..=16).map(|x| bf16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<bf16> = vec![bf16::from_f32(1.0); 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_bf16(x, weight, None, &options);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }
}

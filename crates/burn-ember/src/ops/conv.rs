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

use crate::{EmberTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates a convNd function that delegates to a conv3d function via expand/squeeze.
macro_rules! conv_nd_via_3d {
    ($fn_name:ident, $conv3d_fn:ident, $expand_fn:ident, $squeeze_fn:ident, $dim:literal, $Options:ident) => {
        pub fn $fn_name(
            x: EmberTensor,
            weight: EmberTensor,
            bias: Option<EmberTensor>,
            options: &$Options<$dim>,
        ) -> EmberTensor {
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
            x: EmberTensor,
            weight: EmberTensor,
            bias: Option<EmberTensor>,
            options: &$Options<$dim>,
        ) -> EmberTensor {
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
            x: EmberTensor,
            weight: EmberTensor,
            bias: Option<EmberTensor>,
            options: &ConvOptions<3>,
        ) -> EmberTensor {
            conv3d_1x1_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

/// Generates a conv3d typed function with 1x1 fast-path check.
macro_rules! conv3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr, $fn_1x1:ident) => {
        pub fn $fn_name(
            x: EmberTensor,
            weight: EmberTensor,
            bias: Option<EmberTensor>,
            options: &ConvOptions<3>,
        ) -> EmberTensor {
            let w_shape = weight.layout().shape();
            if is_1x1_conv(w_shape.dims[2], w_shape.dims[3], w_shape.dims[4], options) {
                return $fn_1x1(x, weight, bias, options);
            }
            conv3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

/// Generates a conv_transpose3d typed function.
macro_rules! conv_transpose3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr) => {
        pub fn $fn_name(
            x: EmberTensor,
            weight: EmberTensor,
            bias: Option<EmberTensor>,
            options: &ConvTransposeOptions<3>,
        ) -> EmberTensor {
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
    x: &EmberTensor,
    weight: &EmberTensor,
    options: &ConvOptions<1>,
) -> (EmberTensor, EmberTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape.dims[0],
        x_shape.dims[1],
        1,
        1,
        x_shape.dims[2],
    ]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape.dims[0],
        w_shape.dims[1],
        1,
        1,
        w_shape.dims[2],
    ]));

    let options_3d = ConvOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

fn squeeze_3d_to_1d(tensor: EmberTensor) -> EmberTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(vec![
        shape.dims[0],
        shape.dims[1],
        shape.dims[4],
    ]))
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
    x: &EmberTensor,
    weight: &EmberTensor,
    options: &ConvOptions<2>,
) -> (EmberTensor, EmberTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape.dims[0],
        x_shape.dims[1],
        1,
        x_shape.dims[2],
        x_shape.dims[3],
    ]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape.dims[0],
        w_shape.dims[1],
        1,
        w_shape.dims[2],
        w_shape.dims[3],
    ]));

    let options_3d = ConvOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

fn squeeze_3d_to_2d(tensor: EmberTensor) -> EmberTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(vec![
        shape.dims[0],
        shape.dims[1],
        shape.dims[3],
        shape.dims[4],
    ]))
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
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> EmberTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape.dims[0];
    let channels_in = x_shape.dims[1];
    let in_d = x_shape.dims[2];
    let in_h = x_shape.dims[3];
    let in_w = x_shape.dims[4];

    let channels_out = w_shape.dims[0];
    let channels_per_group = w_shape.dims[1];
    let kernel_d = w_shape.dims[2];
    let kernel_h = w_shape.dims[3];
    let kernel_w = w_shape.dims[4];

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

    // Flatten kernel to [c_out, k_d * k_h * k_w * c_in] for GEMM
    // We do this once and reuse for all tiles
    let mut w_flat = vec![zero; channels_out * col_len];
    for c_out in 0..channels_out {
        for kd in 0..kernel_d {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    for c_in in 0..channels_per_group {
                        let w_idx = c_out * channels_per_group * kernel_d * kernel_h * kernel_w
                            + c_in * kernel_d * kernel_h * kernel_w
                            + kd * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw;
                        let flat_idx = c_out * col_len
                            + kd * kernel_h * kernel_w * channels_per_group
                            + kh * kernel_w * channels_per_group
                            + kw * channels_per_group
                            + c_in;
                        w_flat[flat_idx] = w_data[w_idx];
                    }
                }
            }
        }
    }

    // Convert input to NHWC layout for cache-friendly access
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
        EmberTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    } else {
        let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
        EmberTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    }
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
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> EmberTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape.dims[0];
    let channels_in = x_shape.dims[1];
    let spatial = x_shape.dims[2] * x_shape.dims[3] * x_shape.dims[4];

    let channels_out = w_shape.dims[0];
    let channels_per_group = w_shape.dims[1];
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
            x_shape.dims[2],
            x_shape.dims[3],
            x_shape.dims[4],
        ]);
        EmberTensor::new(
            Bytes::from_elems(output),
            Layout::contiguous(out_shape),
            dtype,
        )
    } else {
        let out_shape = Shape::from(vec![
            batch_size,
            channels_out,
            x_shape.dims[2],
            x_shape.dims[3],
            x_shape.dims[4],
        ]);
        EmberTensor::new(
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
    x: &EmberTensor,
    weight: &EmberTensor,
    options: &ConvTransposeOptions<1>,
) -> (EmberTensor, EmberTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, L] -> [N, C_in, 1, 1, L]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape.dims[0],
        x_shape.dims[1],
        1,
        1,
        x_shape.dims[2],
    ]));

    // weight: [C_in, C_out, K] -> [C_in, C_out, 1, 1, K]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape.dims[0],
        w_shape.dims[1],
        1,
        1,
        w_shape.dims[2],
    ]));

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
    x: &EmberTensor,
    weight: &EmberTensor,
    options: &ConvTransposeOptions<2>,
) -> (EmberTensor, EmberTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, H, W] -> [N, C_in, 1, H, W]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape.dims[0],
        x_shape.dims[1],
        1,
        x_shape.dims[2],
        x_shape.dims[3],
    ]));

    // weight: [C_in, C_out, Kh, Kw] -> [C_in, C_out, 1, Kh, Kw]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape.dims[0],
        w_shape.dims[1],
        1,
        w_shape.dims[2],
        w_shape.dims[3],
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
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvTransposeOptions<3>,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
) -> EmberTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape.dims[0];
    let in_channels = x_shape.dims[1];
    let in_d = x_shape.dims[2];
    let in_h = x_shape.dims[3];
    let in_w = x_shape.dims[4];

    // Weight shape for transpose: [in_channels, out_channels_per_group, kd, kh, kw]
    let out_channels_per_group = w_shape.dims[1];
    let kernel_d = w_shape.dims[2];
    let kernel_h = w_shape.dims[3];
    let kernel_w = w_shape.dims[4];

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
    EmberTensor::new(
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

fn convert_bf16_to_f32(tensor: &EmberTensor) -> EmberTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
    EmberTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(tensor: &EmberTensor) -> EmberTensor {
    let data: &[f32] = tensor.storage();
    let bf16_data: Vec<bf16> = data.iter().map(|x| bf16::from_f32(*x)).collect();
    EmberTensor::new(
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
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 4, 3, 3]));

        // Weight: each output channel sums specific input channels
        // Simple weight: first output channel = sum of all input channels
        let mut w_data = vec![0.0f32; 32]; // 8 * 4 = 32
        for i in 0..4 {
            w_data[i] = 1.0; // First output channel: sum all inputs
        }
        w_data[4] = 1.0; // Second output channel: just first input channel
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![8, 4, 1, 1]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);

        assert_eq!(result.layout().shape().dims, vec![1, 8, 3, 3]);
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
        let x = EmberTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 4, 2, 2]));
        let w_data: Vec<f32> = (0..8).map(|_| 0.5f32).collect(); // 2 output channels, 4 input
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![2, 4, 1, 1]));
        let bias = EmberTensor::from_data(TensorData::new(vec![10.0f32, 20.0f32], vec![2]));

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
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 5]));
        let w_data = vec![1.0f32, 1.0, 1.0];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 3]));
        let options = ConvOptions::new([1], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().dims, vec![1, 1, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_conv2d_simple() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f32; 4];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().dims, vec![1, 1, 3, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_with_padding() {
        let x_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));
        let w_data = vec![1.0f32; 9];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out[4], 45.0); // center element sums all
    }

    #[test]
    fn test_conv2d_with_bias() {
        let x = EmberTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 1, 4, 4]));
        let weight = EmberTensor::from_data(TensorData::new(vec![1.0f32; 4], vec![1, 1, 2, 2]));
        let bias = EmberTensor::from_data(TensorData::new(vec![10.0f32], vec![1]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, Some(bias), &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 14.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_groups() {
        let x = EmberTensor::from_data(TensorData::new(vec![1.0f32; 36], vec![1, 4, 3, 3]));
        let weight = EmberTensor::from_data(TensorData::new(vec![1.0f32; 32], vec![4, 2, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().dims, vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_conv3d_simple() {
        let x = EmberTensor::from_data(TensorData::new(vec![1.0f32; 18], vec![1, 1, 2, 3, 3]));
        let weight = EmberTensor::from_data(TensorData::new(vec![1.0f32; 8], vec![1, 1, 2, 2, 2]));
        let options = ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        let result = conv3d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().dims, vec![1, 1, 1, 2, 2]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 8.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f64; 4];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
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
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
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
        let x = EmberTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<bf16> = vec![bf16::from_f32(1.0); 4];
        let weight = EmberTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_bf16(x, weight, None, &options);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }
}

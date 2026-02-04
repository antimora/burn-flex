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
use burn_backend::ops::ConvOptions;
use burn_backend::ops::conv::calculate_conv_output_size;
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{EmberTensor, Layout};

// ============================================================================
// Conv1d - delegates to conv3d
// ============================================================================

/// 1D convolution for f32 via conv3d.
pub fn conv1d_f32(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<1>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_1d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f32(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_1d(result_3d)
}

/// 1D convolution for f64 via conv3d.
pub fn conv1d_f64(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<1>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_1d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f64(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_1d(result_3d)
}

/// 1D convolution for f16 via conv3d.
pub fn conv1d_f16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<1>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_1d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f16(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_1d(result_3d)
}

/// 1D convolution for bf16 via f32 conversion.
pub fn conv1d_bf16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<1>,
) -> EmberTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let weight_f32 = convert_bf16_to_f32(&weight);
    let bias_f32 = bias.map(|b| convert_bf16_to_f32(&b));
    let result_f32 = conv1d_f32(x_f32, weight_f32, bias_f32, options);
    convert_f32_to_bf16(&result_f32)
}

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

/// 2D convolution for f32 via conv3d.
pub fn conv2d_f32(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<2>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_2d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f32(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_2d(result_3d)
}

/// 2D convolution for f64 via conv3d.
pub fn conv2d_f64(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<2>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_2d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f64(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_2d(result_3d)
}

/// 2D convolution for f16 via conv3d.
pub fn conv2d_f16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<2>,
) -> EmberTensor {
    let (x_3d, weight_3d, options_3d) = expand_2d_to_3d(&x, &weight, options);
    let result_3d = conv3d_f16(x_3d, weight_3d, bias, &options_3d);
    squeeze_3d_to_2d(result_3d)
}

/// 2D convolution for bf16 via f32 conversion.
pub fn conv2d_bf16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<2>,
) -> EmberTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let weight_f32 = convert_bf16_to_f32(&weight);
    let bias_f32 = bias.map(|b| convert_bf16_to_f32(&b));
    let result_f32 = conv2d_f32(x_f32, weight_f32, bias_f32, options);
    convert_f32_to_bf16(&result_f32)
}

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

/// 3D convolution for f32 using im2col + gemm.
pub fn conv3d_f32(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    // Fast path for 1x1x1 convolution
    let w_shape = weight.layout().shape();
    if is_1x1_conv(w_shape.dims[2], w_shape.dims[3], w_shape.dims[4], options) {
        return conv3d_1x1_f32(x, weight, bias, options);
    }
    conv3d_impl::<f32>(
        x,
        weight,
        bias,
        options,
        DType::F32,
        0.0f32,
        gemm_f32,
        |a, b| a + b,
    )
}

/// 3D convolution for f64 using im2col + gemm.
pub fn conv3d_f64(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    let w_shape = weight.layout().shape();
    if is_1x1_conv(w_shape.dims[2], w_shape.dims[3], w_shape.dims[4], options) {
        return conv3d_1x1_f64(x, weight, bias, options);
    }
    conv3d_impl::<f64>(
        x,
        weight,
        bias,
        options,
        DType::F64,
        0.0f64,
        gemm_f64,
        |a, b| a + b,
    )
}

/// 3D convolution for f16 using im2col + gemm.
pub fn conv3d_f16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    let w_shape = weight.layout().shape();
    if is_1x1_conv(w_shape.dims[2], w_shape.dims[3], w_shape.dims[4], options) {
        return conv3d_1x1_f16(x, weight, bias, options);
    }
    conv3d_impl::<f16>(
        x,
        weight,
        bias,
        options,
        DType::F16,
        f16::from_f32(0.0),
        gemm_f16,
        |a, b| f16::from_f32(a.to_f32() + b.to_f32()),
    )
}

/// 3D convolution for bf16 via f32 conversion.
pub fn conv3d_bf16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let weight_f32 = convert_bf16_to_f32(&weight);
    let bias_f32 = bias.map(|b| convert_bf16_to_f32(&b));
    let result_f32 = conv3d_f32(x_f32, weight_f32, bias_f32, options);
    convert_f32_to_bf16(&result_f32)
}

/// Generic 3D convolution implementation using tiled im2col.
///
/// Tiled approach processes output in TILE_SIZE chunks:
/// - Reduces memory usage (smaller im2col buffer per tile)
/// - Enables tile-level parallelism
/// - Improves cache utilization
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
    let _groups = options.groups;

    let out_d = calculate_conv_output_size(kernel_d, stride_d, pad_d, options.dilation[0], in_d);
    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, options.dilation[1], in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, options.dilation[2], in_w);

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

            let dst = vec![zero; batch_size * channels_out * spatial_out];

            // Process batches and tiles in parallel (nested parallelism)
            (0..batch_size).into_par_iter().for_each(|b| {
                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Build im2col for this tile
                    let mut col_tile = vec![zero; col_len * tile_size];

                    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                        // Convert linear index to 3D output coords
                        let out_d_idx = global_idx / (out_h * out_w);
                        let rem = global_idx % (out_h * out_w);
                        let out_h_idx = rem / out_w;
                        let out_w_idx = rem % out_w;

                        // Extract im2col patch for this output position
                        let mut col_offset = 0;
                        for kd in 0..kernel_d {
                            let id =
                                (out_d_idx * stride_d + kd * dilation_d) as isize - pad_d as isize;
                            for kh in 0..kernel_h {
                                let ih = (out_h_idx * stride_h + kh * dilation_h) as isize
                                    - pad_h as isize;
                                for kw in 0..kernel_w {
                                    let iw = (out_w_idx * stride_w + kw * dilation_w) as isize
                                        - pad_w as isize;

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
                                        // NHWC access: x_nhwc[b, d, h, w, c]
                                        let inp_base = b * nhwc_stride.0
                                            + id * nhwc_stride.1
                                            + ih * nhwc_stride.2
                                            + iw * nhwc_stride.3;
                                        for c in 0..channels_per_group {
                                            col_tile[local_idx * col_len + col_offset] =
                                                x_nhwc[inp_base + c];
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

                    // GEMM: w_flat[c_out, col_len] @ col_tile[tile_size, col_len]^T -> [c_out, tile_size]
                    let result = gemm_fn(&w_flat, &col_tile, channels_out, col_len, tile_size);

                    // Write results to output
                    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                        for c_out in 0..channels_out {
                            let dst_idx =
                                b * channels_out * spatial_out + c_out * spatial_out + global_idx;
                            let res_idx = c_out * tile_size + local_idx;
                            unsafe {
                                let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                                *ptr = result[res_idx];
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

                    let mut col_tile = vec![zero; col_len * tile_size];

                    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                        let out_d_idx = global_idx / (out_h * out_w);
                        let rem = global_idx % (out_h * out_w);
                        let out_h_idx = rem / out_w;
                        let out_w_idx = rem % out_w;

                        let mut col_offset = 0;
                        for kd in 0..kernel_d {
                            let id =
                                (out_d_idx * stride_d + kd * dilation_d) as isize - pad_d as isize;
                            for kh in 0..kernel_h {
                                let ih = (out_h_idx * stride_h + kh * dilation_h) as isize
                                    - pad_h as isize;
                                for kw in 0..kernel_w {
                                    let iw = (out_w_idx * stride_w + kw * dilation_w) as isize
                                        - pad_w as isize;

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
                                            + iw * nhwc_stride.3;
                                        for c in 0..channels_per_group {
                                            col_tile[local_idx * col_len + col_offset] =
                                                x_nhwc[inp_base + c];
                                            col_offset += 1;
                                        }
                                    } else {
                                        col_offset += channels_per_group;
                                    }
                                }
                            }
                        }
                    }

                    // GEMM: w_flat[c_out, col_len] @ col_tile[tile_size, col_len]^T -> [c_out, tile_size]
                    let result = gemm_fn(&w_flat, &col_tile, channels_out, col_len, tile_size);

                    // Write results to output
                    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                        for c_out in 0..channels_out {
                            let dst_idx =
                                b * channels_out * spatial_out + c_out * spatial_out + global_idx;
                            let res_idx = c_out * tile_size + local_idx;
                            output[dst_idx] = result[res_idx];
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

/// Optimized 1x1 convolution for f32: skip im2col, use gemm directly.
///
/// For 1x1 conv, im2col just transposes input to [spatial, channels] layout.
/// We do the same transpose but avoid the full im2col kernel iteration overhead.
#[cfg(feature = "gemm")]
fn conv3d_1x1_f32(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
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

    let x_data: &[f32] = x.storage();
    let w_data: &[f32] = weight.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst = vec![0.0f32; batch_size * channels_out * spatial];

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    // Build X transposed: [spatial, channels_per_group]
                    let mut x_t = vec![0.0f32; spatial * channels_per_group];
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
                    let result = gemm_f32(
                        w_slice,
                        &x_t,
                        out_channels_per_group,
                        channels_per_group,
                        spatial,
                    );

                    // Write result to output
                    let out_offset = b * channels_out * spatial + out_c_start * spatial;
                    unsafe {
                        let ptr = dst.as_ptr().add(out_offset) as *mut f32;
                        core::ptr::copy_nonoverlapping(
                            result.as_ptr(),
                            ptr,
                            out_channels_per_group * spatial,
                        );
                    }
                }
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![0.0f32; batch_size * channels_out * spatial];
            for b in 0..batch_size {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![0.0f32; spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_f32(
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
        let bias_data: &[f32] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial,
            |a, b| a + b,
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
            DType::F32,
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
            DType::F32,
        )
    }
}

#[cfg(not(feature = "gemm"))]
fn conv3d_1x1_f32(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    conv3d_impl::<f32>(
        x,
        weight,
        bias,
        options,
        DType::F32,
        0.0f32,
        gemm_f32,
        |a, b| a + b,
    )
}

/// Optimized 1x1 convolution for f64.
#[cfg(feature = "gemm")]
fn conv3d_1x1_f64(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
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

    let x_data: &[f64] = x.storage();
    let w_data: &[f64] = weight.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst = vec![0.0f64; batch_size * channels_out * spatial];

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![0.0f64; spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_f64(
                        w_slice,
                        &x_t,
                        out_channels_per_group,
                        channels_per_group,
                        spatial,
                    );

                    let out_offset = b * channels_out * spatial + out_c_start * spatial;
                    unsafe {
                        let ptr = dst.as_ptr().add(out_offset) as *mut f64;
                        core::ptr::copy_nonoverlapping(
                            result.as_ptr(),
                            ptr,
                            out_channels_per_group * spatial,
                        );
                    }
                }
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![0.0f64; batch_size * channels_out * spatial];
            for b in 0..batch_size {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![0.0f64; spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_f64(
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
        let bias_data: &[f64] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial,
            |a, b| a + b,
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
            DType::F64,
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
            DType::F64,
        )
    }
}

#[cfg(not(feature = "gemm"))]
fn conv3d_1x1_f64(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    conv3d_impl::<f64>(
        x,
        weight,
        bias,
        options,
        DType::F64,
        0.0f64,
        gemm_f64,
        |a, b| a + b,
    )
}

/// Optimized 1x1 convolution for f16.
#[cfg(feature = "gemm")]
fn conv3d_1x1_f16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
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

    let x_data: &[f16] = x.storage();
    let w_data: &[f16] = weight.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst = vec![f16::from_f32(0.0); batch_size * channels_out * spatial];

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![f16::from_f32(0.0); spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_f16(
                        w_slice,
                        &x_t,
                        out_channels_per_group,
                        channels_per_group,
                        spatial,
                    );

                    let out_offset = b * channels_out * spatial + out_c_start * spatial;
                    unsafe {
                        let ptr = dst.as_ptr().add(out_offset) as *mut f16;
                        core::ptr::copy_nonoverlapping(
                            result.as_ptr(),
                            ptr,
                            out_channels_per_group * spatial,
                        );
                    }
                }
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![f16::from_f32(0.0); batch_size * channels_out * spatial];
            for b in 0..batch_size {
                for g in 0..groups {
                    let in_c_start = g * channels_per_group;
                    let out_c_start = g * out_channels_per_group;

                    let mut x_t = vec![f16::from_f32(0.0); spatial * channels_per_group];
                    for c in 0..channels_per_group {
                        let src_offset = b * channels_in * spatial + (in_c_start + c) * spatial;
                        for s in 0..spatial {
                            x_t[s * channels_per_group + c] = x_data[src_offset + s];
                        }
                    }

                    let w_offset = out_c_start * channels_per_group;
                    let w_slice =
                        &w_data[w_offset..w_offset + out_channels_per_group * channels_per_group];
                    let result = gemm_f16(
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
        let bias_data: &[f16] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial,
            |a, b| f16::from_f32(a.to_f32() + b.to_f32()),
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
            DType::F16,
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
            DType::F16,
        )
    }
}

#[cfg(not(feature = "gemm"))]
fn conv3d_1x1_f16(
    x: EmberTensor,
    weight: EmberTensor,
    bias: Option<EmberTensor>,
    options: &ConvOptions<3>,
) -> EmberTensor {
    conv3d_impl::<f16>(
        x,
        weight,
        bias,
        options,
        DType::F16,
        f16::from_f32(0.0),
        gemm_f16,
        |a, b| f16::from_f32(a.to_f32() + b.to_f32()),
    )
}

// ============================================================================
// Bias addition
// ============================================================================

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

#[cfg(feature = "gemm")]
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

#[cfg(not(feature = "gemm"))]
fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(feature = "gemm")]
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

#[cfg(not(feature = "gemm"))]
fn gemm_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(feature = "gemm")]
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

#[cfg(not(feature = "gemm"))]
fn gemm_f16(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Vec<f16> {
    let mut c = vec![f16::from_f32(0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
            }
            c[i * n + j] = f16::from_f32(sum);
        }
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

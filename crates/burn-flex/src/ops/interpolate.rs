//! Interpolation operations for image resizing.
//!
//! Supported modes:
//! - Nearest: Floor-based coordinate mapping (fastest)
//! - Bilinear: 4-point weighted average (good quality/speed tradeoff)
//! - Bicubic: 16-point cubic convolution (highest quality)
//!
//! Optimizations:
//! - Rayon parallelism over (batch, channel) pairs
//! - Precomputed coordinate mappings where beneficial
//!
//! Supported dtypes: f32, f64, f16 (native), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{Bytes, Shape, bf16, f16};
use num_traits::Float;

use crate::{FlexTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates an interpolation forward typed dispatcher.
macro_rules! interpolate_typed {
    ($fn_name:ident, $impl_fn:ident, $T:ty) => {
        pub fn $fn_name(x: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor {
            $impl_fn::<$T>(x, output_size, align_corners)
        }
    };
}

/// Generates an interpolation bf16 forward wrapper via f32 conversion.
macro_rules! interpolate_bf16 {
    ($bf16_fn:ident, $f32_fn:ident) => {
        pub fn $bf16_fn(x: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor {
            let x_f32 = convert_bf16_to_f32(&x);
            let result_f32 = $f32_fn(x_f32, output_size, align_corners);
            convert_f32_to_bf16(&result_f32)
        }
    };
}

/// Generates an interpolation backward typed dispatcher.
macro_rules! interpolate_backward_typed {
    ($fn_name:ident, $impl_fn:ident, $T:ty) => {
        pub fn $fn_name(x: FlexTensor, grad: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor {
            $impl_fn::<$T>(x, grad, output_size, align_corners)
        }
    };
}

/// Generates an interpolation bf16 backward wrapper via f32 conversion.
macro_rules! interpolate_backward_bf16 {
    ($bf16_fn:ident, $f32_fn:ident) => {
        pub fn $bf16_fn(x: FlexTensor, grad: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor {
            let x_f32 = convert_bf16_to_f32(&x);
            let grad_f32 = convert_bf16_to_f32(&grad);
            let result_f32 = $f32_fn(x_f32, grad_f32, output_size, align_corners);
            convert_f32_to_bf16(&result_f32)
        }
    };
}

// ============================================================================
// Public API - dtype dispatch
// ============================================================================

interpolate_typed!(interpolate_nearest_f32, interpolate_nearest_impl, f32);
interpolate_typed!(interpolate_nearest_f64, interpolate_nearest_impl, f64);
interpolate_typed!(interpolate_nearest_f16, interpolate_nearest_impl, f16);
interpolate_bf16!(interpolate_nearest_bf16, interpolate_nearest_f32);

interpolate_typed!(interpolate_bilinear_f32, interpolate_bilinear_impl, f32);
interpolate_typed!(interpolate_bilinear_f64, interpolate_bilinear_impl, f64);
interpolate_typed!(interpolate_bilinear_f16, interpolate_bilinear_impl, f16);
interpolate_bf16!(interpolate_bilinear_bf16, interpolate_bilinear_f32);

interpolate_typed!(interpolate_bicubic_f32, interpolate_bicubic_impl, f32);
interpolate_typed!(interpolate_bicubic_f64, interpolate_bicubic_impl, f64);
interpolate_typed!(interpolate_bicubic_f16, interpolate_bicubic_impl, f16);
interpolate_bf16!(interpolate_bicubic_bf16, interpolate_bicubic_f32);

// ============================================================================
// Backward pass - dtype dispatch
// ============================================================================

interpolate_backward_typed!(
    interpolate_nearest_backward_f32,
    interpolate_nearest_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_nearest_backward_f64,
    interpolate_nearest_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_nearest_backward_f16,
    interpolate_nearest_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_nearest_backward_bf16,
    interpolate_nearest_backward_f32
);

interpolate_backward_typed!(
    interpolate_bilinear_backward_f32,
    interpolate_bilinear_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_bilinear_backward_f64,
    interpolate_bilinear_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_bilinear_backward_f16,
    interpolate_bilinear_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_bilinear_backward_bf16,
    interpolate_bilinear_backward_f32
);

interpolate_backward_typed!(
    interpolate_bicubic_backward_f32,
    interpolate_bicubic_backward_impl,
    f32
);
interpolate_backward_typed!(
    interpolate_bicubic_backward_f64,
    interpolate_bicubic_backward_impl,
    f64
);
interpolate_backward_typed!(
    interpolate_bicubic_backward_f16,
    interpolate_bicubic_backward_impl,
    f16
);
interpolate_backward_bf16!(
    interpolate_bicubic_backward_bf16,
    interpolate_bicubic_backward_f32
);

// ============================================================================
// Generic implementations with rayon parallelism
// ============================================================================

/// Compute coordinate mapping parameters.
///
/// align_corners=true:  ratio = (in_size - 1) / (out_size - 1), coord = out * ratio
/// align_corners=false: ratio = in_size / out_size, coord = (out + 0.5) * ratio - 0.5
fn coord_ratio(in_size: usize, out_size: usize, align_corners: bool) -> f64 {
    if align_corners {
        (in_size as f64 - 1.0) / (out_size.max(1) - 1).max(1) as f64
    } else {
        in_size as f64 / out_size as f64
    }
}

/// Map an output coordinate to input coordinate.
#[inline]
fn map_coord(out_coord: usize, ratio: f64, align_corners: bool) -> f64 {
    if align_corners {
        out_coord as f64 * ratio
    } else {
        (out_coord as f64 + 0.5) * ratio - 0.5
    }
}

/// Nearest neighbor interpolation.
/// Maps output coordinates to input using floor(ratio * out_coord).
fn interpolate_nearest_impl<T>(x: FlexTensor, output_size: [usize; 2], _align_corners: bool) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = in_height as f64 / out_height as f64;
    let x_ratio = in_width as f64 / out_width as f64;

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch).into_par_iter().for_each(|b| {
                (0..channels).into_par_iter().for_each(|c| {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for oh in 0..out_height {
                        let ih = ((oh as f64 * y_ratio).floor() as usize).min(in_height - 1);
                        for ow in 0..out_width {
                            let iw = ((ow as f64 * x_ratio).floor() as usize).min(in_width - 1);
                            let out_idx = out_base + oh * out_width + ow;
                            let val = input[in_base + ih * in_width + iw];
                            unsafe {
                                out_ptr.write(out_idx, val);
                            }
                        }
                    }
                });
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for oh in 0..out_height {
                        let ih = ((oh as f64 * y_ratio).floor() as usize).min(in_height - 1);
                        for ow in 0..out_width {
                            let iw = ((ow as f64 * x_ratio).floor() as usize).min(in_width - 1);
                            output[out_base + oh * out_width + ow] =
                                input[in_base + ih * in_width + iw];
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

/// Bilinear interpolation.
/// Uses 4-point weighted average based on distance to neighbors.
fn interpolate_bilinear_impl<T>(x: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch).into_par_iter().for_each(|b| {
                (0..channels).into_par_iter().for_each(|c| {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for oh in 0..out_height {
                        let y_in = map_coord(oh, y_ratio, align_corners);
                        let y_low = (y_in.floor().max(0.0)) as usize;
                        let y_high = (y_low + 1).min(in_height - 1);
                        let y_weight = T::from((y_in - y_low as f64).max(0.0)).unwrap();

                        for ow in 0..out_width {
                            let x_in = map_coord(ow, x_ratio, align_corners);
                            let x_low = (x_in.floor().max(0.0)) as usize;
                            let x_high = (x_low + 1).min(in_width - 1);
                            let x_weight = T::from((x_in - x_low as f64).max(0.0)).unwrap();

                            let p_a = input[in_base + y_low * in_width + x_low];
                            let p_b = input[in_base + y_low * in_width + x_high];
                            let p_c = input[in_base + y_high * in_width + x_low];
                            let p_d = input[in_base + y_high * in_width + x_high];

                            let one = T::one();
                            let result = p_a * (one - x_weight) * (one - y_weight)
                                + p_b * x_weight * (one - y_weight)
                                + p_c * (one - x_weight) * y_weight
                                + p_d * x_weight * y_weight;

                            let out_idx = out_base + oh * out_width + ow;
                            unsafe {
                                out_ptr.write(out_idx, result);
                            }
                        }
                    }
                });
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for oh in 0..out_height {
                        let y_in = map_coord(oh, y_ratio, align_corners);
                        let y_low = (y_in.floor().max(0.0)) as usize;
                        let y_high = (y_low + 1).min(in_height - 1);
                        let y_weight = T::from((y_in - y_low as f64).max(0.0)).unwrap();

                        for ow in 0..out_width {
                            let x_in = map_coord(ow, x_ratio, align_corners);
                            let x_low = (x_in.floor().max(0.0)) as usize;
                            let x_high = (x_low + 1).min(in_width - 1);
                            let x_weight = T::from((x_in - x_low as f64).max(0.0)).unwrap();

                            let p_a = input[in_base + y_low * in_width + x_low];
                            let p_b = input[in_base + y_low * in_width + x_high];
                            let p_c = input[in_base + y_high * in_width + x_low];
                            let p_d = input[in_base + y_high * in_width + x_high];

                            let one = T::one();
                            let result = p_a * (one - x_weight) * (one - y_weight)
                                + p_b * x_weight * (one - y_weight)
                                + p_c * (one - x_weight) * y_weight
                                + p_d * x_weight * y_weight;

                            output[out_base + oh * out_width + ow] = result;
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

/// Bicubic interpolation using cubic convolution.
fn interpolate_bicubic_impl<T>(x: FlexTensor, output_size: [usize; 2], align_corners: bool) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod + Send + Sync,
{
    let x = x.to_contiguous();
    let input = x.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let out_numel = batch * channels * out_height * out_width;
    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let a = -0.75_f64;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![T::zero(); out_numel];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());
            let num_bc_pairs = batch * channels;

            // Adaptive parallelization: if few (batch, channel) pairs, parallelize rows too
            if num_bc_pairs < 8 {
                // Fine-grained: parallelize over (batch, channel, row) for better CPU utilization
                let total_rows = batch * channels * out_height;
                (0..total_rows).into_par_iter().for_each(|id| {
                    let b = id / (channels * out_height);
                    let remainder = id % (channels * out_height);
                    let c = remainder / out_height;
                    let oh = remainder % out_height;

                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    let y_in = map_coord(oh, y_ratio, align_corners);
                    let y0 = y_in.floor() as isize;

                    for ow in 0..out_width {
                        let x_in = map_coord(ow, x_ratio, align_corners);
                        let x0 = x_in.floor() as isize;

                        let mut sum = 0.0_f64;

                        for dy in -1..=2_isize {
                            let y = y0 + dy;
                            let y_clamped = y.clamp(0, in_height as isize - 1) as usize;
                            let ty = (y_in - y0 as f64) - dy as f64;
                            let wy = cubic_weight(ty, a);

                            for dx in -1..=2_isize {
                                let x = x0 + dx;
                                let x_clamped = x.clamp(0, in_width as isize - 1) as usize;
                                let tx = (x_in - x0 as f64) - dx as f64;
                                let wx = cubic_weight(tx, a);

                                let val = input[in_base + y_clamped * in_width + x_clamped];
                                let val_f64 =
                                    <T as num_traits::ToPrimitive>::to_f64(&val).unwrap_or(0.0);
                                sum += val_f64 * wx * wy;
                            }
                        }

                        let out_idx = out_base + oh * out_width + ow;
                        unsafe {
                            out_ptr.write(out_idx, T::from(sum).unwrap());
                        }
                    }
                });
            } else {
                // Coarse-grained: parallelize over (batch, channel) for cache efficiency
                (0..batch).into_par_iter().for_each(|b| {
                    (0..channels).into_par_iter().for_each(|c| {
                        let in_base = b * channels * in_hw + c * in_hw;
                        let out_base = b * channels * out_hw + c * out_hw;

                        for oh in 0..out_height {
                            let y_in = map_coord(oh, y_ratio, align_corners);
                            let y0 = y_in.floor() as isize;

                            for ow in 0..out_width {
                                let x_in = map_coord(ow, x_ratio, align_corners);
                                let x0 = x_in.floor() as isize;

                                let mut sum = 0.0_f64;

                                for dy in -1..=2_isize {
                                    let y = y0 + dy;
                                    let y_clamped = y.clamp(0, in_height as isize - 1) as usize;
                                    let ty = (y_in - y0 as f64) - dy as f64;
                                    let wy = cubic_weight(ty, a);

                                    for dx in -1..=2_isize {
                                        let x = x0 + dx;
                                        let x_clamped = x.clamp(0, in_width as isize - 1) as usize;
                                        let tx = (x_in - x0 as f64) - dx as f64;
                                        let wx = cubic_weight(tx, a);

                                        let val = input[in_base + y_clamped * in_width + x_clamped];
                                        let val_f64 = <T as num_traits::ToPrimitive>::to_f64(&val)
                                            .unwrap_or(0.0);
                                        sum += val_f64 * wx * wy;
                                    }
                                }

                                let out_idx = out_base + oh * out_width + ow;
                                unsafe {
                                    out_ptr.write(out_idx, T::from(sum).unwrap());
                                }
                            }
                        }
                    });
                });
            }
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![T::zero(); out_numel];

            for b in 0..batch {
                for c in 0..channels {
                    let in_base = b * channels * in_hw + c * in_hw;
                    let out_base = b * channels * out_hw + c * out_hw;

                    for oh in 0..out_height {
                        let y_in = map_coord(oh, y_ratio, align_corners);
                        let y0 = y_in.floor() as isize;

                        for ow in 0..out_width {
                            let x_in = map_coord(ow, x_ratio, align_corners);
                            let x0 = x_in.floor() as isize;

                            let mut sum = 0.0_f64;

                            for dy in -1..=2_isize {
                                let y = y0 + dy;
                                let y_clamped = y.clamp(0, in_height as isize - 1) as usize;
                                let ty = (y_in - y0 as f64) - dy as f64;
                                let wy = cubic_weight(ty, a);

                                for dx in -1..=2_isize {
                                    let x = x0 + dx;
                                    let x_clamped = x.clamp(0, in_width as isize - 1) as usize;
                                    let tx = (x_in - x0 as f64) - dx as f64;
                                    let wx = cubic_weight(tx, a);

                                    let val = input[in_base + y_clamped * in_width + x_clamped];
                                    let val_f64 =
                                        <T as num_traits::ToPrimitive>::to_f64(&val).unwrap_or(0.0);
                                    sum += val_f64 * wx * wy;
                                }
                            }

                            output[out_base + oh * out_width + ow] = T::from(sum).unwrap();
                        }
                    }
                }
            }
            output
        }
    };

    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::from(vec![batch, channels, out_height, out_width])),
        x.dtype(),
    )
}

/// Cubic convolution weight function.
/// Uses the Keys cubic interpolation kernel.
#[inline]
fn cubic_weight(t: f64, a: f64) -> f64 {
    let t_abs = t.abs();
    if t_abs < 1.0 {
        ((a + 2.0) * t_abs - (a + 3.0)) * t_abs * t_abs + 1.0
    } else if t_abs < 2.0 {
        ((a * t_abs - 5.0 * a) * t_abs + 8.0 * a) * t_abs - 4.0 * a
    } else {
        0.0
    }
}

// ============================================================================
// Backward implementations
// ============================================================================

/// Nearest neighbor backward: accumulates gradients at source positions.
fn interpolate_nearest_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2], _align_corners: bool,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = in_height as f64 / out_height as f64;
    let x_ratio = in_width as f64 / out_width as f64;

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    // Backward requires accumulation, so sequential for correctness
    for b in 0..batch {
        for c in 0..channels {
            let in_base = b * channels * in_hw + c * in_hw;
            let out_base = b * channels * out_hw + c * out_hw;

            for oh in 0..out_height {
                let ih = ((oh as f64 * y_ratio).floor() as usize).min(in_height - 1);
                for ow in 0..out_width {
                    let iw = ((ow as f64 * x_ratio).floor() as usize).min(in_width - 1);
                    input_grad[in_base + ih * in_width + iw] = input_grad
                        [in_base + ih * in_width + iw]
                        + grad_data[out_base + oh * out_width + ow];
                }
            }
        }
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bilinear backward: distributes gradients to 4 source positions weighted by bilinear coefficients.
fn interpolate_bilinear_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2], align_corners: bool,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = b * channels * in_hw + c * in_hw;
            let out_base = b * channels * out_hw + c * out_hw;

            for oh in 0..out_height {
                let y_in = map_coord(oh, y_ratio, align_corners);
                let y_low = (y_in.floor().max(0.0)) as usize;
                let y_high = (y_low + 1).min(in_height - 1);
                let y_weight = T::from((y_in - y_low as f64).max(0.0)).unwrap();

                for ow in 0..out_width {
                    let x_in = map_coord(ow, x_ratio, align_corners);
                    let x_low = (x_in.floor().max(0.0)) as usize;
                    let x_high = (x_low + 1).min(in_width - 1);
                    let x_weight = T::from((x_in - x_low as f64).max(0.0)).unwrap();

                    let grad_val = grad_data[out_base + oh * out_width + ow];
                    let one = T::one();

                    input_grad[in_base + y_low * in_width + x_low] = input_grad
                        [in_base + y_low * in_width + x_low]
                        + grad_val * (one - x_weight) * (one - y_weight);
                    input_grad[in_base + y_low * in_width + x_high] = input_grad
                        [in_base + y_low * in_width + x_high]
                        + grad_val * x_weight * (one - y_weight);
                    input_grad[in_base + y_high * in_width + x_low] = input_grad
                        [in_base + y_high * in_width + x_low]
                        + grad_val * (one - x_weight) * y_weight;
                    input_grad[in_base + y_high * in_width + x_high] = input_grad
                        [in_base + y_high * in_width + x_high]
                        + grad_val * x_weight * y_weight;
                }
            }
        }
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

/// Bicubic backward: distributes gradients to 16 source positions weighted by cubic coefficients.
fn interpolate_bicubic_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    output_size: [usize; 2], align_corners: bool,
) -> FlexTensor
where
    T: Float + burn_backend::Element + bytemuck::Pod,
{
    let grad = grad.to_contiguous();
    let grad_data = grad.storage::<T>();
    let shape = x.layout().shape();

    let batch = shape[0];
    let channels = shape[1];
    let in_height = shape[2];
    let in_width = shape[3];
    assert!(
        in_height > 0 && in_width > 0,
        "interpolate: input spatial dimensions must be > 0"
    );
    let [out_height, out_width] = output_size;

    let y_ratio = coord_ratio(in_height, out_height, align_corners);
    let x_ratio = coord_ratio(in_width, out_width, align_corners);

    let in_numel = batch * channels * in_height * in_width;
    let mut input_grad = vec![T::zero(); in_numel];

    let in_hw = in_height * in_width;
    let out_hw = out_height * out_width;
    let a = -0.75_f64;

    for b in 0..batch {
        for c in 0..channels {
            let in_base = b * channels * in_hw + c * in_hw;
            let out_base = b * channels * out_hw + c * out_hw;

            for oh in 0..out_height {
                let y_in = map_coord(oh, y_ratio, align_corners);
                let y0 = y_in.floor() as isize;

                for ow in 0..out_width {
                    let x_in = map_coord(ow, x_ratio, align_corners);
                    let x0 = x_in.floor() as isize;

                    let grad_val = <T as num_traits::ToPrimitive>::to_f64(
                        &grad_data[out_base + oh * out_width + ow],
                    )
                    .unwrap_or(0.0);

                    for dy in -1..=2_isize {
                        let y = y0 + dy;
                        let y_idx = y.clamp(0, in_height as isize - 1) as usize;
                        let ty = (y_in - y0 as f64) - dy as f64;
                        let wy = cubic_weight(ty, a);

                        for dx in -1..=2_isize {
                            let x = x0 + dx;
                            let x_idx = x.clamp(0, in_width as isize - 1) as usize;
                            let tx = (x_in - x0 as f64) - dx as f64;
                            let wx = cubic_weight(tx, a);

                            let weight = wx * wy * grad_val;
                            input_grad[in_base + y_idx * in_width + x_idx] = input_grad
                                [in_base + y_idx * in_width + x_idx]
                                + T::from(weight).unwrap();
                        }
                    }
                }
            }
        }
    }

    FlexTensor::new(
        Bytes::from_elems(input_grad),
        Layout::contiguous(Shape::from(vec![batch, channels, in_height, in_width])),
        x.dtype(),
    )
}

// ============================================================================
// Dtype conversion helpers
// ============================================================================

fn convert_bf16_to_f32(x: &FlexTensor) -> FlexTensor {
    let x = x.clone().to_contiguous();
    let input = x.storage::<bf16>();
    let output: Vec<f32> = input.iter().map(|v| v.to_f32()).collect();
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(x.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(x: &FlexTensor) -> FlexTensor {
    let x = x.clone().to_contiguous();
    let input = x.storage::<f32>();
    let output: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v)).collect();
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(x.layout().shape().clone()),
        DType::BF16,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input_f32(batch: usize, channels: usize, height: usize, width: usize) -> FlexTensor {
        let numel = batch * channels * height * width;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![batch, channels, height, width])),
            DType::F32,
        )
    }

    #[test]
    fn test_nearest_upsample_2x() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let result = interpolate_nearest_f32(x, [4, 4]);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 16);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
    }

    #[test]
    fn test_bilinear_upsample_2x() {
        let data = vec![0.0f32, 1.0, 1.0, 0.0];
        let x = FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let result = interpolate_bilinear_f32(x, [4, 4]);
        let output = result.storage::<f32>();

        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
        assert!((output[12] - 1.0).abs() < 1e-5);
        assert!((output[15] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bicubic_basic() {
        let x = make_input_f32(1, 1, 4, 4);
        let result = interpolate_bicubic_f32(x, [8, 8]);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn test_downsample() {
        let x = make_input_f32(1, 1, 4, 4);
        let result = interpolate_nearest_f32(x, [2, 2]);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_nearest_backward() {
        let x = make_input_f32(1, 1, 2, 2);
        let grad = FlexTensor::new(
            Bytes::from_elems(vec![1.0f32; 16]),
            Layout::contiguous(Shape::from(vec![1, 1, 4, 4])),
            DType::F32,
        );

        let result = interpolate_nearest_backward_f32(x, grad, [4, 4]);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 4);
        assert!((output[0] - 4.0).abs() < 1e-5);
    }
}

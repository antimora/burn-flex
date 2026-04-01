//! Grid sampling operations for FlexTensor.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn_std::{Bytes, Shape};

use crate::{FlexTensor, Layout};

/// Grid sample 2D with bilinear interpolation.
///
/// Input tensor shape: [N, C, H_in, W_in]
/// Grid shape: [N, H_out, W_out, 2] (x, y normalized to [-1, 1])
/// Output shape: [N, C, H_out, W_out]
pub fn grid_sample_2d(
    tensor: FlexTensor,
    grid: FlexTensor,
    options: GridSampleOptions,
) -> FlexTensor {
    match options.mode {
        InterpolateMode::Bilinear | InterpolateMode::Nearest => {}
        other => panic!("grid_sample_2d: {:?} mode is not supported", other),
    }

    let tensor = tensor.to_contiguous();
    let grid = grid.to_contiguous();

    match tensor.dtype() {
        DType::F32 => grid_sample_2d_f32(tensor, grid, options),
        DType::F64 => grid_sample_2d_f64(tensor, grid, options),
        _ => panic!("grid_sample_2d: unsupported dtype {:?}", tensor.dtype()),
    }
}

macro_rules! grid_sample_2d_typed {
    ($name:ident, $elem:ty, $dtype:expr) => {
        fn $name(tensor: FlexTensor, grid: FlexTensor, options: GridSampleOptions) -> FlexTensor {
            let t_shape = tensor.layout().shape();
            let g_shape = grid.layout().shape();

            assert_eq!(t_shape.num_dims(), 4, "grid_sample_2d: input must be 4D");
            assert_eq!(g_shape.num_dims(), 4, "grid_sample_2d: grid must be 4D");
            assert_eq!(g_shape[3], 2, "grid_sample_2d: grid last dim must be 2");
            assert_eq!(
                t_shape[0], g_shape[0],
                "grid_sample_2d: batch size mismatch"
            );

            let batch_size = t_shape[0];
            let channels = t_shape[1];
            let h_in = t_shape[2];
            let w_in = t_shape[3];
            let h_out = g_shape[1];
            let w_out = g_shape[2];

            let tensor_data: &[$elem] = tensor.storage();
            let grid_data: &[$elem] = grid.storage();

            let out_shape = Shape::from(vec![batch_size, channels, h_out, w_out]);
            let out_len = batch_size * channels * h_out * w_out;
            let mut output: Vec<$elem> = vec![0.0; out_len];

            let align = options.align_corners;
            let pad_mode = options.padding_mode;

            let t_stride_n = channels * h_in * w_in;
            let t_stride_c = h_in * w_in;
            let t_stride_h = w_in;

            let g_stride_n = h_out * w_out * 2;
            let g_stride_h = w_out * 2;

            let o_stride_n = channels * h_out * w_out;
            let o_stride_c = h_out * w_out;
            let o_stride_h = w_out;

            for b in 0..batch_size {
                for y in 0..h_out {
                    for x in 0..w_out {
                        let g_idx = b * g_stride_n + y * g_stride_h + x * 2;
                        let sample_x = grid_data[g_idx] as f64;
                        let sample_y = grid_data[g_idx + 1] as f64;

                        let (px, py) = if align {
                            let px = (sample_x + 1.0) * ((w_in - 1) as f64) / 2.0;
                            let py = (sample_y + 1.0) * ((h_in - 1) as f64) / 2.0;
                            (px, py)
                        } else {
                            let px = (sample_x + 1.0) * (w_in as f64) / 2.0 - 0.5;
                            let py = (sample_y + 1.0) * (h_in as f64) / 2.0 - 0.5;
                            (px, py)
                        };

                        let (px, py) = apply_padding(px, py, w_in, h_in, pad_mode, align);

                        let read = |t_base: usize, xi: i64, yi: i64| -> f64 {
                            match pad_mode {
                                GridSamplePaddingMode::Zeros => {
                                    if xi >= 0 && xi < w_in as i64 && yi >= 0 && yi < h_in as i64 {
                                        tensor_data[t_base + yi as usize * t_stride_h + xi as usize]
                                            as f64
                                    } else {
                                        0.0
                                    }
                                }
                                GridSamplePaddingMode::Border
                                | GridSamplePaddingMode::Reflection => {
                                    let xi = xi.clamp(0, (w_in - 1) as i64) as usize;
                                    let yi = yi.clamp(0, (h_in - 1) as i64) as usize;
                                    tensor_data[t_base + yi * t_stride_h + xi] as f64
                                }
                            }
                        };

                        for c in 0..channels {
                            let t_base = b * t_stride_n + c * t_stride_c;
                            let o_idx = b * o_stride_n + c * o_stride_c + y * o_stride_h + x;

                            let val = if matches!(options.mode, InterpolateMode::Nearest) {
                                let xi = px.round() as i64;
                                let yi = py.round() as i64;
                                read(t_base, xi, yi)
                            } else {
                                // Bilinear
                                let x0 = px.floor() as i64;
                                let y0 = py.floor() as i64;
                                let x1 = x0 + 1;
                                let y1 = y0 + 1;

                                let x_frac = px - px.floor();
                                let y_frac = py - py.floor();

                                let w00 = (1.0 - x_frac) * (1.0 - y_frac);
                                let w01 = (1.0 - x_frac) * y_frac;
                                let w10 = x_frac * (1.0 - y_frac);
                                let w11 = x_frac * y_frac;

                                read(t_base, x0, y0) * w00
                                    + read(t_base, x0, y1) * w01
                                    + read(t_base, x1, y0) * w10
                                    + read(t_base, x1, y1) * w11
                            };

                            output[o_idx] = val as $elem;
                        }
                    }
                }
            }

            let bytes = Bytes::from_elems(output);
            FlexTensor::new(bytes, Layout::contiguous(out_shape), $dtype)
        }
    };
}

grid_sample_2d_typed!(grid_sample_2d_f32, f32, DType::F32);
grid_sample_2d_typed!(grid_sample_2d_f64, f64, DType::F64);

fn apply_padding(
    px: f64,
    py: f64,
    w: usize,
    h: usize,
    mode: GridSamplePaddingMode,
    align_corners: bool,
) -> (f64, f64) {
    if !px.is_finite() || !py.is_finite() {
        return match mode {
            GridSamplePaddingMode::Border => {
                let cx = ((w - 1) as f64 / 2.0).clamp(0.0, (w - 1) as f64);
                let cy = ((h - 1) as f64 / 2.0).clamp(0.0, (h - 1) as f64);
                (cx, cy)
            }
            _ => (px, py),
        };
    }

    match mode {
        GridSamplePaddingMode::Zeros => (px, py),
        GridSamplePaddingMode::Border => {
            let px = px.clamp(0.0, (w - 1) as f64);
            let py = py.clamp(0.0, (h - 1) as f64);
            (px, py)
        }
        GridSamplePaddingMode::Reflection => {
            let px = reflect_coordinate(px, w, align_corners);
            let py = reflect_coordinate(py, h, align_corners);
            (px, py)
        }
    }
}

fn reflect_coordinate(coord: f64, size: usize, align_corners: bool) -> f64 {
    let size_f = size as f64;
    let (min_val, max_val) = if align_corners {
        (0.0, size_f - 1.0)
    } else {
        (-0.5, size_f - 0.5)
    };

    let span = max_val - min_val;
    if span <= 0.0 {
        return min_val;
    }

    let period = 2.0 * span;
    let x = (coord - min_val).abs();
    let x_mod = x - (x / period).floor() * period;
    span - (x_mod - span).abs() + min_val
}

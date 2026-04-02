//! Real FFT (rfft) via Cooley-Tukey radix-2 DIT.
//!
//! Computes the 1D discrete Fourier transform of real-valued input along a
//! given dimension, returning the non-redundant N/2+1 complex bins as separate
//! real and imaginary tensors.
//!
//! Optimizations:
//! - Compile-time twiddle tables via const fn Taylor-series sin/cos
//! - SIMD-vectorized butterfly passes via macerator (direct loads from twiddle table)
//! - Rayon parallelism across independent fibers for large tensors
//! - Static tables for N up to 65536; runtime fallback for larger sizes

use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Bytes, Shape};

use crate::{FlexTensor, Layout};

// ============================================================================
// Const-evaluable sin/cos via Taylor series
// ============================================================================

/// Pi constant for const evaluation.
const PI: f64 = core::f64::consts::PI;

/// Const-evaluable sine using Taylor series. 13 terms give ~1e-15 accuracy
/// for |x| <= pi. We reduce the input to [-pi, pi] first.
const fn const_sin(x: f64) -> f64 {
    // Range reduction to [-pi, pi]
    let mut x = x;
    // First reduce to [-2pi, 2pi] range
    x = x - ((x / (2.0 * PI)) as i64 as f64) * 2.0 * PI;
    if x > PI {
        x -= 2.0 * PI;
    } else if x < -PI {
        x += 2.0 * PI;
    }

    // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut i = 1u32;
    while i <= 12 {
        term *= -x2 / ((2 * i) as f64 * (2 * i + 1) as f64);
        sum += term;
        i += 1;
    }
    sum
}

/// Const-evaluable cosine: cos(x) = sin(x + pi/2).
const fn const_cos(x: f64) -> f64 {
    const_sin(x + PI / 2.0)
}

// ============================================================================
// Compile-time twiddle table generation
// ============================================================================

/// A flat twiddle factor table for all stages of an FFT of size N.
///
/// For an FFT of size N with log2(N) stages, the table stores twiddle factors
/// for each stage consecutively: stage 0 has 1 entry (half=1 for len=2),
/// stage 1 has 2 entries (half=2 for len=4), etc.
///
/// Total entries = 1 + 2 + 4 + ... + N/2 = N - 1.
///
/// Each entry stores (cos, sin) as f32 for direct SIMD consumption.
/// Layout: [re_0, re_1, ..., re_{N-2}, im_0, im_1, ..., im_{N-2}]
/// (split real/imaginary for better SIMD load patterns).
struct TwiddleTable<const M: usize> {
    /// Real parts (cosines), length M = N-1.
    re: [f32; M],
    /// Imaginary parts (sines), length M = N-1.
    im: [f32; M],
    /// Stage offsets: offset[s] = start index for stage s. Length = num_stages + 1.
    /// Max stages = 17 (for N=131072). We use a fixed-size array.
    offsets: [usize; 18],
    num_stages: usize,
}

/// Generate a compile-time twiddle table for FFT size N.
/// M must equal N - 1 (total twiddle entries across all stages).
const fn make_twiddle_table<const N: usize, const M: usize>() -> TwiddleTable<M> {
    let mut re = [0.0f32; M];
    let mut im = [0.0f32; M];
    let mut offsets = [0usize; 18];

    let num_stages = N.trailing_zeros() as usize;
    let mut pos = 0usize;
    let mut len = 2usize;
    let mut stage = 0usize;

    while stage < num_stages {
        offsets[stage] = pos;
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;

        let mut k = 0usize;
        while k < half {
            let angle = angle_step * k as f64;
            re[pos] = const_cos(angle) as f32;
            im[pos] = const_sin(angle) as f32;
            pos += 1;
            k += 1;
        }

        len <<= 1;
        stage += 1;
    }
    offsets[num_stages] = pos; // sentinel

    TwiddleTable {
        re,
        im,
        offsets,
        num_stages,
    }
}

/// Macro to define a static twiddle table for a given FFT size.
macro_rules! def_twiddle {
    ($name:ident, $n:expr) => {
        static $name: TwiddleTable<{ $n - 1 }> = make_twiddle_table::<$n, { $n - 1 }>();
    };
}

// Static tables for common power-of-2 sizes
def_twiddle!(TW_1, 1);
def_twiddle!(TW_2, 2);
def_twiddle!(TW_4, 4);
def_twiddle!(TW_8, 8);
def_twiddle!(TW_16, 16);
def_twiddle!(TW_32, 32);
def_twiddle!(TW_64, 64);
def_twiddle!(TW_128, 128);
def_twiddle!(TW_256, 256);
def_twiddle!(TW_512, 512);
def_twiddle!(TW_1024, 1024);
def_twiddle!(TW_2048, 2048);
def_twiddle!(TW_4096, 4096);
def_twiddle!(TW_8192, 8192);
def_twiddle!(TW_16384, 16384);
def_twiddle!(TW_32768, 32768);
def_twiddle!(TW_65536, 65536);

/// Lookup or compute twiddle factors for FFT of size n.
/// Returns (re_slice, im_slice, offsets_slice, num_stages).
/// For common sizes, returns references to static tables (zero allocation).
/// For uncommon sizes, computes at runtime.
fn get_twiddles(n: usize) -> TwiddleRef {
    macro_rules! match_static {
        ($($size:expr => $table:ident),+ $(,)?) => {
            match n {
                $($size => TwiddleRef::Static {
                    re: &$table.re,
                    im: &$table.im,
                    offsets: &$table.offsets[..$table.num_stages + 1],
                },)+
                _ => {
                    // Runtime fallback for sizes > 65536 or non-standard
                    let (re, im, offsets) = precompute_twiddles_runtime(n);
                    TwiddleRef::Owned { re, im, offsets }
                }
            }
        };
    }

    match_static!(
        1 => TW_1,
        2 => TW_2,
        4 => TW_4,
        8 => TW_8,
        16 => TW_16,
        32 => TW_32,
        64 => TW_64,
        128 => TW_128,
        256 => TW_256,
        512 => TW_512,
        1024 => TW_1024,
        2048 => TW_2048,
        4096 => TW_4096,
        8192 => TW_8192,
        16384 => TW_16384,
        32768 => TW_32768,
        65536 => TW_65536,
    )
}

enum TwiddleRef {
    Static {
        re: &'static [f32],
        im: &'static [f32],
        offsets: &'static [usize],
    },
    Owned {
        re: Vec<f32>,
        im: Vec<f32>,
        offsets: Vec<usize>,
    },
}

impl TwiddleRef {
    fn re(&self) -> &[f32] {
        match self {
            Self::Static { re, .. } => re,
            Self::Owned { re, .. } => re,
        }
    }
    fn im(&self) -> &[f32] {
        match self {
            Self::Static { im, .. } => im,
            Self::Owned { im, .. } => im,
        }
    }
    fn offsets(&self) -> &[usize] {
        match self {
            Self::Static { offsets, .. } => offsets,
            Self::Owned { offsets, .. } => offsets,
        }
    }
}

/// Runtime fallback twiddle computation for sizes not in the static table.
fn precompute_twiddles_runtime(n: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let num_stages = n.trailing_zeros() as usize;
    let total = n - 1;
    let mut re = Vec::with_capacity(total);
    let mut im = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(num_stages + 1);

    let mut len = 2;
    for _ in 0..num_stages {
        offsets.push(re.len());
        let half = len / 2;
        let angle_step = -2.0 * core::f64::consts::PI / len as f64;
        for k in 0..half {
            let angle = angle_step * k as f64;
            re.push(angle.cos() as f32);
            im.push(angle.sin() as f32);
        }
        len <<= 1;
    }
    offsets.push(re.len());

    (re, im, offsets)
}

// ============================================================================
// Bit-reversal permutation
// ============================================================================

#[inline]
fn bit_reverse_permute(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

// ============================================================================
// Butterfly passes
// ============================================================================

/// Scalar butterfly passes using f32 twiddle table.
#[cfg(not(feature = "simd"))]
fn fft_butterfly_scalar(
    re: &mut [f32],
    im: &mut [f32],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
) {
    let mut len = 2;
    for stage in 0..offsets.len() - 1 {
        let half = len / 2;
        let tw_off = offsets[stage];

        let mut start = 0;
        while start < n {
            for k in 0..half {
                let wr = tw_re[tw_off + k];
                let wi = tw_im[tw_off + k];
                let even = start + k;
                let odd = even + half;

                let t_re = wr * re[odd] - wi * im[odd];
                let t_im = wr * im[odd] + wi * re[odd];

                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] = re[even] + t_re;
                im[even] = im[even] + t_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

#[cfg(feature = "simd")]
mod simd_fft {
    use macerator::{Simd, vload_unaligned, vstore_unaligned};

    /// SIMD butterfly passes. Twiddle factors are f32 slices that can be loaded
    /// directly into SIMD registers (consecutive entries = consecutive lanes).
    #[macerator::with_simd]
    pub fn butterfly_simd<S: Simd>(
        re: &mut [f32],
        im: &mut [f32],
        n: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        offsets: &[usize],
    ) {
        let lanes = S::lanes32();
        let num_stages = offsets.len() - 1;

        let mut len = 2;
        for stage in 0..num_stages {
            let half = len / 2;
            let tw_off = offsets[stage];

            if half >= lanes {
                let mut start = 0;
                while start < n {
                    let mut k = 0;
                    while k + lanes <= half {
                        unsafe {
                            // Direct SIMD load from twiddle table
                            let wr = vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off + k));
                            let wi = vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off + k));

                            let even_idx = start + k;
                            let odd_idx = even_idx + half;

                            let re_even = vload_unaligned::<S, f32>(re.as_ptr().add(even_idx));
                            let im_even = vload_unaligned::<S, f32>(im.as_ptr().add(even_idx));
                            let re_odd = vload_unaligned::<S, f32>(re.as_ptr().add(odd_idx));
                            let im_odd = vload_unaligned::<S, f32>(im.as_ptr().add(odd_idx));

                            let t_re = wr * re_odd - wi * im_odd;
                            let t_im = wr * im_odd + wi * re_odd;

                            vstore_unaligned::<S, f32>(
                                re.as_mut_ptr().add(even_idx),
                                re_even + t_re,
                            );
                            vstore_unaligned::<S, f32>(
                                im.as_mut_ptr().add(even_idx),
                                im_even + t_im,
                            );
                            vstore_unaligned::<S, f32>(
                                re.as_mut_ptr().add(odd_idx),
                                re_even - t_re,
                            );
                            vstore_unaligned::<S, f32>(
                                im.as_mut_ptr().add(odd_idx),
                                im_even - t_im,
                            );
                        }
                        k += lanes;
                    }

                    // Scalar tail
                    while k < half {
                        let wr = tw_re[tw_off + k];
                        let wi = tw_im[tw_off + k];
                        let even = start + k;
                        let odd = even + half;

                        let t_re = wr * re[odd] - wi * im[odd];
                        let t_im = wr * im[odd] + wi * re[odd];

                        re[odd] = re[even] - t_re;
                        im[odd] = im[even] - t_im;
                        re[even] = re[even] + t_re;
                        im[even] = im[even] + t_im;
                        k += 1;
                    }

                    start += len;
                }
            } else {
                // Small stage: scalar
                let mut start = 0;
                while start < n {
                    for k in 0..half {
                        let wr = tw_re[tw_off + k];
                        let wi = tw_im[tw_off + k];
                        let even = start + k;
                        let odd = even + half;

                        let t_re = wr * re[odd] - wi * im[odd];
                        let t_im = wr * im[odd] + wi * re[odd];

                        re[odd] = re[even] - t_re;
                        im[odd] = im[even] - t_im;
                        re[even] = re[even] + t_re;
                        im[even] = im[even] + t_im;
                    }
                    start += len;
                }
            }
            len <<= 1;
        }
    }
}

// ============================================================================
// Core FFT dispatch
// ============================================================================

#[inline]
fn fft_f32_with_twiddles(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    bit_reverse_permute(re, im, n);

    #[cfg(feature = "simd")]
    {
        simd_fft::butterfly_simd(re, im, n, tw.re(), tw.im(), tw.offsets());
    }
    #[cfg(not(feature = "simd"))]
    {
        fft_butterfly_scalar(re, im, n, tw.re(), tw.im(), tw.offsets());
    }
}

// ============================================================================
// Tensor-level rfft
// ============================================================================

/// Contiguous strides for a shape.
fn contiguous_strides(shape: &Shape) -> Vec<usize> {
    let ndims = shape.num_dims();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute flat offset for fiber_idx, with position along `dim` = 0.
fn fiber_base_offset(
    mut fiber_idx: usize,
    shape: &Shape,
    strides: &[usize],
    dim: usize,
    ndims: usize,
) -> usize {
    let mut offset = 0;
    for d in (0..ndims).rev() {
        if d == dim {
            continue;
        }
        let size = shape[d];
        offset += (fiber_idx % size) * strides[d];
        fiber_idx /= size;
    }
    offset
}

pub fn rfft_f32(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();
    let n = shape[dim];
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / n;

    let data: &[f32] = tensor.storage();
    let in_strides = contiguous_strides(&shape);
    let out_strides = contiguous_strides(&out_shape);

    // Lookup compile-time twiddle table (zero allocation for N <= 65536)
    let tw = get_twiddles(n);

    let mut re_out = vec![0.0f32; total_out];
    let mut im_out = vec![0.0f32; total_out];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = fiber_base_offset(fiber_idx, &shape, &in_strides, dim, ndims);
                let in_stride = in_strides[dim];

                let mut re_buf = vec![0.0f32; n];
                let mut im_buf = vec![0.0f32; n];
                for i in 0..n {
                    re_buf[i] = data[base_offset + i * in_stride];
                }

                fft_f32_with_twiddles(&mut re_buf, &mut im_buf, n, &tw);
                (fiber_idx, re_buf, im_buf)
            })
            .collect();

        for (fiber_idx, re_buf, im_buf) in fiber_results {
            let out_base = fiber_base_offset(fiber_idx, &out_shape, &out_strides, dim, ndims);
            let out_stride = out_strides[dim];
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = re_buf[k];
                im_out[out_base + k * out_stride] = im_buf[k];
            }
        }
    } else {
        rfft_fibers_sequential(
            data, &mut re_out, &mut im_out, &shape, &out_shape, &in_strides, &out_strides,
            dim, ndims, n, out_len, num_fibers, &tw,
        );
        return make_tensors(re_out, im_out, out_shape);
    }

    #[cfg(not(feature = "rayon"))]
    rfft_fibers_sequential(
        data, &mut re_out, &mut im_out, &shape, &out_shape, &in_strides, &out_strides,
        dim, ndims, n, out_len, num_fibers, &tw,
    );

    make_tensors(re_out, im_out, out_shape)
}

#[allow(clippy::too_many_arguments)]
fn rfft_fibers_sequential(
    data: &[f32],
    re_out: &mut [f32],
    im_out: &mut [f32],
    shape: &Shape,
    out_shape: &Shape,
    in_strides: &[usize],
    out_strides: &[usize],
    dim: usize,
    ndims: usize,
    n: usize,
    out_len: usize,
    num_fibers: usize,
    tw: &TwiddleRef,
) {
    let mut re_buf = vec![0.0f32; n];
    let mut im_buf = vec![0.0f32; n];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    for fiber_idx in 0..num_fibers {
        let base_offset = fiber_base_offset(fiber_idx, shape, in_strides, dim, ndims);
        let out_base = fiber_base_offset(fiber_idx, out_shape, out_strides, dim, ndims);

        for i in 0..n {
            re_buf[i] = data[base_offset + i * in_stride];
            im_buf[i] = 0.0;
        }

        fft_f32_with_twiddles(&mut re_buf, &mut im_buf, n, tw);

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = re_buf[k];
            im_out[out_base + k * out_stride] = im_buf[k];
        }
    }
}

fn make_tensors(re: Vec<f32>, im: Vec<f32>, shape: Shape) -> (FlexTensor, FlexTensor) {
    let re_tensor = FlexTensor::new(
        Bytes::from_elems(re),
        Layout::contiguous(shape.clone()),
        burn_backend::DType::F32,
    );
    let im_tensor = FlexTensor::new(
        Bytes::from_elems(im),
        Layout::contiguous(shape),
        burn_backend::DType::F32,
    );
    (re_tensor, im_tensor)
}

/// Rfft for f64 tensors.
pub fn rfft_f64(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let ndims = shape.num_dims();
    let n = shape[dim];
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / n;

    let data: &[f64] = tensor.storage();
    let in_strides = contiguous_strides(&shape);
    let out_strides = contiguous_strides(&out_shape);

    // For f64, we use the f32 twiddle table and widen to f64 in the inner loop.
    // The compile-time table is still f32 but the extra precision from f64 butterfly
    // arithmetic provides sufficient accuracy.
    let tw = get_twiddles(n);

    let mut re_out = vec![0.0f64; total_out];
    let mut im_out = vec![0.0f64; total_out];

    let mut re_buf = vec![0.0f64; n];
    let mut im_buf = vec![0.0f64; n];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    let tw_re = tw.re();
    let tw_im = tw.im();
    let offsets = tw.offsets();

    for fiber_idx in 0..num_fibers {
        let base_offset = fiber_base_offset(fiber_idx, &shape, &in_strides, dim, ndims);
        let out_base = fiber_base_offset(fiber_idx, &out_shape, &out_strides, dim, ndims);

        for i in 0..n {
            re_buf[i] = data[base_offset + i * in_stride];
            im_buf[i] = 0.0;
        }

        // Bit-reversal
        {
            let mut j = 0usize;
            for i in 1..n {
                let mut bit = n >> 1;
                while j & bit != 0 {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                if i < j {
                    re_buf.swap(i, j);
                    im_buf.swap(i, j);
                }
            }
        }

        // Butterfly passes
        let mut len = 2;
        for stage in 0..offsets.len() - 1 {
            let half = len / 2;
            let tw_off = offsets[stage];

            let mut start = 0;
            while start < n {
                for k in 0..half {
                    let wr = tw_re[tw_off + k] as f64;
                    let wi = tw_im[tw_off + k] as f64;
                    let even = start + k;
                    let odd = even + half;

                    let t_re = wr * re_buf[odd] - wi * im_buf[odd];
                    let t_im = wr * im_buf[odd] + wi * re_buf[odd];

                    re_buf[odd] = re_buf[even] - t_re;
                    im_buf[odd] = im_buf[even] - t_im;
                    re_buf[even] = re_buf[even] + t_re;
                    im_buf[even] = im_buf[even] + t_im;
                }
                start += len;
            }
            len <<= 1;
        }

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = re_buf[k];
            im_out[out_base + k * out_stride] = im_buf[k];
        }
    }

    let re_tensor = FlexTensor::new(
        Bytes::from_elems(re_out),
        Layout::contiguous(out_shape.clone()),
        burn_backend::DType::F64,
    );
    let im_tensor = FlexTensor::new(
        Bytes::from_elems(im_out),
        Layout::contiguous(out_shape),
        burn_backend::DType::F64,
    );
    (re_tensor, im_tensor)
}

pub fn rfft_f16(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    use burn_std::f16;
    let tensor = super::module::cast_to_f32(tensor, f16::to_f32);
    let (re, im) = rfft_f32(tensor, dim);
    (
        super::module::cast_from_f32(re, f16::from_f32),
        super::module::cast_from_f32(im, f16::from_f32),
    )
}

pub fn rfft_bf16(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    use burn_std::bf16;
    let tensor = super::module::cast_to_f32(tensor, bf16::to_f32);
    let (re, im) = rfft_f32(tensor, dim);
    (
        super::module::cast_from_f32(re, bf16::from_f32),
        super::module::cast_from_f32(im, bf16::from_f32),
    )
}

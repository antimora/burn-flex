//! Real FFT (rfft) via Cooley-Tukey with aggressive optimization.
//!
//! Key optimizations:
//! - Real FFT via complex packing: pack N real values as N/2 complex,
//!   do a half-size complex FFT, then unpack using Hermitian symmetry (~2x)
//! - Compile-time twiddle tables via const fn Taylor-series sin/cos
//! - Unrolled small complex FFT kernels for N=2, 4, 8
//! - Mixed radix-4/radix-2 butterfly stages (halves passes over data)
//! - SIMD-vectorized butterflies via macerator
//! - Rayon parallelism across independent fibers

use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Bytes, Shape};

use super::sort::slice_base_offset;
use crate::layout::contiguous_strides_usize;
use crate::{FlexTensor, Layout};

// ============================================================================
// Const-evaluable sin/cos via Taylor series (13 terms, ~13 digit accuracy)
// ============================================================================

const PI: f64 = core::f64::consts::PI;

const fn const_sin(x: f64) -> f64 {
    let mut x = x;
    x = x - ((x / (2.0 * PI)) as i64 as f64) * 2.0 * PI;
    if x > PI {
        x -= 2.0 * PI;
    } else if x < -PI {
        x += 2.0 * PI;
    }
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

const fn const_cos(x: f64) -> f64 {
    const_sin(x + PI / 2.0)
}

// ============================================================================
// Compile-time twiddle table
// ============================================================================

struct TwiddleTable<const M: usize> {
    re: [f32; M],
    im: [f32; M],
    offsets: [usize; 18],
    num_stages: usize,
}

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
    offsets[num_stages] = pos;
    TwiddleTable {
        re,
        im,
        offsets,
        num_stages,
    }
}

macro_rules! def_twiddle {
    ($name:ident, $n:expr) => {
        static $name: TwiddleTable<{ $n - 1 }> = make_twiddle_table::<$n, { $n - 1 }>();
    };
}

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

fn get_twiddles(n: usize) -> TwiddleRef {
    macro_rules! match_static {
        ($($size:expr => $table:ident),+ $(,)?) => {
            match n {
                0 | 1 => TwiddleRef::Static { re: &[], im: &[], offsets: &[0] },
                $($size => TwiddleRef::Static {
                    re: &$table.re, im: &$table.im,
                    offsets: &$table.offsets[..$table.num_stages + 1],
                },)+
                _ => {
                    let (re, im, offsets) = precompute_twiddles_runtime(n);
                    TwiddleRef::Owned { re, im, offsets }
                }
            }
        };
    }
    match_static!(
        2 => TW_2, 4 => TW_4, 8 => TW_8, 16 => TW_16,
        32 => TW_32, 64 => TW_64, 128 => TW_128, 256 => TW_256,
        512 => TW_512, 1024 => TW_1024, 2048 => TW_2048, 4096 => TW_4096,
        8192 => TW_8192, 16384 => TW_16384, 32768 => TW_32768, 65536 => TW_65536,
    )
}

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
// Unrolled small complex FFT kernels
// ============================================================================

/// Complex FFT of size 2: single butterfly, no twiddles.
#[inline(always)]
fn complex_fft_2(re: &mut [f32], im: &mut [f32]) {
    let (r0, r1) = (re[0], re[1]);
    let (i0, i1) = (im[0], im[1]);
    re[0] = r0 + r1;
    re[1] = r0 - r1;
    im[0] = i0 + i1;
    im[1] = i0 - i1;
}

/// Complex FFT of size 4: 2 stages, fully unrolled.
/// Twiddle for stage 1, k=1 is W_4^1 = -i.
#[inline(always)]
fn complex_fft_4(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal: swap indices 1 and 2
    re.swap(1, 2);
    im.swap(1, 2);

    // Stage 0: two size-2 butterflies
    let (r0, r1) = (re[0] + re[1], re[0] - re[1]);
    let (i0, i1) = (im[0] + im[1], im[0] - im[1]);
    let (r2, r3) = (re[2] + re[3], re[2] - re[3]);
    let (i2, i3) = (im[2] + im[3], im[2] - im[3]);

    // Stage 1: size-4 butterfly
    // k=0: W=1, butterfly (0,2)
    re[0] = r0 + r2;
    im[0] = i0 + i2;
    re[2] = r0 - r2;
    im[2] = i0 - i2;
    // k=1: W=-i, butterfly (1,3): -i*(r3+i*i3) = (i3, -r3)
    re[1] = r1 + i3;
    im[1] = i1 - r3;
    re[3] = r1 - i3;
    im[3] = i1 + r3;
}

/// Complex FFT of size 8: 3 stages, fully unrolled.
#[inline(always)]
fn complex_fft_8(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal for n=8: [0,4,2,6,1,5,3,7]
    re.swap(1, 4);
    im.swap(1, 4);
    re.swap(3, 6);
    im.swap(3, 6);

    // Stage 0: four size-2 butterflies
    macro_rules! butterfly2 {
        ($a:expr, $b:expr) => {
            let (ra, rb) = (re[$a] + re[$b], re[$a] - re[$b]);
            let (ia, ib) = (im[$a] + im[$b], im[$a] - im[$b]);
            re[$a] = ra;
            re[$b] = rb;
            im[$a] = ia;
            im[$b] = ib;
        };
    }
    butterfly2!(0, 1);
    butterfly2!(2, 3);
    butterfly2!(4, 5);
    butterfly2!(6, 7);

    // Stage 1: two size-4 butterflies
    // Group [0,1,2,3]: k=0 W=1, k=1 W=-i
    {
        let (r0, r2) = (re[0] + re[2], re[0] - re[2]);
        let (i0, i2) = (im[0] + im[2], im[0] - im[2]);
        re[0] = r0;
        im[0] = i0;
        re[2] = r2;
        im[2] = i2;
        // k=1: W=-i → (im[3], -re[3])
        let (t_re, t_im) = (im[3], -re[3]);
        let (r1a, r1b) = (re[1] + t_re, re[1] - t_re);
        let (i1a, i1b) = (im[1] + t_im, im[1] - t_im);
        re[1] = r1a;
        re[3] = r1b;
        im[1] = i1a;
        im[3] = i1b;
    }
    // Group [4,5,6,7]: same pattern
    {
        let (r4, r6) = (re[4] + re[6], re[4] - re[6]);
        let (i4, i6) = (im[4] + im[6], im[4] - im[6]);
        re[4] = r4;
        im[4] = i4;
        re[6] = r6;
        im[6] = i6;
        let (t_re, t_im) = (im[7], -re[7]);
        let (r5a, r5b) = (re[5] + t_re, re[5] - t_re);
        let (i5a, i5b) = (im[5] + t_im, im[5] - t_im);
        re[5] = r5a;
        re[7] = r5b;
        im[5] = i5a;
        im[7] = i5b;
    }

    // Stage 2: one size-8 butterfly
    // k=0: W=1
    {
        let (a, b) = (re[0] + re[4], re[0] - re[4]);
        let (c, d) = (im[0] + im[4], im[0] - im[4]);
        re[0] = a;
        re[4] = b;
        im[0] = c;
        im[4] = d;
    }
    // k=1: W_8^1 = (sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2; // 0.7071...
        let t_re = W * re[5] - (-W) * im[5]; // W*re + W*im
        let t_im = W * im[5] + (-W) * re[5]; // W*im - W*re
        re[5] = re[1] - t_re;
        im[5] = im[1] - t_im;
        re[1] += t_re;
        im[1] += t_im;
    }
    // k=2: W_8^2 = -i
    {
        let (t_re, t_im) = (im[6], -re[6]);
        re[6] = re[2] - t_re;
        im[6] = im[2] - t_im;
        re[2] += t_re;
        im[2] += t_im;
    }
    // k=3: W_8^3 = (-sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2;
        let t_re = -W * re[7] - (-W) * im[7]; // -W*re + W*im
        let t_im = -W * im[7] + (-W) * re[7]; // -W*im - W*re
        re[7] = re[3] - t_re;
        im[7] = im[3] - t_im;
        re[3] += t_re;
        im[3] += t_im;
    }
}

// ============================================================================
// General complex FFT: radix-4/radix-2 with SIMD
// ============================================================================

/// Complex FFT of size n (power of 2) using precomputed twiddles.
#[inline]
fn complex_fft(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    match n {
        0 | 1 => return,
        2 => {
            complex_fft_2(re, im);
            return;
        }
        4 => {
            complex_fft_4(re, im);
            return;
        }
        8 => {
            complex_fft_8(re, im);
            return;
        }
        _ => {}
    }

    bit_reverse_permute(re, im, n);

    let offsets = tw.offsets();
    let tw_re = tw.re();
    let tw_im = tw.im();
    let num_stages = offsets.len() - 1;

    // For odd number of stages, do one radix-2 pass first
    let start_stage = if num_stages % 2 == 1 {
        // Stage 0: radix-2, len=2, half=1, twiddle=1 for all groups
        let mut start = 0;
        while start < n {
            let (a, b) = (re[start] + re[start + 1], re[start] - re[start + 1]);
            let (c, d) = (im[start] + im[start + 1], im[start] - im[start + 1]);
            re[start] = a;
            re[start + 1] = b;
            im[start] = c;
            im[start + 1] = d;
            start += 2;
        }
        1
    } else {
        0
    };

    // Remaining stages: radix-4 (pairs of radix-2 stages)
    #[cfg(feature = "simd")]
    {
        simd_fft::radix4_simd(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
    #[cfg(not(feature = "simd"))]
    {
        radix4_scalar(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
}

/// Single radix-4 butterfly at position p0 with given quarter stride and twiddle offsets.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scalar_radix4_butterfly(
    re: &mut [f32],
    im: &mut [f32],
    p0: usize,
    quarter: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    tw_off_inner: usize,
    tw_off_outer: usize,
    k: usize,
) {
    let p1 = p0 + quarter;
    let p2 = p1 + quarter;
    let p3 = p2 + quarter;

    let w1_re = tw_re[tw_off_outer + k];
    let w1_im = tw_im[tw_off_outer + k];
    let w2_re = tw_re[tw_off_inner + k];
    let w2_im = tw_im[tw_off_inner + k];
    // W3 = W1 * W2
    let w3_re = w1_re * w2_re - w1_im * w2_im;
    let w3_im = w1_re * w2_im + w1_im * w2_re;

    let b_re = w1_re * re[p1] - w1_im * im[p1];
    let b_im = w1_re * im[p1] + w1_im * re[p1];
    let c_re = w2_re * re[p2] - w2_im * im[p2];
    let c_im = w2_re * im[p2] + w2_im * re[p2];
    let d_re = w3_re * re[p3] - w3_im * im[p3];
    let d_im = w3_re * im[p3] + w3_im * re[p3];

    let a_re = re[p0];
    let a_im = im[p0];

    // Radix-4 core: -i * (b - d) where -i*(x+iy) = (y, -x)
    let u0_re = a_re + c_re;
    let u0_im = a_im + c_im;
    let u1_re = a_re - c_re;
    let u1_im = a_im - c_im;
    let u2_re = b_re + d_re;
    let u2_im = b_im + d_im;
    let diff_re = b_re - d_re;
    let diff_im = b_im - d_im;

    re[p0] = u0_re + u2_re;
    im[p0] = u0_im + u2_im;
    re[p1] = u1_re + diff_im;
    im[p1] = u1_im - diff_re;
    re[p2] = u0_re - u2_re;
    im[p2] = u0_im - u2_im;
    re[p3] = u1_re - diff_im;
    im[p3] = u1_im + diff_re;
}

/// Scalar radix-4 butterfly stages. Processes two radix-2 stages at once.
#[cfg(not(feature = "simd"))]
#[allow(clippy::too_many_arguments)]
fn radix4_scalar(
    re: &mut [f32],
    im: &mut [f32],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
    start_stage: usize,
    num_stages: usize,
) {
    let mut stage = start_stage;
    while stage + 1 < num_stages {
        let quarter = 1 << stage;
        let group_size = quarter << 2;
        let tw_off_inner = offsets[stage];
        let tw_off_outer = offsets[stage + 1];

        let mut group_start = 0;
        while group_start < n {
            for k in 0..quarter {
                scalar_radix4_butterfly(
                    re,
                    im,
                    group_start + k,
                    quarter,
                    tw_re,
                    tw_im,
                    tw_off_inner,
                    tw_off_outer,
                    k,
                );
            }
            group_start += group_size;
        }
        stage += 2;
    }
}

#[cfg(feature = "simd")]
mod simd_fft {
    use macerator::{Simd, vload_unaligned, vstore_unaligned};

    #[macerator::with_simd]
    #[allow(clippy::too_many_arguments)]
    pub fn radix4_simd<S: Simd>(
        re: &mut [f32],
        im: &mut [f32],
        n: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        offsets: &[usize],
        start_stage: usize,
        num_stages: usize,
    ) {
        let lanes = S::lanes32();
        let mut stage = start_stage;

        while stage + 1 < num_stages {
            let quarter = 1 << stage;
            let group_size = quarter << 2;
            let tw_off_inner = offsets[stage];
            let tw_off_outer = offsets[stage + 1];

            if quarter >= lanes {
                // SIMD path
                let mut group_start = 0;
                while group_start < n {
                    let mut k = 0;
                    while k + lanes <= quarter {
                        unsafe {
                            // Load twiddle factors
                            let w1r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_outer + k));
                            let w1i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_outer + k));
                            let w2r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_inner + k));
                            let w2i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_inner + k));
                            // W3 = W1 * W2
                            let w3r = w1r * w2r - w1i * w2i;
                            let w3i = w1r * w2i + w1i * w2r;

                            let p0 = group_start + k;
                            let p1 = p0 + quarter;
                            let p2 = p1 + quarter;
                            let p3 = p2 + quarter;

                            let a_re = vload_unaligned::<S, f32>(re.as_ptr().add(p0));
                            let a_im = vload_unaligned::<S, f32>(im.as_ptr().add(p0));
                            let re_p1 = vload_unaligned::<S, f32>(re.as_ptr().add(p1));
                            let im_p1 = vload_unaligned::<S, f32>(im.as_ptr().add(p1));
                            let re_p2 = vload_unaligned::<S, f32>(re.as_ptr().add(p2));
                            let im_p2 = vload_unaligned::<S, f32>(im.as_ptr().add(p2));
                            let re_p3 = vload_unaligned::<S, f32>(re.as_ptr().add(p3));
                            let im_p3 = vload_unaligned::<S, f32>(im.as_ptr().add(p3));

                            // Apply twiddles
                            let b_re = w1r * re_p1 - w1i * im_p1;
                            let b_im = w1r * im_p1 + w1i * re_p1;
                            let c_re = w2r * re_p2 - w2i * im_p2;
                            let c_im = w2r * im_p2 + w2i * re_p2;
                            let d_re = w3r * re_p3 - w3i * im_p3;
                            let d_im = w3r * im_p3 + w3i * re_p3;

                            // Radix-4 core
                            let u0r = a_re + c_re;
                            let u0i = a_im + c_im;
                            let u1r = a_re - c_re;
                            let u1i = a_im - c_im;
                            let u2r = b_re + d_re;
                            let u2i = b_im + d_im;
                            let dr = b_re - d_re;
                            let di = b_im - d_im;
                            let u3r = di; // -i * (dr + di*i) = (di, -dr)
                            let u3i = dr; // negated below via subtraction

                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p0), u0r + u2r);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p0), u0i + u2i);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p1), u1r + u3r);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p1), u1i - u3i);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p2), u0r - u2r);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p2), u0i - u2i);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p3), u1r - u3r);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p3), u1i + u3i);
                        }
                        k += lanes;
                    }
                    while k < quarter {
                        super::scalar_radix4_butterfly(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                        k += 1;
                    }
                    group_start += group_size;
                }
            } else {
                // Small quarter: all scalar
                let mut group_start = 0;
                while group_start < n {
                    for k in 0..quarter {
                        super::scalar_radix4_butterfly(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                    }
                    group_start += group_size;
                }
            }
            stage += 2;
        }
    }
}

// ============================================================================
// Real FFT unpacking
// ============================================================================

/// Unpack N/2-point complex FFT result into N/2+1 real FFT bins.
///
/// Given Z = FFT(pack(x)), recovers X = FFT(x) using:
///   Xe[k] = (Z[k] + conj(Z[N/2-k])) / 2
///   Xo[k] = -i * (Z[k] - conj(Z[N/2-k])) / 2
///   X[k]  = Xe[k] + W_N^k * Xo[k]
fn unpack_rfft(
    z_re: &[f32],
    z_im: &[f32],
    half: usize,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    // k=0: X[0] = Z_re[0] + Z_im[0] (real)
    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;

    // k=N/2: X[N/2] = Z_re[0] - Z_im[0] (real)
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    // k=1..half-1
    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        // Xe = (Z[k] + conj(Z[j])) / 2
        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;

        // Xo = -i * (Z[k] - conj(Z[j])) / 2
        // diff = Z[k] - conj(Z[j]) = (zk_re - zj_re, zk_im + zj_im)
        // -i * diff = (diff_im, -diff_re)
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        // X[k] = Xe + W * Xo
        let wr = unpack_tw_re[k];
        let wi = unpack_tw_im[k];

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

// ============================================================================
// Tensor helpers
// ============================================================================

fn make_tensors_typed<E: burn_backend::Element + bytemuck::Pod>(
    re: Vec<E>,
    im: Vec<E>,
    shape: Shape,
) -> (FlexTensor, FlexTensor) {
    let dtype = E::dtype();
    let re_t = FlexTensor::new(
        Bytes::from_elems(re),
        Layout::contiguous(shape.clone()),
        dtype,
    );
    let im_t = FlexTensor::new(Bytes::from_elems(im), Layout::contiguous(shape), dtype);
    (re_t, im_t)
}

// ============================================================================
// Top-level rfft: real FFT via complex packing
// ============================================================================

/// Process a single fiber: pack real signal as complex, FFT, unpack.
#[allow(clippy::too_many_arguments)]
#[inline]
fn rfft_fiber(
    signal: &[f32],
    in_stride: usize,
    n: usize,
    out_re: &mut [f32],
    out_im: &mut [f32],
    tw_half: &TwiddleRef,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    let half = n / 2;

    if n == 1 {
        out_re[0] = signal[0];
        out_im[0] = 0.0;
        return;
    }

    if in_stride == 1 {
        for k in 0..half {
            z_re[k] = signal[2 * k];
            z_im[k] = signal[2 * k + 1];
        }
    } else {
        for k in 0..half {
            z_re[k] = signal[(2 * k) * in_stride];
            z_im[k] = signal[(2 * k + 1) * in_stride];
        }
    }

    complex_fft(z_re, z_im, half, tw_half);
    unpack_rfft(z_re, z_im, half, unpack_tw_re, unpack_tw_im, out_re, out_im);
}

pub fn rfft_f32(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    debug_assert!(
        dim < shape.num_dims(),
        "rfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );
    let n = shape[dim];
    debug_assert!(
        n > 0 && n.is_power_of_two(),
        "rfft: dimension size must be a power of 2, got {n}"
    );
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / n;

    let data: &[f32] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);

    // N=1: each element is its own DFT, no twiddles needed
    if n == 1 {
        let re_out: Vec<f32> = data.to_vec();
        let im_out = vec![0.0f32; total_out];
        return make_tensors_typed(re_out, im_out, out_shape);
    }

    let half = n / 2;
    let tw_half = get_twiddles(half);

    // Unpacking twiddles: last stage of size-N twiddle table = W_N^k for k=0..N/2-1
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_tw_re = &tw_full.re()[last_stage_off..];
    let unpack_tw_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f32; total_out];
    let mut im_out = vec![0.0f32; total_out];

    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f32; half.max(1)];
                let mut z_im = vec![0.0f32; half.max(1)];
                let mut fiber_re = vec![0.0f32; out_len];
                let mut fiber_im = vec![0.0f32; out_len];

                rfft_fiber(
                    &data[base_offset..],
                    in_stride,
                    n,
                    &mut fiber_re,
                    &mut fiber_im,
                    &tw_half,
                    unpack_tw_re,
                    unpack_tw_im,
                    &mut z_re,
                    &mut z_im,
                );
                (fiber_idx, fiber_re, fiber_im)
            })
            .collect();

        for (fiber_idx, fiber_re, fiber_im) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = fiber_re[k];
                im_out[out_base + k * out_stride] = fiber_im[k];
            }
        }

        return make_tensors_typed(re_out, im_out, out_shape);
    }

    let mut z_re_buf = vec![0.0f32; half.max(1)];
    let mut z_im_buf = vec![0.0f32; half.max(1)];
    let mut fiber_re = vec![0.0f32; out_len];
    let mut fiber_im = vec![0.0f32; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber(
            &data[base_offset..],
            in_stride,
            n,
            &mut fiber_re,
            &mut fiber_im,
            &tw_half,
            unpack_tw_re,
            unpack_tw_im,
            &mut z_re_buf,
            &mut z_im_buf,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    make_tensors_typed(re_out, im_out, out_shape)
}

#[allow(clippy::too_many_arguments)]
fn rfft_fiber_f64(
    signal: &[f64],
    in_stride: usize,
    n: usize,
    half: usize,
    out_re: &mut [f64],
    out_im: &mut [f64],
    tw_re: &[f32],
    tw_im: &[f32],
    tw_offsets: &[usize],
    unpack_re: &[f32],
    unpack_im: &[f32],
    z_re: &mut [f64],
    z_im: &mut [f64],
) {
    if n == 1 {
        out_re[0] = signal[0];
        out_im[0] = 0.0;
        return;
    }

    for k in 0..half {
        z_re[k] = signal[(2 * k) * in_stride];
        z_im[k] = signal[(2 * k + 1) * in_stride];
    }

    fft_f64_inplace(z_re, z_im, half, tw_re, tw_im, tw_offsets);

    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        let wr = unpack_re[k] as f64;
        let wi = unpack_im[k] as f64;

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

pub fn rfft_f64(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    debug_assert!(
        dim < shape.num_dims(),
        "rfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );
    let n = shape[dim];
    debug_assert!(
        n > 0 && n.is_power_of_two(),
        "rfft: dimension size must be a power of 2, got {n}"
    );
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / n;

    let data: &[f64] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);
    let half = n / 2;

    // Use f32 twiddles widened to f64
    let tw_half = get_twiddles(half);
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_re = &tw_full.re()[last_stage_off..];
    let unpack_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f64; total_out];
    let mut im_out = vec![0.0f64; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    let tw_half_re = tw_half.re();
    let tw_half_im = tw_half.im();
    let tw_half_offsets = tw_half.offsets();

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f64; half.max(1)];
                let mut z_im = vec![0.0f64; half.max(1)];
                let mut fiber_re = vec![0.0f64; out_len];
                let mut fiber_im = vec![0.0f64; out_len];

                rfft_fiber_f64(
                    &data[base_offset..],
                    in_stride,
                    n,
                    half,
                    &mut fiber_re,
                    &mut fiber_im,
                    tw_half_re,
                    tw_half_im,
                    tw_half_offsets,
                    unpack_re,
                    unpack_im,
                    &mut z_re,
                    &mut z_im,
                );
                (fiber_idx, fiber_re, fiber_im)
            })
            .collect();

        for (fiber_idx, fiber_re, fiber_im) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = fiber_re[k];
                im_out[out_base + k * out_stride] = fiber_im[k];
            }
        }

        return make_tensors_typed(re_out, im_out, out_shape);
    }

    let mut z_re = vec![0.0f64; half.max(1)];
    let mut z_im = vec![0.0f64; half.max(1)];
    let mut fiber_re = vec![0.0f64; out_len];
    let mut fiber_im = vec![0.0f64; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber_f64(
            &data[base_offset..],
            in_stride,
            n,
            half,
            &mut fiber_re,
            &mut fiber_im,
            tw_half_re,
            tw_half_im,
            tw_half_offsets,
            unpack_re,
            unpack_im,
            &mut z_re,
            &mut z_im,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    make_tensors_typed(re_out, im_out, out_shape)
}

/// f64 complex FFT using f32 twiddle table (widened in inner loop).
/// Twiddle precision is limited to ~7 digits (f32), so output accuracy
/// is below full f64 precision for large N.
fn fft_f64_inplace(
    re: &mut [f64],
    im: &mut [f64],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
) {
    if n <= 1 {
        return;
    }

    // Bit-reversal
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

    // Scalar radix-2 passes (radix-4 fusion not implemented for f64)
    let num_stages = offsets.len() - 1;
    let mut len = 2;
    for &tw_off in &offsets[..num_stages] {
        let half = len / 2;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let wr = tw_re[tw_off + k] as f64;
                let wi = tw_im[tw_off + k] as f64;
                let even = start + k;
                let odd = even + half;
                let t_re = wr * re[odd] - wi * im[odd];
                let t_im = wr * im[odd] + wi * re[odd];
                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] += t_re;
                im[even] += t_im;
            }
            start += len;
        }
        len <<= 1;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, TensorData, Tolerance};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape))
    }

    fn make_f64(data: Vec<f64>, shape: Vec<usize>) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape))
    }

    fn assert_approx(tensor: FlexTensor, expected: &[f32], tol: f32) {
        tensor
            .into_data()
            .assert_approx_eq::<f32>(&TensorData::from(expected), Tolerance::absolute(tol));
    }

    fn assert_approx_f64(tensor: FlexTensor, expected: &[f64], tol: f64) {
        tensor
            .into_data()
            .assert_approx_eq::<f64>(&TensorData::from(expected), Tolerance::absolute(tol));
    }

    // ---- N=1 ----

    #[test]
    fn rfft_n1() {
        let signal = make_f32(vec![5.0], vec![1]);
        let (re, im) = rfft_f32(signal, 0);
        assert_approx(re, &[5.0], 1e-6);
        assert_approx(im, &[0.0], 1e-6);
    }

    // ---- N=2 ----

    #[test]
    fn rfft_n2() {
        let signal = make_f32(vec![1.0, -1.0], vec![2]);
        let (re, im) = rfft_f32(signal, 0);
        assert_approx(re, &[0.0, 2.0], 1e-6);
        assert_approx(im, &[0.0, 0.0], 1e-6);
    }

    // ---- N=4: known DFT of [1,0,0,0] = [1,1,1] (all real) ----

    #[test]
    fn rfft_n4_impulse() {
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0);
        assert_approx(re, &[1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: constant signal [1,1,1,1] -> DC only ----

    #[test]
    fn rfft_n4_constant() {
        let signal = make_f32(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0);
        assert_approx(re, &[4.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: zeros ----

    #[test]
    fn rfft_n4_zeros() {
        let signal = make_f32(vec![0.0; 4], vec![4]);
        let (re, im) = rfft_f32(signal, 0);
        assert_approx(re, &[0.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=8 ----

    #[test]
    fn rfft_n8_impulse() {
        let mut signal = vec![0.0f32; 8];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0);
        // DFT of impulse is all 1s
        assert_approx(re, &[1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn rfft_n8_cosine() {
        // cos(2*pi*k/8) for k=0..7 -> energy at bin 1
        let signal: Vec<f32> = (0..8)
            .map(|k| (2.0 * std::f32::consts::PI * k as f32 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0);
        // Bin 1 should have amplitude 4 (real), rest ~0
        assert_approx(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-4);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-4);
    }

    // ---- Larger size: N=256 ----

    #[test]
    fn rfft_n256_impulse() {
        let mut signal = vec![0.0f32; 256];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![256]), 0);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 129);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "re bin should be 1.0, got {v}");
        }
        for &v in im_vals {
            assert!(v.abs() < 1e-5, "im bin should be 0.0, got {v}");
        }
    }

    // ---- Multi-dimensional: FFT along dim 1 ----

    #[test]
    fn rfft_2d_dim1() {
        // 2 rows, each of length 4: impulse and constant
        let data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: impulse
            1.0, 1.0, 1.0, 1.0, // row 1: constant
        ];
        let signal = make_f32(data, vec![2, 4]);
        let (re, im) = rfft_f32(signal, 1);
        // Shape should be [2, 3]
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6); // 2 * 3
        // Row 0 (impulse): [1, 1, 1]
        assert!((re_vals[0] - 1.0).abs() < 1e-5);
        assert!((re_vals[1] - 1.0).abs() < 1e-5);
        assert!((re_vals[2] - 1.0).abs() < 1e-5);
        // Row 1 (constant): [4, 0, 0]
        assert!((re_vals[3] - 4.0).abs() < 1e-5);
        assert!((re_vals[4]).abs() < 1e-5);
        assert!((re_vals[5]).abs() < 1e-5);
        // All imaginary should be ~0
        for &v in im_vals {
            assert!(v.abs() < 1e-5);
        }
    }

    // ---- FFT along dim 0 ----

    #[test]
    fn rfft_2d_dim0() {
        // 4 rows, 2 cols: impulse in each column
        let data = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let signal = make_f32(data, vec![4, 2]);
        let (re, im) = rfft_f32(signal, 0);
        // Shape should be [3, 2]
        let re_data = re.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6);
        // Each column is an impulse -> all bins = 1
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    // ---- f64 dtype ----

    #[test]
    fn rfft_f64_n4_impulse() {
        let signal = make_f64(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f64(signal, 0);
        assert_approx_f64(re, &[1.0, 1.0, 1.0], 1e-10);
        assert_approx_f64(im, &[0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn rfft_f64_n8_cosine() {
        let signal: Vec<f64> = (0..8)
            .map(|k| (2.0 * std::f64::consts::PI * k as f64 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f64(make_f64(signal, vec![8]), 0);
        assert_approx_f64(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-6);
        assert_approx_f64(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    // ---- f16 dtype ----

    #[test]
    fn rfft_f16_n4_impulse() {
        use burn_std::f16;
        let f16_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
        ];
        let signal = FlexTensor::new(
            Bytes::from_elems(f16_data),
            Layout::contiguous(Shape::from(vec![4])),
            DType::F16,
        );
        let (re, _im) = rfft_f16(signal, 0);
        // Verify via round-trip to f32
        let re_f32 = super::super::module::cast_to_f32(re, f16::to_f32);
        let re_data = re_f32.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 3);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    // ---- Const twiddle accuracy ----

    #[test]
    fn const_sin_cos_accuracy() {
        let test_angles = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, -1.0, -3.0, 6.0];
        for &angle in &test_angles {
            let cs = const_sin(angle);
            let cc = const_cos(angle);
            let rs = angle.sin();
            let rc = angle.cos();
            assert!(
                (cs - rs).abs() < 1e-12,
                "const_sin({angle}) = {cs}, expected {rs}"
            );
            assert!(
                (cc - rc).abs() < 1e-12,
                "const_cos({angle}) = {cc}, expected {rc}"
            );
        }
    }

    // ---- N=1024 round-trip with known property: Parseval's theorem ----
    // Sum of |x|^2 = (1/N) * Sum of |X|^2

    #[test]
    fn rfft_n1024_parseval() {
        let n = 1024;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let time_energy: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();

        let (re, im) = rfft_f32(make_f32(signal, vec![n]), 0);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();

        // Frequency energy: DC and Nyquist count once, others count double
        let out_len = n / 2 + 1;
        let mut freq_energy = 0.0f64;
        for k in 0..out_len {
            let mag2 = (re_vals[k] as f64).powi(2) + (im_vals[k] as f64).powi(2);
            if k == 0 || k == n / 2 {
                freq_energy += mag2;
            } else {
                freq_energy += 2.0 * mag2;
            }
        }
        freq_energy /= n as f64;

        let rel_err = (freq_energy - time_energy).abs() / time_energy;
        assert!(
            rel_err < 1e-4,
            "Parseval's theorem violated: time={time_energy}, freq={freq_energy}, rel_err={rel_err}"
        );
    }
}

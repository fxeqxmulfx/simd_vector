use std::arch::x86_64::*;
use std::iter::Sum;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

use crate::*;

/// A 4-lane SIMD vector of `f32` values, backed by SSE 128-bit registers.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec4(pub [f32; 4]);

/// Formats the vector as `Vec4([a, b, c, d])` for debug output.
impl std::fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Vec4").field(&self.0).finish()
    }
}

/// Compares two vectors for exact element-wise equality.
impl PartialEq for Vec4 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Vec4 {
    pub const ZERO: Self = Vec4([0.0; 4]);
    pub const ONE: Self = Vec4([1.0; 4]);

    /// Loads the array contents into an SSE register.
    #[inline(always)]
    pub(crate) fn load(&self) -> __m128 {
        unsafe { _mm_load_ps(self.0.as_ptr()) }
    }

    /// Stores an SSE register into a new `Vec4`.
    #[inline(always)]
    pub(crate) fn store(v: __m128) -> Self {
        let mut r = Vec4([0.0; 4]);
        unsafe { _mm_store_ps(r.0.as_mut_ptr(), v) };
        r
    }

    /// Creates a vector with all lanes set to `val`.
    #[inline(always)]
    pub const fn splat(val: f32) -> Self {
        Vec4([val; 4])
    }

    /// Computes fused multiply-add: `self * a + b` per lane.
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self::store(_mm_fmadd_ps(self.load(), a.load(), b.load())) }
    }


    /// Returns the absolute value of each lane.
    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));
            Self::store(_mm_and_ps(self.load(), mask))
        }
    }

    /// Returns the square root of each lane.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe { Self::store(_mm_sqrt_ps(self.load())) }
    }

    /// Returns the floor (round toward negative infinity) of each lane.
    #[inline(always)]
    pub fn floor(self) -> Self {
        unsafe {
            Self::store(_mm_round_ps(
                self.load(),
                _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            ))
        }
    }

    /// Returns the ceil (round toward positive infinity) of each lane.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        unsafe {
            Self::store(_mm_round_ps(
                self.load(),
                _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            ))
        }
    }

    /// Returns the nearest integer (round half away from zero) of each lane.
    #[inline(always)]
    pub fn round(self) -> Self {
        unsafe {
            let v = self.load();
            let t = _mm_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            let r = _mm_sub_ps(v, t);
            let abs_r = _mm_andnot_ps(_mm_set1_ps(-0.0), r);
            let need_adjust = _mm_cmpge_ps(abs_r, _mm_set1_ps(0.5));
            let sign = _mm_and_ps(v, _mm_set1_ps(-0.0));
            let one_signed = _mm_or_ps(_mm_set1_ps(1.0), sign);
            let adjusted = _mm_add_ps(t, one_signed);
            Self::store(_mm_or_ps(
                _mm_and_ps(need_adjust, adjusted),
                _mm_andnot_ps(need_adjust, t),
            ))
        }
    }
}

impl crate::precise::PreciseMath for Vec4 {
    fn sin(self) -> Self {
        unsafe { Self::store(sinf_u10_sse(self.load())) }
    }
    fn cos(self) -> Self {
        unsafe { Self::store(cosf_u10_sse(self.load())) }
    }
    fn exp(self) -> Self {
        unsafe { Self::store(expf_sse(self.load())) }
    }
    fn sum(self) -> f32 {
        unsafe {
            let v = _mm256_cvtps_pd(self.load());
            let hi = _mm256_extractf128_pd(v, 1);
            let lo = _mm256_castpd256_pd128(v);
            let sum2 = _mm_add_pd(lo, hi);
            let hi_elem = _mm_unpackhi_pd(sum2, sum2);
            let sum1 = _mm_add_sd(sum2, hi_elem);
            _mm_cvtss_f32(_mm_cvtsd_ss(_mm_setzero_ps(), sum1))
        }
    }
    fn dot(self, other: Self) -> f32 {
        unsafe {
            let a_f64 = _mm256_cvtps_pd(self.load());
            let b_f64 = _mm256_cvtps_pd(other.load());
            let prod = _mm256_mul_pd(a_f64, b_f64);
            let hi = _mm256_extractf128_pd(prod, 1);
            let lo = _mm256_castpd256_pd128(prod);
            let sum2 = _mm_add_pd(lo, hi);
            let hi_elem = _mm_unpackhi_pd(sum2, sum2);
            let sum1 = _mm_add_sd(sum2, hi_elem);
            _mm_cvtss_f32(_mm_cvtsd_ss(_mm_setzero_ps(), sum1))
        }
    }
}

impl crate::fast::FastMath for Vec4 {
    fn sin(self) -> Self {
        unsafe { Self::store(sinf_u35_sse(self.load())) }
    }
    fn cos(self) -> Self {
        unsafe { Self::store(cosf_u35_sse(self.load())) }
    }
    fn exp(self) -> Self {
        unsafe { Self::store(expf_sse(self.load())) }
    }
    fn sum(self) -> f32 {
        unsafe {
            let v = self.load();
            let shuf = _mm_movehdup_ps(v);
            let sums = _mm_add_ps(v, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        }
    }
    fn dot(self, other: Self) -> f32 {
        unsafe {
            let prod = _mm_mul_ps(self.load(), other.load());
            let shuf = _mm_movehdup_ps(prod);
            let sums = _mm_add_ps(prod, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        }
    }
}

/// Sums an iterator of `Vec4` values element-wise, starting from zero.
impl Sum for Vec4 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

/// Indexes into the vector to retrieve a single `f32` lane.
impl Index<usize> for Vec4 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Negates each lane of the vector.
impl Neg for Vec4 {
    type Output = Vec4;
    fn neg(self) -> Self::Output {
        unsafe {
            let sign = _mm_set1_ps(-0.0);
            Self::store(_mm_xor_ps(self.load(), sign))
        }
    }
}

/// Element-wise addition of two vectors.
impl Add<Vec4> for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Self::Output {
        unsafe { Self::store(_mm_add_ps(self.load(), rhs.load())) }
    }
}

/// Element-wise subtraction of two vectors.
impl Sub<Vec4> for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Self::Output {
        unsafe { Self::store(_mm_sub_ps(self.load(), rhs.load())) }
    }
}

/// Element-wise multiplication of two vectors.
impl Mul<Vec4> for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        unsafe { Self::store(_mm_mul_ps(self.load(), rhs.load())) }
    }
}

/// Element-wise division of two vectors.
impl Div<Vec4> for Vec4 {
    type Output = Vec4;
    fn div(self, rhs: Vec4) -> Self::Output {
        unsafe { Self::store(_mm_div_ps(self.load(), rhs.load())) }
    }
}

/// Adds a scalar to each lane of the vector.
impl Add<f32> for Vec4 {
    type Output = Vec4;
    fn add(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm_add_ps(self.load(), _mm_set1_ps(rhs))) }
    }
}

/// Subtracts a scalar from each lane of the vector.
impl Sub<f32> for Vec4 {
    type Output = Vec4;
    fn sub(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm_sub_ps(self.load(), _mm_set1_ps(rhs))) }
    }
}

/// Multiplies each lane of the vector by a scalar.
impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm_mul_ps(self.load(), _mm_set1_ps(rhs))) }
    }
}

/// Divides each lane of the vector by a scalar.
impl Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm_div_ps(self.load(), _mm_set1_ps(rhs))) }
    }
}

/// Adds a scalar to each lane of the vector (`f32 + Vec4`).
impl Add<Vec4> for f32 {
    type Output = Vec4;
    fn add(self, rhs: Vec4) -> Self::Output {
        unsafe { Vec4::store(_mm_add_ps(_mm_set1_ps(self), rhs.load())) }
    }
}

/// Subtracts each lane of the vector from a scalar (`f32 - Vec4`).
impl Sub<Vec4> for f32 {
    type Output = Vec4;
    fn sub(self, rhs: Vec4) -> Self::Output {
        unsafe { Vec4::store(_mm_sub_ps(_mm_set1_ps(self), rhs.load())) }
    }
}

/// Multiplies a scalar by each lane of the vector (`f32 * Vec4`).
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Self::Output {
        unsafe { Vec4::store(_mm_mul_ps(_mm_set1_ps(self), rhs.load())) }
    }
}

/// Divides a scalar by each lane of the vector (`f32 / Vec4`).
impl Div<Vec4> for f32 {
    type Output = Vec4;
    fn div(self, rhs: Vec4) -> Self::Output {
        unsafe { Vec4::store(_mm_div_ps(_mm_set1_ps(self), rhs.load())) }
    }
}

/// Creates a `Vec4` from a `[f32; 4]` array.
impl From<[f32; 4]> for Vec4 {
    fn from(arr: [f32; 4]) -> Self {
        Vec4(arr)
    }
}

/// Extracts the inner `[f32; 4]` array from a `Vec4`.
impl From<Vec4> for [f32; 4] {
    fn from(v: Vec4) -> Self {
        v.0
    }
}

// ---------------------------------------------------------------------------
// SLEEF transcendental implementations (SSE4.1 + FMA3)
// ---------------------------------------------------------------------------

/// Double-float pair (hi, lo) across 4 SSE lanes, for extended precision arithmetic.
#[derive(Clone, Copy)]
struct F2x4 {
    hi: __m128,
    lo: __m128,
}

/// Normalizes a double-float pair so that `hi` carries the leading bits (SSE).
#[inline(always)]
unsafe fn df_normalize_sse(a: F2x4) -> F2x4 {
    let s = _mm_add_ps(a.hi, a.lo);
    F2x4 {
        hi: s,
        lo: _mm_add_ps(_mm_sub_ps(a.hi, s), a.lo),
    }
}

/// Error-free addition of two `__m128` floats into a double-float pair (SSE).
#[inline(always)]
unsafe fn df_add2_f_f_sse(x: __m128, y: __m128) -> F2x4 {
    let s = _mm_add_ps(x, y);
    let v = _mm_sub_ps(s, x);
    F2x4 {
        hi: s,
        lo: _mm_add_ps(_mm_sub_ps(x, _mm_sub_ps(s, v)), _mm_sub_ps(y, v)),
    }
}

/// Adds a single `__m128` to a double-float pair (fast, SSE).
#[inline(always)]
unsafe fn df_add_f2_f_sse(x: F2x4, y: __m128) -> F2x4 {
    let s = _mm_add_ps(x.hi, y);
    F2x4 {
        hi: s,
        lo: _mm_add_ps(_mm_add_ps(_mm_sub_ps(x.hi, s), y), x.lo),
    }
}

/// Error-free addition of a double-float pair and a single `__m128` (SSE).
#[inline(always)]
unsafe fn df_add2_f2_f_sse(x: F2x4, y: __m128) -> F2x4 {
    let s = _mm_add_ps(x.hi, y);
    let v = _mm_sub_ps(s, x.hi);
    let t = _mm_add_ps(_mm_sub_ps(x.hi, _mm_sub_ps(s, v)), _mm_sub_ps(y, v));
    F2x4 {
        hi: s,
        lo: _mm_add_ps(t, x.lo),
    }
}

/// Error-free addition of two double-float pairs (SSE).
#[inline(always)]
unsafe fn df_add2_f2_f2_sse(x: F2x4, y: F2x4) -> F2x4 {
    let s = _mm_add_ps(x.hi, y.hi);
    let v = _mm_sub_ps(s, x.hi);
    let t = _mm_add_ps(_mm_sub_ps(x.hi, _mm_sub_ps(s, v)), _mm_sub_ps(y.hi, v));
    F2x4 {
        hi: s,
        lo: _mm_add_ps(t, _mm_add_ps(x.lo, y.lo)),
    }
}

/// Adds a single `__m128` to a double-float pair, scalar first (SSE).
#[inline(always)]
unsafe fn df_add_f_f2_sse(x: __m128, y: F2x4) -> F2x4 {
    let s = _mm_add_ps(x, y.hi);
    F2x4 {
        hi: s,
        lo: _mm_add_ps(_mm_add_ps(_mm_sub_ps(x, s), y.hi), y.lo),
    }
}

/// FMA-based multiplication of two double-float pairs (SSE).
#[inline(always)]
unsafe fn df_mul_f2_f2_sse(x: F2x4, y: F2x4) -> F2x4 {
    let s = _mm_mul_ps(x.hi, y.hi);
    let t = _mm_fmadd_ps(
        x.hi,
        y.lo,
        _mm_fmadd_ps(x.lo, y.hi, _mm_fmsub_ps(x.hi, y.hi, s)),
    );
    F2x4 { hi: s, lo: t }
}

/// Squares a double-float pair using FMA (SSE).
#[inline(always)]
unsafe fn df_squ_f2_sse(x: F2x4) -> F2x4 {
    let s = _mm_mul_ps(x.hi, x.hi);
    let t = _mm_fmadd_ps(_mm_add_ps(x.hi, x.hi), x.lo, _mm_fmsub_ps(x.hi, x.hi, s));
    F2x4 { hi: s, lo: t }
}

/// Evaluates the product of two double-float pairs as a single `__m128` (SSE).
#[inline(always)]
unsafe fn df_to_f_sse(x: F2x4, y: F2x4) -> __m128 {
    _mm_fmadd_ps(x.hi, y.hi, _mm_fmadd_ps(x.lo, y.hi, _mm_mul_ps(x.hi, y.lo)))
}

/// Constructs 2^q as a float for each lane (SSE).
#[inline(always)]
unsafe fn vpow2i_sse(q: __m128i) -> __m128 {
    _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(q, _mm_set1_epi32(0x7F)), 23))
}

/// Computes `d * 2^e` per lane, splitting the exponent to avoid overflow (SSE).
#[inline(always)]
unsafe fn vldexp2_sse(d: __m128, e: __m128i) -> __m128 {
    let e1 = _mm_srai_epi32(e, 1);
    let e2 = _mm_sub_epi32(e, e1);
    _mm_mul_ps(_mm_mul_ps(d, vpow2i_sse(e1)), vpow2i_sse(e2))
}

/// Payne-Hanek pi reduction for 4 SSE lanes (scalar fallback, no gather).
#[inline(always)]
unsafe fn rempif_sse(d: __m128) -> (F2x4, __m128i) {
    #[repr(C, align(16))]
    struct A16([f32; 4]);
    #[repr(C, align(16))]
    struct I16([i32; 4]);

    let mut arr = A16([0.0; 4]);
    _mm_store_ps(arr.0.as_mut_ptr(), d);
    let mut rhi = A16([0.0; 4]);
    let mut rlo = A16([0.0; 4]);
    let mut rq = I16([0; 4]);
    for i in 0..4 {
        let (hi, lo, q) = rempif_scalar(arr.0[i]);
        rhi.0[i] = hi;
        rlo.0[i] = lo;
        rq.0[i] = q;
    }
    (
        F2x4 {
            hi: _mm_load_ps(rhi.0.as_ptr()),
            lo: _mm_load_ps(rlo.0.as_ptr()),
        },
        _mm_load_si128(rq.0.as_ptr() as *const __m128i),
    )
}

/// Applies the sign of `y` to `x` per lane (SSE).
#[inline(always)]
unsafe fn vmulsign_sse(x: __m128, y: __m128) -> __m128 {
    _mm_xor_ps(x, _mm_and_ps(y, _mm_set1_ps(-0.0)))
}

/// Returns a mask that is true where `d` is negative zero (SSE).
#[inline(always)]
unsafe fn visnegzero_sse(d: __m128) -> __m128 {
    _mm_cmpeq_ps(d, _mm_set1_ps(-0.0))
}

/// SLEEF `xsinf_u10` sine implementation for SSE4.1+FMA3.
unsafe fn sinf_u10_sse(d: __m128) -> __m128 {
    let neg_zero = _mm_set1_ps(-0.0);
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));

    // Range reduction: q = round(d / pi)
    let u = _mm_round_ps(
        _mm_mul_ps(d, _mm_set1_ps(M_1_PI_F)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let q = _mm_cvtps_epi32(u);

    // Cody-Waite: s = d - q*pi in double-float
    let v = _mm_fmadd_ps(u, _mm_set1_ps(-PI_A2F), d);
    let mut s = df_add2_f_f_sse(v, _mm_mul_ps(u, _mm_set1_ps(-PI_B2F)));
    s = df_add_f2_f_sse(s, _mm_mul_ps(u, _mm_set1_ps(-PI_C2F)));

    let abs_d = _mm_and_ps(d, abs_mask);
    let in_range = _mm_cmplt_ps(abs_d, _mm_set1_ps(TRIGRANGEMAX2F));
    let all_in_range = _mm_movemask_ps(in_range) == 0xF;

    let (mut final_s, mut final_q) = (s, q);

    if !all_in_range {
        let (dfi, q2_raw) = rempif_sse(d);
        let q2_and = _mm_and_si128(q2_raw, _mm_set1_epi32(3));
        let dfi_x_gt0 = _mm_cmpgt_ps(dfi.hi, _mm_setzero_ps());
        let sel = _mm_castps_si128(dfi_x_gt0);
        let mut q2 = _mm_add_epi32(
            _mm_add_epi32(q2_and, q2_and),
            _mm_or_si128(
                _mm_and_si128(sel, _mm_set1_epi32(2)),
                _mm_andnot_si128(sel, _mm_set1_epi32(1)),
            ),
        );
        q2 = _mm_srai_epi32(q2, 2);

        // If quadrant is odd, subtract pi/2
        let odd = _mm_cmpeq_epi32(_mm_and_si128(q2_raw, _mm_set1_epi32(1)), _mm_set1_epi32(1));
        let odd_f = _mm_castsi128_ps(odd);
        let half_pi_hi = _mm_set1_ps(3.1415927410125732422 * -0.5);
        let half_pi_lo = _mm_set1_ps(-8.7422776573475857731e-08 * -0.5);
        let pi_adj = F2x4 {
            hi: vmulsign_sse(half_pi_hi, dfi.hi),
            lo: vmulsign_sse(half_pi_lo, dfi.hi),
        };
        let adj = df_add2_f2_f2_sse(dfi, pi_adj);
        let t_hi = _mm_or_ps(_mm_and_ps(odd_f, adj.hi), _mm_andnot_ps(odd_f, dfi.hi));
        let t_lo = _mm_or_ps(_mm_and_ps(odd_f, adj.lo), _mm_andnot_ps(odd_f, dfi.lo));
        let mut t = df_normalize_sse(F2x4 { hi: t_hi, lo: t_lo });

        // NaN/Inf -> NaN
        let is_bad = _mm_or_ps(
            _mm_cmpeq_ps(_mm_and_ps(d, abs_mask), _mm_set1_ps(f32::INFINITY)),
            _mm_cmpunord_ps(d, d),
        );
        t.hi = _mm_or_ps(t.hi, is_bad);

        // Blend: use rempif result where out of range
        let in_range_i = _mm_castps_si128(in_range);
        final_q = _mm_or_si128(
            _mm_and_si128(in_range_i, q),
            _mm_andnot_si128(in_range_i, q2),
        );
        final_s = F2x4 {
            hi: _mm_or_ps(_mm_and_ps(in_range, s.hi), _mm_andnot_ps(in_range, t.hi)),
            lo: _mm_or_ps(_mm_and_ps(in_range, s.lo), _mm_andnot_ps(in_range, t.lo)),
        };
    }

    let t = final_s;
    let s2 = df_squ_f2_sse(final_s);

    // Polynomial: coefficients from SLEEF
    let mut u = _mm_set1_ps(2.6083159809786593541503e-06);
    u = _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(-0.0001981069071916863322258));
    u = _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(0.00833307858556509017944336));

    // x = 1 + ((−0.1666... + u*s²) * s²)
    let inner = F2x4 {
        hi: _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(-0.166666597127914428710938)),
        lo: _mm_setzero_ps(),
    };
    let x = df_add_f_f2_sse(_mm_set1_ps(1.0), df_mul_f2_f2_sse(inner, s2));

    // result = t * x (as scalar)
    let mut result = df_to_f_sse(t, x);

    // Apply sign flip for odd quadrants
    let q_and_1 = _mm_cmpeq_epi32(_mm_and_si128(final_q, _mm_set1_epi32(1)), _mm_set1_epi32(1));
    let sign_flip = _mm_and_ps(_mm_castsi128_ps(q_and_1), neg_zero);
    result = _mm_xor_ps(result, sign_flip);

    // Preserve -0.0
    let is_neg_zero = visnegzero_sse(d);
    result = _mm_or_ps(
        _mm_and_ps(is_neg_zero, d),
        _mm_andnot_ps(is_neg_zero, result),
    );

    result
}

/// SLEEF `xcosf_u10` cosine implementation for SSE4.1+FMA3.
unsafe fn cosf_u10_sse(d: __m128) -> __m128 {
    let neg_zero = _mm_set1_ps(-0.0);
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));

    // Range reduction: q = round(d/pi - 0.5) * 2 + 1
    let dq = _mm_fmadd_ps(
        _mm_round_ps(
            _mm_fmadd_ps(d, _mm_set1_ps(M_1_PI_F), _mm_set1_ps(-0.5)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ),
        _mm_set1_ps(2.0),
        _mm_set1_ps(1.0),
    );
    let q = _mm_cvtps_epi32(dq);

    // Cody-Waite: s = d - dq*pi/2
    let mut s = df_add2_f_f_sse(d, _mm_mul_ps(dq, _mm_set1_ps(-PI_A2F * 0.5)));
    s = df_add2_f2_f_sse(s, _mm_mul_ps(dq, _mm_set1_ps(-PI_B2F * 0.5)));
    s = df_add2_f2_f_sse(s, _mm_mul_ps(dq, _mm_set1_ps(-PI_C2F * 0.5)));

    let abs_d = _mm_and_ps(d, abs_mask);
    let in_range = _mm_cmplt_ps(abs_d, _mm_set1_ps(TRIGRANGEMAX2F));
    let all_in_range = _mm_movemask_ps(in_range) == 0xF;

    let (mut final_s, mut final_q) = (s, q);

    if !all_in_range {
        let (dfi, q2_raw) = rempif_sse(d);
        let q2_and = _mm_and_si128(q2_raw, _mm_set1_epi32(3));
        let dfi_x_gt0 = _mm_cmpgt_ps(dfi.hi, _mm_setzero_ps());
        let sel = _mm_castps_si128(dfi_x_gt0);
        let mut q2 = _mm_add_epi32(
            _mm_add_epi32(q2_and, q2_and),
            _mm_or_si128(
                _mm_and_si128(sel, _mm_set1_epi32(8)),
                _mm_andnot_si128(sel, _mm_set1_epi32(7)),
            ),
        );
        q2 = _mm_srai_epi32(q2, 1);

        let even = _mm_cmpeq_epi32(
            _mm_and_si128(q2_raw, _mm_set1_epi32(1)),
            _mm_setzero_si128(),
        );
        let even_f = _mm_castsi128_ps(even);
        let y = _mm_or_ps(
            _mm_and_ps(dfi_x_gt0, _mm_setzero_ps()),
            _mm_andnot_ps(dfi_x_gt0, _mm_set1_ps(-1.0)),
        );
        let half_pi_hi = _mm_set1_ps(3.1415927410125732422 * -0.5);
        let half_pi_lo = _mm_set1_ps(-8.7422776573475857731e-08 * -0.5);
        let pi_adj = F2x4 {
            hi: vmulsign_sse(half_pi_hi, y),
            lo: vmulsign_sse(half_pi_lo, y),
        };
        let adj = df_add2_f2_f2_sse(dfi, pi_adj);
        let t_hi = _mm_or_ps(_mm_and_ps(even_f, adj.hi), _mm_andnot_ps(even_f, dfi.hi));
        let t_lo = _mm_or_ps(_mm_and_ps(even_f, adj.lo), _mm_andnot_ps(even_f, dfi.lo));
        let mut t = df_normalize_sse(F2x4 { hi: t_hi, lo: t_lo });

        let is_bad = _mm_or_ps(
            _mm_cmpeq_ps(abs_d, _mm_set1_ps(f32::INFINITY)),
            _mm_cmpunord_ps(d, d),
        );
        t.hi = _mm_or_ps(t.hi, is_bad);

        let in_range_i = _mm_castps_si128(in_range);
        final_q = _mm_or_si128(
            _mm_and_si128(in_range_i, q),
            _mm_andnot_si128(in_range_i, q2),
        );
        final_s = F2x4 {
            hi: _mm_or_ps(_mm_and_ps(in_range, s.hi), _mm_andnot_ps(in_range, t.hi)),
            lo: _mm_or_ps(_mm_and_ps(in_range, s.lo), _mm_andnot_ps(in_range, t.lo)),
        };
    }

    let t = final_s;
    let s2 = df_squ_f2_sse(final_s);

    let mut u = _mm_set1_ps(2.6083159809786593541503e-06);
    u = _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(-0.0001981069071916863322258));
    u = _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(0.00833307858556509017944336));

    let inner = F2x4 {
        hi: _mm_fmadd_ps(u, s2.hi, _mm_set1_ps(-0.166666597127914428710938)),
        lo: _mm_setzero_ps(),
    };
    let x = df_add_f_f2_sse(_mm_set1_ps(1.0), df_mul_f2_f2_sse(inner, s2));

    let mut result = df_to_f_sse(t, x);

    // Sign flip: cos flips when (q & 2) == 0
    let q_and_2_is_0 = _mm_cmpeq_epi32(
        _mm_and_si128(final_q, _mm_set1_epi32(2)),
        _mm_setzero_si128(),
    );
    let sign_flip = _mm_and_ps(_mm_castsi128_ps(q_and_2_is_0), neg_zero);
    result = _mm_xor_ps(result, sign_flip);

    result
}

/// SLEEF `xexpf` exponential implementation for SSE4.1+FMA3.
unsafe fn expf_sse(d: __m128) -> __m128 {
    // Range reduction: q = round(d / ln2)
    let q = _mm_cvtps_epi32(_mm_mul_ps(d, _mm_set1_ps(R_LN2F)));
    let qf = _mm_cvtepi32_ps(q);

    // s = d - q * ln2 (in two steps for precision)
    let mut s = _mm_fmadd_ps(qf, _mm_set1_ps(-L2UF), d);
    s = _mm_fmadd_ps(qf, _mm_set1_ps(-L2LF), s);

    // Polynomial approximation of exp(s) - 1 - s
    let mut u = _mm_set1_ps(0.000198527617612853646278381);
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.00139304355252534151077271));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.00833336077630519866943359));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.0416664853692054748535156));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.166666671633720397949219));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.5));

    // exp(s) = 1 + s + s^2 * u
    u = _mm_add_ps(_mm_set1_ps(1.0), _mm_fmadd_ps(_mm_mul_ps(s, s), u, s));

    // Scale by 2^q
    u = vldexp2_sse(u, q);

    // Clamp: d < -104 -> 0, d > 100 -> inf
    u = _mm_andnot_ps(_mm_cmplt_ps(d, _mm_set1_ps(-104.0)), u);
    u = _mm_or_ps(
        _mm_and_ps(
            _mm_cmplt_ps(_mm_set1_ps(100.0), d),
            _mm_set1_ps(f32::INFINITY),
        ),
        _mm_andnot_ps(_mm_cmplt_ps(_mm_set1_ps(100.0), d), u),
    );

    u
}

// ---------------------------------------------------------------------------
// SLEEF u35 (≤ 3.5 ULP) transcendental implementations (SSE4.1 + FMA3)
// ---------------------------------------------------------------------------

/// SLEEF `xsinf` sine for SSE4.1+FMA3 (≤ 3.5 ULP).
///
/// Same polynomial as u10, but uses scalar arithmetic instead of double-float pairs.
/// Three-tier range reduction:
/// - |x| < 125: 3-part Cody-Waite (PI_A2f/B2f/C2f)
/// - |x| < 39000: 4-part Cody-Waite (PI_Af/Bf/Cf/Df)
/// - |x| >= 39000: Payne-Hanek table-based reduction
unsafe fn sinf_u35_sse(d: __m128) -> __m128 {
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));
    let r = d;

    // Range reduction: q = round(d / pi)
    let q = _mm_cvtps_epi32(_mm_mul_ps(d, _mm_set1_ps(M_1_PI_F)));
    let u = _mm_cvtepi32_ps(q);

    // 3-part Cody-Waite: d = d - q*PI_A2f - q*PI_B2f - q*PI_C2f
    let mut d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_A2F), d);
    d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_B2F), d);
    d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_C2F), d);

    let abs_r = _mm_and_ps(r, abs_mask);
    let in_range1 = _mm_cmplt_ps(abs_r, _mm_set1_ps(TRIGRANGEMAX2F));
    let all_in_range1 = _mm_movemask_ps(in_range1) == 0xF;

    let (mut final_d, mut final_q) = (d, q);

    if !all_in_range1 {
        // 4-part Cody-Waite for medium range
        let s = _mm_cvtepi32_ps(q);
        let mut u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_AF), r);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_BF), u2);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_CF), u2);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_DF), u2);

        final_d = _mm_or_ps(
            _mm_and_ps(in_range1, d),
            _mm_andnot_ps(in_range1, u2),
        );

        let in_range2 = _mm_cmplt_ps(abs_r, _mm_set1_ps(TRIGRANGEMAXF));
        let all_in_range2 = _mm_movemask_ps(in_range2) == 0xF;

        if !all_in_range2 {
            // Payne-Hanek for very large args
            let (dfi, q2_raw) = rempif_sse(r);
            let q2_and = _mm_and_si128(q2_raw, _mm_set1_epi32(3));
            let dfi_x_gt0 = _mm_cmpgt_ps(dfi.hi, _mm_setzero_ps());
            let sel = _mm_castps_si128(dfi_x_gt0);
            let mut q2 = _mm_add_epi32(
                _mm_add_epi32(q2_and, q2_and),
                _mm_or_si128(
                    _mm_and_si128(sel, _mm_set1_epi32(2)),
                    _mm_andnot_si128(sel, _mm_set1_epi32(1)),
                ),
            );
            q2 = _mm_srai_epi32(q2, 2);

            let odd = _mm_cmpeq_epi32(
                _mm_and_si128(q2_raw, _mm_set1_epi32(1)),
                _mm_set1_epi32(1),
            );
            let odd_f = _mm_castsi128_ps(odd);
            let half_pi_hi = _mm_set1_ps(3.1415927410125732422 * -0.5);
            let half_pi_lo = _mm_set1_ps(-8.7422776573475857731e-08 * -0.5);
            let pi_adj = F2x4 {
                hi: vmulsign_sse(half_pi_hi, dfi.hi),
                lo: vmulsign_sse(half_pi_lo, dfi.hi),
            };
            let adj = df_add2_f2_f2_sse(dfi, pi_adj);
            let t_hi = _mm_or_ps(_mm_and_ps(odd_f, adj.hi), _mm_andnot_ps(odd_f, dfi.hi));
            let t_lo = _mm_or_ps(_mm_and_ps(odd_f, adj.lo), _mm_andnot_ps(odd_f, dfi.lo));
            let mut u3 = _mm_add_ps(t_hi, t_lo);

            let is_bad = _mm_or_ps(
                _mm_cmpeq_ps(abs_r, _mm_set1_ps(f32::INFINITY)),
                _mm_cmpunord_ps(r, r),
            );
            u3 = _mm_or_ps(u3, is_bad);

            let in_range2_i = _mm_castps_si128(in_range2);
            final_q = _mm_or_si128(
                _mm_and_si128(in_range2_i, final_q),
                _mm_andnot_si128(in_range2_i, q2),
            );
            final_d = _mm_or_ps(
                _mm_and_ps(in_range2, final_d),
                _mm_andnot_ps(in_range2, u3),
            );
        }
    }

    let s = _mm_mul_ps(final_d, final_d);

    // Sign flip for odd quadrants (applied to d before polynomial)
    let q_and_1 = _mm_cmpeq_epi32(
        _mm_and_si128(final_q, _mm_set1_epi32(1)),
        _mm_set1_epi32(1),
    );
    let sign_flip = _mm_and_ps(_mm_castsi128_ps(q_and_1), _mm_set1_ps(-0.0));
    final_d = _mm_xor_ps(final_d, sign_flip);

    // Polynomial (same coefficients as u10)
    let mut u = _mm_set1_ps(2.6083159809786593541503e-06);
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(-0.0001981069071916863322258));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.00833307858556509017944336));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(-0.166666597127914428710938));

    // u = s * (u * d) + d
    u = _mm_add_ps(_mm_mul_ps(s, _mm_mul_ps(u, final_d)), final_d);

    // Preserve -0.0
    let is_neg_zero = visnegzero_sse(r);
    u = _mm_or_ps(
        _mm_and_ps(is_neg_zero, r),
        _mm_andnot_ps(is_neg_zero, u),
    );

    u
}

/// SLEEF `xcosf` cosine for SSE4.1+FMA3 (≤ 3.5 ULP).
unsafe fn cosf_u35_sse(d: __m128) -> __m128 {
    let abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));
    let r = d;

    // Range reduction: q = 2*round(d/π - 0.5) + 1
    let q = _mm_cvtps_epi32(_mm_sub_ps(
        _mm_mul_ps(d, _mm_set1_ps(M_1_PI_F)),
        _mm_set1_ps(0.5),
    ));
    let q = _mm_add_epi32(_mm_add_epi32(q, q), _mm_set1_epi32(1));
    let u = _mm_cvtepi32_ps(q);

    // 3-part Cody-Waite
    let mut d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_A2F * 0.5), d);
    d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_B2F * 0.5), d);
    d = _mm_fmadd_ps(u, _mm_set1_ps(-PI_C2F * 0.5), d);

    let abs_r = _mm_and_ps(r, abs_mask);
    let in_range1 = _mm_cmplt_ps(abs_r, _mm_set1_ps(TRIGRANGEMAX2F));
    let all_in_range1 = _mm_movemask_ps(in_range1) == 0xF;

    let (mut final_d, mut final_q) = (d, q);

    if !all_in_range1 {
        let s = _mm_cvtepi32_ps(q);
        let mut u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_AF * 0.5), r);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_BF * 0.5), u2);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_CF * 0.5), u2);
        u2 = _mm_fmadd_ps(s, _mm_set1_ps(-PI_DF * 0.5), u2);

        final_d = _mm_or_ps(
            _mm_and_ps(in_range1, d),
            _mm_andnot_ps(in_range1, u2),
        );

        let in_range2 = _mm_cmplt_ps(abs_r, _mm_set1_ps(TRIGRANGEMAXF));
        let all_in_range2 = _mm_movemask_ps(in_range2) == 0xF;

        if !all_in_range2 {
            let (dfi, q2_raw) = rempif_sse(r);
            let q2_and = _mm_and_si128(q2_raw, _mm_set1_epi32(3));
            let dfi_x_gt0 = _mm_cmpgt_ps(dfi.hi, _mm_setzero_ps());
            let sel = _mm_castps_si128(dfi_x_gt0);
            let mut q2 = _mm_add_epi32(
                _mm_add_epi32(q2_and, q2_and),
                _mm_or_si128(
                    _mm_and_si128(sel, _mm_set1_epi32(8)),
                    _mm_andnot_si128(sel, _mm_set1_epi32(7)),
                ),
            );
            q2 = _mm_srai_epi32(q2, 1);

            let even = _mm_cmpeq_epi32(
                _mm_and_si128(q2_raw, _mm_set1_epi32(1)),
                _mm_setzero_si128(),
            );
            let even_f = _mm_castsi128_ps(even);
            let y = _mm_or_ps(
                _mm_and_ps(dfi_x_gt0, _mm_setzero_ps()),
                _mm_andnot_ps(dfi_x_gt0, _mm_set1_ps(-1.0)),
            );
            let half_pi_hi = _mm_set1_ps(3.1415927410125732422 * -0.5);
            let half_pi_lo = _mm_set1_ps(-8.7422776573475857731e-08 * -0.5);
            let pi_adj = F2x4 {
                hi: vmulsign_sse(half_pi_hi, y),
                lo: vmulsign_sse(half_pi_lo, y),
            };
            let adj = df_add2_f2_f2_sse(dfi, pi_adj);
            let t_hi = _mm_or_ps(_mm_and_ps(even_f, adj.hi), _mm_andnot_ps(even_f, dfi.hi));
            let t_lo = _mm_or_ps(_mm_and_ps(even_f, adj.lo), _mm_andnot_ps(even_f, dfi.lo));
            let mut u3 = _mm_add_ps(t_hi, t_lo);

            let is_bad = _mm_or_ps(
                _mm_cmpeq_ps(abs_r, _mm_set1_ps(f32::INFINITY)),
                _mm_cmpunord_ps(r, r),
            );
            u3 = _mm_or_ps(u3, is_bad);

            let in_range2_i = _mm_castps_si128(in_range2);
            final_q = _mm_or_si128(
                _mm_and_si128(in_range2_i, final_q),
                _mm_andnot_si128(in_range2_i, q2),
            );
            final_d = _mm_or_ps(
                _mm_and_ps(in_range2, final_d),
                _mm_andnot_ps(in_range2, u3),
            );
        }
    }

    let s = _mm_mul_ps(final_d, final_d);

    // Sign flip when (q & 2) == 0 (applied to d)
    let q_and_2_is_0 = _mm_cmpeq_epi32(
        _mm_and_si128(final_q, _mm_set1_epi32(2)),
        _mm_setzero_si128(),
    );
    let sign_flip = _mm_and_ps(_mm_castsi128_ps(q_and_2_is_0), _mm_set1_ps(-0.0));
    final_d = _mm_xor_ps(final_d, sign_flip);

    let mut u = _mm_set1_ps(2.6083159809786593541503e-06);
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(-0.0001981069071916863322258));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(0.00833307858556509017944336));
    u = _mm_fmadd_ps(u, s, _mm_set1_ps(-0.166666597127914428710938));

    u = _mm_add_ps(_mm_mul_ps(s, _mm_mul_ps(u, final_d)), final_d);

    u
}


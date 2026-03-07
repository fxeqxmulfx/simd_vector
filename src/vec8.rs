use std::arch::x86_64::*;
use std::iter::Sum;
use std::ops::{Add, Div, Index, Mul, Neg, Sub};

use crate::*;

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Vec8(pub [f32; 8]);

impl std::fmt::Debug for Vec8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Vec8").field(&self.0).finish()
    }
}

impl PartialEq for Vec8 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Vec8 {
    pub const ZERO: Self = Vec8([0.0; 8]);
    pub const ONE: Self = Vec8([1.0; 8]);

    #[inline(always)]
    fn load(&self) -> __m256 {
        unsafe { _mm256_load_ps(self.0.as_ptr()) }
    }

    #[inline(always)]
    fn store(v: __m256) -> Self {
        let mut r = Vec8([0.0; 8]);
        unsafe { _mm256_store_ps(r.0.as_mut_ptr(), v) };
        r
    }

    #[inline(always)]
    pub fn splat(val: f32) -> Self {
        Self::store(unsafe { _mm256_set1_ps(val) })
    }

    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { Self::store(_mm256_fmadd_ps(self.load(), a.load(), b.load())) }
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> f32 {
        unsafe {
            let prod = _mm256_mul_ps(self.load(), other.load());
            // Horizontal sum of 8 floats
            let hi128 = _mm256_extractf128_ps(prod, 1);
            let lo128 = _mm256_castps256_ps128(prod);
            let sum128 = _mm_add_ps(lo128, hi128);
            // Now sum the 4 floats in sum128
            let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
            let sums = _mm_add_ps(sum128, shuf); // [0+1, -, 2+3, -]
            let shuf2 = _mm_movehl_ps(sums, sums); // [2+3, -, -, -]
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
            Self::store(_mm256_and_ps(self.load(), mask))
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe { Self::store(_mm256_sqrt_ps(self.load())) }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        unsafe {
            Self::store(_mm256_round_ps(
                self.load(),
                _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            ))
        }
    }

    pub fn sin(self) -> Self {
        unsafe { Self::store(sinf_u1_avx2(self.load())) }
    }

    pub fn cos(self) -> Self {
        unsafe { Self::store(cosf_u1_avx2(self.load())) }
    }

    pub fn exp(self) -> Self {
        unsafe { Self::store(expf_avx2(self.load())) }
    }
}

impl Sum for Vec8 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Index<usize> for Vec8 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Neg for Vec8 {
    type Output = Vec8;
    fn neg(self) -> Self::Output {
        unsafe {
            let sign = _mm256_set1_ps(-0.0);
            Self::store(_mm256_xor_ps(self.load(), sign))
        }
    }
}

impl Add<Vec8> for Vec8 {
    type Output = Vec8;
    fn add(self, rhs: Vec8) -> Self::Output {
        unsafe { Self::store(_mm256_add_ps(self.load(), rhs.load())) }
    }
}

impl Sub<Vec8> for Vec8 {
    type Output = Vec8;
    fn sub(self, rhs: Vec8) -> Self::Output {
        unsafe { Self::store(_mm256_sub_ps(self.load(), rhs.load())) }
    }
}

impl Mul<Vec8> for Vec8 {
    type Output = Vec8;
    fn mul(self, rhs: Vec8) -> Self::Output {
        unsafe { Self::store(_mm256_mul_ps(self.load(), rhs.load())) }
    }
}

impl Div<Vec8> for Vec8 {
    type Output = Vec8;
    fn div(self, rhs: Vec8) -> Self::Output {
        unsafe { Self::store(_mm256_div_ps(self.load(), rhs.load())) }
    }
}

impl Add<f32> for Vec8 {
    type Output = Vec8;
    fn add(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm256_add_ps(self.load(), _mm256_set1_ps(rhs))) }
    }
}

impl Sub<f32> for Vec8 {
    type Output = Vec8;
    fn sub(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm256_sub_ps(self.load(), _mm256_set1_ps(rhs))) }
    }
}

impl Mul<f32> for Vec8 {
    type Output = Vec8;
    fn mul(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm256_mul_ps(self.load(), _mm256_set1_ps(rhs))) }
    }
}

impl Div<f32> for Vec8 {
    type Output = Vec8;
    fn div(self, rhs: f32) -> Self::Output {
        unsafe { Self::store(_mm256_div_ps(self.load(), _mm256_set1_ps(rhs))) }
    }
}

impl Add<Vec8> for f32 {
    type Output = Vec8;
    fn add(self, rhs: Vec8) -> Self::Output {
        unsafe { Vec8::store(_mm256_add_ps(_mm256_set1_ps(self), rhs.load())) }
    }
}

impl Sub<Vec8> for f32 {
    type Output = Vec8;
    fn sub(self, rhs: Vec8) -> Self::Output {
        unsafe { Vec8::store(_mm256_sub_ps(_mm256_set1_ps(self), rhs.load())) }
    }
}

impl Mul<Vec8> for f32 {
    type Output = Vec8;
    fn mul(self, rhs: Vec8) -> Self::Output {
        unsafe { Vec8::store(_mm256_mul_ps(_mm256_set1_ps(self), rhs.load())) }
    }
}

impl Div<Vec8> for f32 {
    type Output = Vec8;
    fn div(self, rhs: Vec8) -> Self::Output {
        unsafe { Vec8::store(_mm256_div_ps(_mm256_set1_ps(self), rhs.load())) }
    }
}

impl From<[f32; 8]> for Vec8 {
    fn from(arr: [f32; 8]) -> Self {
        Vec8(arr)
    }
}

impl From<Vec8> for [f32; 8] {
    fn from(v: Vec8) -> Self {
        v.0
    }
}

// ---------------------------------------------------------------------------
// SLEEF transcendental implementations (AVX2 + FMA3)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct F2x8 {
    hi: __m256,
    lo: __m256,
}

#[inline(always)]
unsafe fn df_normalize_avx(a: F2x8) -> F2x8 {
    let s = _mm256_add_ps(a.hi, a.lo);
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(_mm256_sub_ps(a.hi, s), a.lo),
    }
}

#[inline(always)]
unsafe fn df_add2_f_f_avx(x: __m256, y: __m256) -> F2x8 {
    let s = _mm256_add_ps(x, y);
    let v = _mm256_sub_ps(s, x);
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(
            _mm256_sub_ps(x, _mm256_sub_ps(s, v)),
            _mm256_sub_ps(y, v),
        ),
    }
}

#[inline(always)]
unsafe fn df_add_f2_f_avx(x: F2x8, y: __m256) -> F2x8 {
    let s = _mm256_add_ps(x.hi, y);
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(x.hi, s), y), x.lo),
    }
}

#[inline(always)]
unsafe fn df_add2_f2_f_avx(x: F2x8, y: __m256) -> F2x8 {
    let s = _mm256_add_ps(x.hi, y);
    let v = _mm256_sub_ps(s, x.hi);
    let t = _mm256_add_ps(
        _mm256_sub_ps(x.hi, _mm256_sub_ps(s, v)),
        _mm256_sub_ps(y, v),
    );
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(t, x.lo),
    }
}

#[inline(always)]
unsafe fn df_add2_f2_f2_avx(x: F2x8, y: F2x8) -> F2x8 {
    let s = _mm256_add_ps(x.hi, y.hi);
    let v = _mm256_sub_ps(s, x.hi);
    let t = _mm256_add_ps(
        _mm256_sub_ps(x.hi, _mm256_sub_ps(s, v)),
        _mm256_sub_ps(y.hi, v),
    );
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(t, _mm256_add_ps(x.lo, y.lo)),
    }
}

#[inline(always)]
unsafe fn df_add_f_f2_avx(x: __m256, y: F2x8) -> F2x8 {
    let s = _mm256_add_ps(x, y.hi);
    F2x8 {
        hi: s,
        lo: _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(x, s), y.hi), y.lo),
    }
}

#[inline(always)]
unsafe fn df_mul_f2_f2_avx(x: F2x8, y: F2x8) -> F2x8 {
    let s = _mm256_mul_ps(x.hi, y.hi);
    let t = _mm256_fmadd_ps(
        x.hi,
        y.lo,
        _mm256_fmadd_ps(x.lo, y.hi, _mm256_fmsub_ps(x.hi, y.hi, s)),
    );
    F2x8 { hi: s, lo: t }
}

#[inline(always)]
unsafe fn df_squ_f2_avx(x: F2x8) -> F2x8 {
    let s = _mm256_mul_ps(x.hi, x.hi);
    let t = _mm256_fmadd_ps(
        _mm256_add_ps(x.hi, x.hi),
        x.lo,
        _mm256_fmsub_ps(x.hi, x.hi, s),
    );
    F2x8 { hi: s, lo: t }
}

#[inline(always)]
unsafe fn df_to_f_avx(x: F2x8, y: F2x8) -> __m256 {
    _mm256_fmadd_ps(
        x.hi,
        y.hi,
        _mm256_fmadd_ps(x.lo, y.hi, _mm256_mul_ps(x.hi, y.lo)),
    )
}

#[inline(always)]
unsafe fn vpow2i_avx(q: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(q, _mm256_set1_epi32(0x7F)),
        23,
    ))
}

#[inline(always)]
unsafe fn vldexp2_avx(d: __m256, e: __m256i) -> __m256 {
    let e1 = _mm256_srai_epi32(e, 1);
    let e2 = _mm256_sub_epi32(e, e1);
    _mm256_mul_ps(_mm256_mul_ps(d, vpow2i_avx(e1)), vpow2i_avx(e2))
}

// Scalar-per-lane rempif for AVX2
#[inline(always)]
unsafe fn rempif_avx(d: __m256) -> (F2x8, __m256i) {
    #[repr(C, align(32))]
    struct A32([f32; 8]);
    #[repr(C, align(32))]
    struct I32([i32; 8]);

    let mut arr = A32([0.0; 8]);
    _mm256_store_ps(arr.0.as_mut_ptr(), d);
    let mut rhi = A32([0.0; 8]);
    let mut rlo = A32([0.0; 8]);
    let mut rq = I32([0; 8]);
    for i in 0..8 {
        let (hi, lo, q) = rempif_scalar(arr.0[i]);
        rhi.0[i] = hi;
        rlo.0[i] = lo;
        rq.0[i] = q;
    }
    (
        F2x8 {
            hi: _mm256_load_ps(rhi.0.as_ptr()),
            lo: _mm256_load_ps(rlo.0.as_ptr()),
        },
        _mm256_load_si256(rq.0.as_ptr() as *const __m256i),
    )
}

#[inline(always)]
unsafe fn vmulsign_avx(x: __m256, y: __m256) -> __m256 {
    _mm256_xor_ps(x, _mm256_and_ps(y, _mm256_set1_ps(-0.0)))
}

#[inline(always)]
unsafe fn visnegzero_avx(d: __m256) -> __m256 {
    _mm256_cmp_ps(d, _mm256_set1_ps(-0.0), _CMP_EQ_OQ)
}

// SLEEF xsinf_u1 for AVX2+FMA3
unsafe fn sinf_u1_avx2(d: __m256) -> __m256 {
    let neg_zero = _mm256_set1_ps(-0.0);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

    let u = _mm256_round_ps(
        _mm256_mul_ps(d, _mm256_set1_ps(M_1_PI_F)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let q = _mm256_cvtps_epi32(u);

    let v = _mm256_fmadd_ps(u, _mm256_set1_ps(-PI_A2F), d);
    let mut s = df_add2_f_f_avx(v, _mm256_mul_ps(u, _mm256_set1_ps(-PI_B2F)));
    s = df_add_f2_f_avx(s, _mm256_mul_ps(u, _mm256_set1_ps(-PI_C2F)));

    let abs_d = _mm256_and_ps(d, abs_mask);
    let in_range = _mm256_cmp_ps(abs_d, _mm256_set1_ps(TRIGRANGEMAX2F), _CMP_LT_OQ);
    let all_in_range = _mm256_movemask_ps(in_range) == 0xFF;

    let (mut final_s, mut final_q) = (s, q);

    if !all_in_range {
        let (dfi, q2_raw) = rempif_avx(d);
        let q2_and = _mm256_and_si256(q2_raw, _mm256_set1_epi32(3));
        let dfi_x_gt0 = _mm256_cmp_ps(dfi.hi, _mm256_setzero_ps(), _CMP_GT_OQ);
        let sel = _mm256_castps_si256(dfi_x_gt0);
        let mut q2 = _mm256_add_epi32(
            _mm256_add_epi32(q2_and, q2_and),
            _mm256_or_si256(
                _mm256_and_si256(sel, _mm256_set1_epi32(2)),
                _mm256_andnot_si256(sel, _mm256_set1_epi32(1)),
            ),
        );
        q2 = _mm256_srai_epi32(q2, 2);

        let odd = _mm256_cmpeq_epi32(
            _mm256_and_si256(q2_raw, _mm256_set1_epi32(1)),
            _mm256_set1_epi32(1),
        );
        let odd_f = _mm256_castsi256_ps(odd);
        let half_pi_hi = _mm256_set1_ps(3.1415927410125732422 * -0.5);
        let half_pi_lo = _mm256_set1_ps(-8.7422776573475857731e-08 * -0.5);
        let pi_adj = F2x8 {
            hi: vmulsign_avx(half_pi_hi, dfi.hi),
            lo: vmulsign_avx(half_pi_lo, dfi.hi),
        };
        let adj = df_add2_f2_f2_avx(dfi, pi_adj);
        let t_hi = _mm256_or_ps(
            _mm256_and_ps(odd_f, adj.hi),
            _mm256_andnot_ps(odd_f, dfi.hi),
        );
        let t_lo = _mm256_or_ps(
            _mm256_and_ps(odd_f, adj.lo),
            _mm256_andnot_ps(odd_f, dfi.lo),
        );
        let mut t = df_normalize_avx(F2x8 { hi: t_hi, lo: t_lo });

        let is_bad = _mm256_or_ps(
            _mm256_cmp_ps(abs_d, _mm256_set1_ps(f32::INFINITY), _CMP_EQ_OQ),
            _mm256_cmp_ps(d, d, _CMP_UNORD_Q),
        );
        t.hi = _mm256_or_ps(t.hi, is_bad);

        let in_range_i = _mm256_castps_si256(in_range);
        final_q = _mm256_or_si256(
            _mm256_and_si256(in_range_i, q),
            _mm256_andnot_si256(in_range_i, q2),
        );
        final_s = F2x8 {
            hi: _mm256_or_ps(
                _mm256_and_ps(in_range, s.hi),
                _mm256_andnot_ps(in_range, t.hi),
            ),
            lo: _mm256_or_ps(
                _mm256_and_ps(in_range, s.lo),
                _mm256_andnot_ps(in_range, t.lo),
            ),
        };
    }

    let t = final_s;
    let s2 = df_squ_f2_avx(final_s);

    let mut u = _mm256_set1_ps(2.6083159809786593541503e-06);
    u = _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(-0.0001981069071916863322258));
    u = _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(0.00833307858556509017944336));

    let inner = F2x8 {
        hi: _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(-0.166666597127914428710938)),
        lo: _mm256_setzero_ps(),
    };
    let x = df_add_f_f2_avx(
        _mm256_set1_ps(1.0),
        df_mul_f2_f2_avx(inner, s2),
    );

    let mut result = df_to_f_avx(t, x);

    let q_and_1 = _mm256_cmpeq_epi32(
        _mm256_and_si256(final_q, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    );
    let sign_flip = _mm256_and_ps(_mm256_castsi256_ps(q_and_1), neg_zero);
    result = _mm256_xor_ps(result, sign_flip);

    let is_neg_zero = visnegzero_avx(d);
    result = _mm256_or_ps(
        _mm256_and_ps(is_neg_zero, d),
        _mm256_andnot_ps(is_neg_zero, result),
    );

    result
}

// SLEEF xcosf_u1 for AVX2+FMA3
unsafe fn cosf_u1_avx2(d: __m256) -> __m256 {
    let neg_zero = _mm256_set1_ps(-0.0);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

    let dq = _mm256_fmadd_ps(
        _mm256_round_ps(
            _mm256_fmadd_ps(d, _mm256_set1_ps(M_1_PI_F), _mm256_set1_ps(-0.5)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
        ),
        _mm256_set1_ps(2.0),
        _mm256_set1_ps(1.0),
    );
    let q = _mm256_cvtps_epi32(dq);

    let mut s = df_add2_f_f_avx(d, _mm256_mul_ps(dq, _mm256_set1_ps(-PI_A2F * 0.5)));
    s = df_add2_f2_f_avx(s, _mm256_mul_ps(dq, _mm256_set1_ps(-PI_B2F * 0.5)));
    s = df_add2_f2_f_avx(s, _mm256_mul_ps(dq, _mm256_set1_ps(-PI_C2F * 0.5)));

    let abs_d = _mm256_and_ps(d, abs_mask);
    let in_range = _mm256_cmp_ps(abs_d, _mm256_set1_ps(TRIGRANGEMAX2F), _CMP_LT_OQ);
    let all_in_range = _mm256_movemask_ps(in_range) == 0xFF;

    let (mut final_s, mut final_q) = (s, q);

    if !all_in_range {
        let (dfi, q2_raw) = rempif_avx(d);
        let q2_and = _mm256_and_si256(q2_raw, _mm256_set1_epi32(3));
        let dfi_x_gt0 = _mm256_cmp_ps(dfi.hi, _mm256_setzero_ps(), _CMP_GT_OQ);
        let sel = _mm256_castps_si256(dfi_x_gt0);
        let mut q2 = _mm256_add_epi32(
            _mm256_add_epi32(q2_and, q2_and),
            _mm256_or_si256(
                _mm256_and_si256(sel, _mm256_set1_epi32(8)),
                _mm256_andnot_si256(sel, _mm256_set1_epi32(7)),
            ),
        );
        q2 = _mm256_srai_epi32(q2, 1);

        let even = _mm256_cmpeq_epi32(
            _mm256_and_si256(q2_raw, _mm256_set1_epi32(1)),
            _mm256_set1_epi32(0),
        );
        let even_f = _mm256_castsi256_ps(even);
        let y = _mm256_or_ps(
            _mm256_and_ps(dfi_x_gt0, _mm256_setzero_ps()),
            _mm256_andnot_ps(dfi_x_gt0, _mm256_set1_ps(-1.0)),
        );
        let half_pi_hi = _mm256_set1_ps(3.1415927410125732422 * -0.5);
        let half_pi_lo = _mm256_set1_ps(-8.7422776573475857731e-08 * -0.5);
        let pi_adj = F2x8 {
            hi: vmulsign_avx(half_pi_hi, y),
            lo: vmulsign_avx(half_pi_lo, y),
        };
        let adj = df_add2_f2_f2_avx(dfi, pi_adj);
        let t_hi = _mm256_or_ps(
            _mm256_and_ps(even_f, adj.hi),
            _mm256_andnot_ps(even_f, dfi.hi),
        );
        let t_lo = _mm256_or_ps(
            _mm256_and_ps(even_f, adj.lo),
            _mm256_andnot_ps(even_f, dfi.lo),
        );
        let mut t = df_normalize_avx(F2x8 { hi: t_hi, lo: t_lo });

        let is_bad = _mm256_or_ps(
            _mm256_cmp_ps(abs_d, _mm256_set1_ps(f32::INFINITY), _CMP_EQ_OQ),
            _mm256_cmp_ps(d, d, _CMP_UNORD_Q),
        );
        t.hi = _mm256_or_ps(t.hi, is_bad);

        let in_range_i = _mm256_castps_si256(in_range);
        final_q = _mm256_or_si256(
            _mm256_and_si256(in_range_i, q),
            _mm256_andnot_si256(in_range_i, q2),
        );
        final_s = F2x8 {
            hi: _mm256_or_ps(
                _mm256_and_ps(in_range, s.hi),
                _mm256_andnot_ps(in_range, t.hi),
            ),
            lo: _mm256_or_ps(
                _mm256_and_ps(in_range, s.lo),
                _mm256_andnot_ps(in_range, t.lo),
            ),
        };
    }

    let t = final_s;
    let s2 = df_squ_f2_avx(final_s);

    let mut u = _mm256_set1_ps(2.6083159809786593541503e-06);
    u = _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(-0.0001981069071916863322258));
    u = _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(0.00833307858556509017944336));

    let inner = F2x8 {
        hi: _mm256_fmadd_ps(u, s2.hi, _mm256_set1_ps(-0.166666597127914428710938)),
        lo: _mm256_setzero_ps(),
    };
    let x = df_add_f_f2_avx(
        _mm256_set1_ps(1.0),
        df_mul_f2_f2_avx(inner, s2),
    );

    let mut result = df_to_f_avx(t, x);

    let q_and_2_is_0 = _mm256_cmpeq_epi32(
        _mm256_and_si256(final_q, _mm256_set1_epi32(2)),
        _mm256_setzero_si256(),
    );
    let sign_flip = _mm256_and_ps(_mm256_castsi256_ps(q_and_2_is_0), neg_zero);
    result = _mm256_xor_ps(result, sign_flip);

    result
}

// SLEEF xexpf for AVX2+FMA3
unsafe fn expf_avx2(d: __m256) -> __m256 {
    let q = _mm256_cvtps_epi32(_mm256_mul_ps(d, _mm256_set1_ps(R_LN2F)));
    let qf = _mm256_cvtepi32_ps(q);

    let mut s = _mm256_fmadd_ps(qf, _mm256_set1_ps(-L2UF), d);
    s = _mm256_fmadd_ps(qf, _mm256_set1_ps(-L2LF), s);

    let mut u = _mm256_set1_ps(0.000198527617612853646278381);
    u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(0.00139304355252534151077271));
    u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(0.00833336077630519866943359));
    u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(0.0416664853692054748535156));
    u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(0.166666671633720397949219));
    u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(0.5));

    u = _mm256_add_ps(
        _mm256_set1_ps(1.0),
        _mm256_fmadd_ps(_mm256_mul_ps(s, s), u, s),
    );

    u = vldexp2_avx(u, q);

    u = _mm256_andnot_ps(
        _mm256_cmp_ps(d, _mm256_set1_ps(-104.0), _CMP_LT_OQ),
        u,
    );
    u = _mm256_or_ps(
        _mm256_and_ps(
            _mm256_cmp_ps(_mm256_set1_ps(100.0), d, _CMP_LT_OQ),
            _mm256_set1_ps(f32::INFINITY),
        ),
        _mm256_andnot_ps(
            _mm256_cmp_ps(_mm256_set1_ps(100.0), d, _CMP_LT_OQ),
            u,
        ),
    );

    u
}

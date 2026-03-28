# simd_vector

SIMD vector types for x86-64 in pure stable Rust.

- `Vec4` — 4×f32, SSE4.1 + FMA3
- `Vec8` — 8×f32, AVX2 + FMA3

## Features

- Arithmetic: `+`, `-`, `*`, `/` (vec×vec, vec×f32, f32×vec)
- `splat`, `abs`, `neg`, `sqrt`, `floor`, `ceil`, `round`, `mul_add` (FMA)
- `Sum`, `Index`, `From`/`Into` array, `Clone`, `Copy`, `Debug`, `PartialEq`
- Precision-selected via module import:
  - **Precise** (`use simd_vector::precise::*`): `sin`, `cos`, `exp`, `sum`, `dot` — ≤ 1.0 ULP (f64 reductions)
  - **Fast** (`use simd_vector::fast::*`): `sin`, `cos`, `exp`, `sum`, `dot` — ≤ 3.5 ULP transcendentals, f32 reductions

## Precision modules

Select precision by importing the corresponding module. Both modules export the same method names — precision is chosen at import time, not by calling different functions.

```rust
// Precise path: sin, cos, exp (≤ 1.0 ULP), sum, dot (≤ 0.5 ULP via f64)
use simd_vector::precise::*;

let v = Vec4([1.0, 2.0, 3.0, 4.0]);
let s = v.sin();  // ≤ 1.0 ULP
let c = v.cos();  // ≤ 1.0 ULP
let e = v.exp();  // ≤ 1.0 ULP
let total = v.sum();       // f64 intermediate, ≤ 0.5 ULP
let d = v.dot(v);          // f64 intermediate, ≤ 0.5 ULP
```

```rust
// Fast path: sin, cos (≤ 3.5 ULP), exp (≤ 1.0 ULP), sum, dot (f32 native)
use simd_vector::fast::*;

let v = Vec4([1.0, 2.0, 3.0, 4.0]);
let s = v.sin();  // ≤ 3.5 ULP
let c = v.cos();  // ≤ 3.5 ULP
let e = v.exp();  // ≤ 1.0 ULP (same impl as precise)
let total = v.sum();       // f32 horizontal add
let d = v.dot(v);          // f32 multiply + horizontal add
```

## Accuracy

| Function | ULP error | Notes |
|----------|-----------|-------|
| `+`, `-`, `*`, `/` | 0.0 | IEEE 754 correctly rounded |
| `neg` | 0.0 | exact bit flip |
| `abs` | 0.0 | exact bit mask |
| `floor` | 0.0 | exact (IEEE 754 `roundps`) |
| `ceil` | 0.0 | exact (IEEE 754 `roundps`) |
| `round` | 0.0 | exact (half away from zero, matches `f32::round`) |
| `splat` | 0.0 | exact broadcast |
| `mul_add` | ≤ 0.5 | single FMA instruction |
| `sqrt` | 0.0 | IEEE 754 correctly rounded (`sqrtps`) |
| `precise::sum` | ≤ 0.5 | f64 intermediate: exact tree sum, single rounding on f64→f32 |
| `precise::dot` | ≤ 0.5 | f64 intermediate: exact products + tree sum, single rounding on f64→f32 |
| `precise::sin` | ≤ 1.0 | SLEEF `xsinf_u1` — Cody-Waite + Payne-Hanek + double-float polynomial |
| `precise::cos` | ≤ 1.0 | SLEEF `xcosf_u1` — Cody-Waite + Payne-Hanek + double-float polynomial |
| `precise::exp` | ≤ 1.0 | SLEEF `xexpf` — ln(2) range reduction + degree-6 polynomial + ldexp |
| `fast::sum` | ≤ N/2 | f32 horizontal add (N = lane count) |
| `fast::dot` | ≤ N/2 | f32 multiply + horizontal add (N = lane count) |
| `fast::sin` | ≤ 3.5 | SLEEF `xsinf` — 3-tier range reduction + scalar polynomial |
| `fast::cos` | ≤ 3.5 | SLEEF `xcosf` — 3-tier range reduction + scalar polynomial |
| `fast::exp` | ≤ 1.0 | same as `precise::exp` (SLEEF has no reduced-precision variant) |

### Implementation details

- **Precise `sum`/`dot`**: widen f32→f64, accumulate in f64, single f64→f32 rounding at the end
- **Fast `sum`/`dot`**: native f32 horizontal add (`movehdup`/`movehl`/`addps` shuffle tree)
- **Precise `sin`/`cos`**: Cody-Waite (3-constant) for |x| < 125, Payne-Hanek table-based for larger arguments, double-float arithmetic for polynomial evaluation
- **Fast `sin`/`cos`**: 3-tier range reduction — 3-part Cody-Waite (|x| < 125), 4-part Cody-Waite (|x| < 39000), Payne-Hanek (|x| ≥ 39000) — scalar polynomial, no double-float pairs
- **Edge cases**: NaN/Inf propagation, sin(-0) = -0, exp(-inf) = 0, exp(inf) = inf
- Polynomial coefficients, constants, and the 416-entry `Sleef_rempitabsp` table are taken directly from [SLEEF](https://github.com/shibatch/sleef)

## Required CPU features

SSE4.1, AVX2, FMA3. Enabled globally via `.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+sse4.1,+avx2,+fma"]
```

## Installation

```
cargo add simd_vector
```

## Usage

```rust
use simd_vector::precise::*;

let a = Vec4([1.0, 2.0, 3.0, 4.0]);
let b = Vec4([5.0, 6.0, 7.0, 8.0]);
let c = a + b;              // [6.0, 8.0, 10.0, 12.0]
let d = a.dot(b);           // 70.0
let s = a.sin();            // [0.8415, 0.9093, 0.1411, -0.7568]
let e = Vec8::splat(1.0).exp(); // [2.71828, 2.71828, ...] (8 lanes)
```

## Tests

453 tests covering all operations, edge cases (NaN, Inf, -0.0, subnormals), sampled ULP sweep verification, Sollya-verified reference values, trigonometric identities, arithmetic properties, and AVX2 lane independence.

```
cargo test
```

## License

MIT

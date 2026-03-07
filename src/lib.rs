#![allow(unsafe_op_in_unsafe_fn)]

mod vec4;
mod vec8;

pub use vec4::Vec4;
pub use vec8::Vec8;

// SLEEF constants for single-precision trig range reduction (u10 variants)
const PI_A2F: f32 = 3.1414794921875;
const PI_B2F: f32 = 0.00011315941810607910156;
const PI_C2F: f32 = 1.9841872589410058936e-09;
const TRIGRANGEMAX2F: f32 = 125.0;
const M_1_PI_F: f32 = std::f32::consts::FRAC_1_PI;

// SLEEF constants for exp
const R_LN2F: f32 = 1.442695040888963407359924681001892137426645954152985934135449406931;
const L2UF: f32 = 0.693145751953125;
const L2LF: f32 = 1.428606765330187045e-06;

// SLEEF Payne-Hanek table for single-precision pi reduction
#[allow(clippy::excessive_precision)]
const REMPI_TABSP: [f32; 416] = [
    f32::from_bits(0x3E22F980),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x3D0BE60C),
    f32::from_bits(0x31DC9C88),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x3D0BE60C),
    f32::from_bits(0x31DC9C88),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x3B3E60DC),
    f32::from_bits(0xAED8DDF4),
    f32::from_bits(0xA33580F4),
    f32::from_bits(0x980A82E1),
    f32::from_bits(0x3B3E60DC),
    f32::from_bits(0xAED8DDF4),
    f32::from_bits(0xA33580F4),
    f32::from_bits(0x980A82E1),
    f32::from_bits(0x3B3E60DC),
    f32::from_bits(0xAED8DDF4),
    f32::from_bits(0xA33580F4),
    f32::from_bits(0x980A82E1),
    f32::from_bits(0x3B3E60DC),
    f32::from_bits(0xAED8DDF4),
    f32::from_bits(0xA33580F4),
    f32::from_bits(0x980A82E1),
    f32::from_bits(0x3A79836C),
    f32::from_bits(0x2F139104),
    f32::from_bits(0x23A53F84),
    f32::from_bits(0x17EAFA3F),
    f32::from_bits(0x3A79836C),
    f32::from_bits(0x2F139104),
    f32::from_bits(0x23A53F84),
    f32::from_bits(0x17EAFA3F),
    f32::from_bits(0x39F306DC),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E0B),
    f32::from_bits(0x39660DB8),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E0B),
    f32::from_bits(0x38CC1B70),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E0B),
    f32::from_bits(0x381836E4),
    f32::from_bits(0x2C644150),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3EA),
    f32::from_bits(0x36C1B724),
    f32::from_bits(0x2BC882A4),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FAA),
    f32::from_bits(0x36C1B724),
    f32::from_bits(0x2BC882A4),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FAA),
    f32::from_bits(0x36C1B724),
    f32::from_bits(0x2BC882A4),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FAA),
    f32::from_bits(0x36036E4C),
    f32::from_bits(0x2B110548),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FAA),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x335B9390),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x32B72720),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6F),
    f32::from_bits(0x31DC9C88),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x31DC9C88),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x31393910),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x3064E440),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x3064E440),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x2FC9C880),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x2F139104),
    f32::from_bits(0x23A53F84),
    f32::from_bits(0x17EAFA3C),
    f32::from_bits(0x0CA9A6EE),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E08),
    f32::from_bits(0x8B32C890),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E08),
    f32::from_bits(0x8B32C890),
    f32::from_bits(0x2D9C8828),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E08),
    f32::from_bits(0x8B32C890),
    f32::from_bits(0x2C644150),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x2C644150),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x2C644150),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x2BC882A4),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FA8),
    f32::from_bits(0x09537703),
    f32::from_bits(0x2B110548),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FA8),
    f32::from_bits(0x09537703),
    f32::from_bits(0x29882A54),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x29882A54),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x29882A54),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x2782A540),
    f32::from_bits(0x9B762A0C),
    f32::from_bits(0x0EFA9A6C),
    f32::from_bits(0x03B81B6C),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x24A94FE0),
    f32::from_bits(0x191D5F48),
    f32::from_bits(0x8C2CB224),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x23A53F84),
    f32::from_bits(0x17EAFA3C),
    f32::from_bits(0x0CA9A6EC),
    f32::from_bits(0x0181B6C5),
    f32::from_bits(0x23A53F84),
    f32::from_bits(0x17EAFA3C),
    f32::from_bits(0x0CA9A6EC),
    f32::from_bits(0x0181B6C5),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E08),
    f32::from_bits(0x8B32C890),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x2294FE14),
    f32::from_bits(0x96282E08),
    f32::from_bits(0x8B32C890),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x2127F09C),
    f32::from_bits(0x15AFA3E8),
    f32::from_bits(0x0A9A6EE0),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FA8),
    f32::from_bits(0x09537700),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x201FC274),
    f32::from_bits(0x14BE8FA8),
    f32::from_bits(0x09537700),
    f32::from_bits(0x0006DB15),
    f32::from_bits(0x1EFE13AC),
    f32::from_bits(0x91382B2C),
    f32::from_bits(0x8508FC90),
    f32::from_bits(0x800004EB),
    f32::from_bits(0x1EFE13AC),
    f32::from_bits(0x91382B2C),
    f32::from_bits(0x8508FC90),
    f32::from_bits(0x800004EB),
    f32::from_bits(0x1EFE13AC),
    f32::from_bits(0x91382B2C),
    f32::from_bits(0x8508FC90),
    f32::from_bits(0x800004EB),
    f32::from_bits(0x1E7C2758),
    f32::from_bits(0x91382B2C),
    f32::from_bits(0x8508FC90),
    f32::from_bits(0x800004EB),
    f32::from_bits(0x1DF84EB0),
    f32::from_bits(0x91382B2C),
    f32::from_bits(0x8508FC90),
    f32::from_bits(0x800004EB),
    f32::from_bits(0x3D709D5C),
    f32::from_bits(0x3251F534),
    f32::from_bits(0x265DC0D8),
    f32::from_bits(0x1B58A566),
    f32::from_bits(0x3CE13ABC),
    f32::from_bits(0x31A3EA68),
    f32::from_bits(0x265DC0D8),
    f32::from_bits(0x1B58A566),
    f32::from_bits(0x3C42757C),
    f32::from_bits(0x308FA9A4),
    f32::from_bits(0x25BB81B4),
    f32::from_bits(0x1AB14ACD),
    f32::from_bits(0x3B84EAF8),
    f32::from_bits(0x308FA9A4),
    f32::from_bits(0x25BB81B4),
    f32::from_bits(0x1AB14ACD),
    f32::from_bits(0x391D5F48),
    f32::from_bits(0xAC2CB224),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x391D5F48),
    f32::from_bits(0xAC2CB224),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x391D5F48),
    f32::from_bits(0xAC2CB224),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x391D5F48),
    f32::from_bits(0xAC2CB224),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x391D5F48),
    f32::from_bits(0xAC2CB224),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x37EAFA3C),
    f32::from_bits(0x2CA9A6EC),
    f32::from_bits(0x2181B6C4),
    f32::from_bits(0x1615993C),
    f32::from_bits(0x37EAFA3C),
    f32::from_bits(0x2CA9A6EC),
    f32::from_bits(0x2181B6C4),
    f32::from_bits(0x1615993C),
    f32::from_bits(0x37EAFA3C),
    f32::from_bits(0x2CA9A6EC),
    f32::from_bits(0x2181B6C4),
    f32::from_bits(0x1615993C),
    f32::from_bits(0x3755F47C),
    f32::from_bits(0x2BA69BB8),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x36ABE8F8),
    f32::from_bits(0x2BA69BB8),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x35AFA3E8),
    f32::from_bits(0x2A9A6EE0),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x35AFA3E8),
    f32::from_bits(0x2A9A6EE0),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x34BE8FA8),
    f32::from_bits(0x29537700),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x34BE8FA8),
    f32::from_bits(0x29537700),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x33FA3EA4),
    f32::from_bits(0x28A6EE04),
    f32::from_bits(0x1DB6C528),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x33FA3EA4),
    f32::from_bits(0x28A6EE04),
    f32::from_bits(0x1DB6C528),
    f32::from_bits(0x12CC9E22),
    f32::from_bits(0x33747D4C),
    f32::from_bits(0x279BB818),
    f32::from_bits(0x1CDB14AC),
    f32::from_bits(0x10C9E21D),
    f32::from_bits(0x32E8FA98),
    f32::from_bits(0x279BB818),
    f32::from_bits(0x1CDB14AC),
    f32::from_bits(0x10C9E21D),
    f32::from_bits(0x3251F534),
    f32::from_bits(0x265DC0D8),
    f32::from_bits(0x1B58A564),
    f32::from_bits(0x1013C439),
    f32::from_bits(0x31A3EA68),
    f32::from_bits(0x265DC0D8),
    f32::from_bits(0x1B58A564),
    f32::from_bits(0x1013C439),
    f32::from_bits(0x308FA9A4),
    f32::from_bits(0x25BB81B4),
    f32::from_bits(0x1AB14ACC),
    f32::from_bits(0x0E9E21C8),
    f32::from_bits(0x308FA9A4),
    f32::from_bits(0x25BB81B4),
    f32::from_bits(0x1AB14ACC),
    f32::from_bits(0x0E9E21C8),
    f32::from_bits(0x2EFA9A6C),
    f32::from_bits(0x23B81B6C),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2EFA9A6C),
    f32::from_bits(0x23B81B6C),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2EFA9A6C),
    f32::from_bits(0x23B81B6C),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2EFA9A6C),
    f32::from_bits(0x23B81B6C),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2E7534DC),
    f32::from_bits(0x22E06DB0),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2DEA69BC),
    f32::from_bits(0xA17C9274),
    f32::from_bits(0x95D4CD84),
    f32::from_bits(0x8ADE37DF),
    f32::from_bits(0x2D54D374),
    f32::from_bits(0x2240DB60),
    f32::from_bits(0x1725664C),
    f32::from_bits(0x0C443904),
    f32::from_bits(0x2CA9A6EC),
    f32::from_bits(0x2181B6C4),
    f32::from_bits(0x1615993C),
    f32::from_bits(0x09872084),
    f32::from_bits(0x2BA69BB8),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E20),
    f32::from_bits(0x07641080),
    f32::from_bits(0x2BA69BB8),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E20),
    f32::from_bits(0x07641080),
    f32::from_bits(0x2A9A6EE0),
    f32::from_bits(0x1E5B6294),
    f32::from_bits(0x12CC9E20),
    f32::from_bits(0x07641080),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
];

/// Scalar Payne-Hanek pi reduction for a single `f32`.
///
/// Returns `(reduced_hi, reduced_lo, quadrant)` where `reduced_hi + reduced_lo`
/// is `a` reduced modulo pi/2, and `quadrant` encodes which quadrant of the
/// original angle was occupied.
fn rempif_scalar(a: f32) -> (f32, f32, i32) {
    // vilogb2k: extract exponent
    let mut ex = ((a.to_bits() >> 23) & 0xFF) as i32 - 0x7F;
    ex -= 25;
    let mut a = a;

    // Scale down very large inputs (original exponent > 90) to prevent
    // f32 intermediate precision loss: without this, a * table[idx] products
    // are so large that rempisubf cannot extract quadrant/fractional bits.
    if ex > 90 - 25 {
        // vldexp3: add -64 to exponent field
        a = f32::from_bits((a.to_bits() as i32 + (-64i32 << 23)) as u32);
    }
    // Clamp negative to 0: vandnot with arithmetic right shift
    if ex < 0 {
        ex = 0;
    }
    let idx = (ex as usize) * 4;

    // First multiply
    let (mut x_hi, mut x_lo) = df_mul_f_f(a, REMPI_TABSP[idx]);
    // rempisubf: extract quadrant
    let y = (x_hi * 4.0).round();
    let mut q = (y - (x_hi.round()) * 4.0) as i32;
    x_hi -= y * 0.25;
    let (x_hi2, x_lo2) = df_normalize(x_hi, x_lo);
    x_hi = x_hi2;
    x_lo = x_lo2;

    // Second multiply
    let (y_hi, y_lo) = df_mul_f_f(a, REMPI_TABSP[idx + 1]);
    let (x_hi2, x_lo2) = df_add2_f2_f2(x_hi, x_lo, y_hi, y_lo);
    x_hi = x_hi2;
    x_lo = x_lo2;

    let y2 = (x_hi * 4.0).round();
    q += (y2 - (x_hi.round()) * 4.0) as i32;
    x_hi -= y2 * 0.25;
    let (x_hi2, x_lo2) = df_normalize(x_hi, x_lo);
    x_hi = x_hi2;
    x_lo = x_lo2;

    // Third and fourth multiply
    let (y_hi, y_lo) = (REMPI_TABSP[idx + 2], REMPI_TABSP[idx + 3]);
    let (y_hi2, y_lo2) = df_mul_f2_f(y_hi, y_lo, a);
    let (x_hi2, x_lo2) = df_add2_f2_f2(x_hi, x_lo, y_hi2, y_lo2);
    x_hi = x_hi2;
    x_lo = x_lo2;
    let (x_hi2, x_lo2) = df_normalize(x_hi, x_lo);
    x_hi = x_hi2;
    x_lo = x_lo2;

    // Multiply by 2*pi
    const TWO_PI_HI: f32 = 3.1415927410125732422 * 2.0;
    const TWO_PI_LO: f32 = -8.7422776573475857731e-08 * 2.0;
    let (r_hi, r_lo) = df_mul_f2_f2(x_hi, x_lo, TWO_PI_HI, TWO_PI_LO);

    // For very small inputs, return a directly
    if a.abs() < 0.7 {
        return (a, 0.0, 0);
    }

    (r_hi, r_lo, q)
}

/// Normalizes a double-float pair so that `hi` carries the leading bits.
///
/// Returns `(s, err)` where `s = a_hi + a_lo` and `err` captures the rounding error.
#[inline(always)]
fn df_normalize(a_hi: f32, a_lo: f32) -> (f32, f32) {
    let s = a_hi + a_lo;
    (s, (a_hi - s) + a_lo)
}

/// Error-free addition of two double-float pairs: `(a_hi, a_lo) + (b_hi, b_lo)`.
#[inline(always)]
fn df_add2_f2_f2(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> (f32, f32) {
    let s = a_hi + b_hi;
    let v = s - a_hi;
    let t = (a_hi - (s - v)) + (b_hi - v);
    (s, t + a_lo + b_lo)
}

/// Error-free multiplication of two `f32` values using FMA. Returns `(product, error)`.
#[inline(always)]
fn df_mul_f_f(a: f32, b: f32) -> (f32, f32) {
    let s = a * b;
    let t = a.mul_add(b, -s);
    (s, t)
}

/// Multiplies a double-float pair `(a_hi, a_lo)` by a single `f32`.
#[inline(always)]
fn df_mul_f2_f(a_hi: f32, a_lo: f32, b: f32) -> (f32, f32) {
    let s = a_hi * b;
    let t = a_hi.mul_add(b, -s) + a_lo * b;
    (s, t)
}

/// Multiplies two double-float pairs: `(a_hi, a_lo) * (b_hi, b_lo)`.
#[inline(always)]
fn df_mul_f2_f2(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> (f32, f32) {
    let s = a_hi * b_hi;
    let t = a_hi.mul_add(b_hi, -s) + a_hi * b_lo + a_lo * b_hi;
    (s, t)
}

#[cfg(test)]
mod tests;

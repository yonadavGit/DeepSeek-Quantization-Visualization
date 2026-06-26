from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NumberFormat:
    name: str
    total_bits: int
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    kind: str
    use: str
    note: str

    @property
    def value_bits(self):
        return self.total_bits - self.sign_bits


FORMATS = {
    "FP32": NumberFormat(
        "FP32",
        32,
        1,
        8,
        23,
        "float",
        "Master weights, optimizer states, and sensitive accumulated values.",
        "Large dynamic range and high precision; expensive for memory bandwidth.",
    ),
    "FP16": NumberFormat(
        "FP16",
        16,
        1,
        5,
        10,
        "float",
        "Common reduced-precision baseline, but less robust than BF16 for training.",
        "More mantissa than BF16, but much less exponent range than FP32/BF16.",
    ),
    "BF16": NumberFormat(
        "BF16",
        16,
        1,
        8,
        7,
        "float",
        "Inter-layer activations and sensitive modules in DeepSeek's framework.",
        "Keeps FP32's exponent width, so overflow is much less likely than FP16.",
    ),
    "INT8": NumberFormat(
        "INT8",
        8,
        1,
        0,
        7,
        "int",
        "Useful mental model for scaling: map a tensor range onto -127..127.",
        "No exponent or mantissa; decimal detail must come from an external scale.",
    ),
    "FP8 E4M3": NumberFormat(
        "FP8 E4M3",
        8,
        1,
        4,
        3,
        "float",
        "DeepSeek's preferred FP8 format for forward and backward GEMM inputs.",
        "More mantissa precision than E5M2; fine-grained scaling manages local range.",
    ),
    "FP8 E5M2": NumberFormat(
        "FP8 E5M2",
        8,
        1,
        5,
        2,
        "float",
        "Conventional choice for gradients when extra dynamic range is needed.",
        "Wider range but coarser spacing because it spends one bit on exponent.",
    ),
}


def exponent_bias(exponent_bits):
    return 2 ** (exponent_bits - 1) - 1


def float_range(fmt: NumberFormat):
    bias = exponent_bias(fmt.exponent_bits)
    min_exp = 1 - bias
    max_exp = (2**fmt.exponent_bits - 2) - bias
    min_normal = 2.0**min_exp
    max_normal = (2.0 - 2.0 ** (-fmt.mantissa_bits)) * (2.0**max_exp)
    return min_normal, max_normal


def quantize_float(value, fmt: NumberFormat):
    if value == 0:
        return {
            "represented": 0.0,
            "sign": 0,
            "exponent_unbiased": 0,
            "exponent_stored": 0,
            "mantissa_int": 0,
            "overflow": False,
            "underflow": False,
        }

    sign = 1 if value < 0 else 0
    abs_value = abs(float(value))
    min_normal, max_normal = float_range(fmt)
    overflow = abs_value > max_normal
    underflow = abs_value < min_normal
    clipped = min(max(abs_value, min_normal), max_normal)

    exponent = int(np.floor(np.log2(clipped)))
    significand = clipped / (2.0**exponent)
    mantissa_scale = 2**fmt.mantissa_bits
    mantissa_int = int(np.round((significand - 1.0) * mantissa_scale))

    if mantissa_int == mantissa_scale:
        mantissa_int = 0
        exponent += 1

    bias = exponent_bias(fmt.exponent_bits)
    max_stored = 2**fmt.exponent_bits - 2
    exponent_stored = int(np.clip(exponent + bias, 1, max_stored))
    exponent = exponent_stored - bias
    represented_abs = (1.0 + mantissa_int / mantissa_scale) * (2.0**exponent)
    represented = -represented_abs if sign else represented_abs

    return {
        "represented": represented,
        "sign": sign,
        "exponent_unbiased": exponent,
        "exponent_stored": exponent_stored,
        "mantissa_int": mantissa_int,
        "overflow": overflow,
        "underflow": underflow,
    }


def quantize_int8(value):
    q = int(np.clip(np.round(value), -127, 127))
    return {
        "represented": float(q),
        "sign": 1 if q < 0 else 0,
        "exponent_unbiased": None,
        "exponent_stored": None,
        "mantissa_int": abs(q),
        "overflow": abs(value) > 127,
        "underflow": False,
    }


def represent_value(value, fmt: NumberFormat):
    if fmt.kind == "int":
        return quantize_int8(value)
    return quantize_float(value, fmt)


def bit_segments(fmt: NumberFormat, representation):
    if fmt.kind == "int":
        value_bits = format(int(representation["mantissa_int"]), f"0{fmt.value_bits}b")[-fmt.value_bits :]
        return [("Sign", str(representation["sign"]), "#2f6f73")] + [
            ("Integer value", bit, "#4776a8") for bit in value_bits
        ]

    exp = format(int(representation["exponent_stored"]), f"0{fmt.exponent_bits}b")
    man = format(int(representation["mantissa_int"]), f"0{fmt.mantissa_bits}b")
    return (
        [("Sign", str(representation["sign"]), "#2f6f73")]
        + [("Exponent", bit, "#4776a8") for bit in exp]
        + [("Mantissa", bit, "#e0855b") for bit in man]
    )


def decoded_formula(fmt: NumberFormat, representation):
    if fmt.kind == "int":
        sign_factor = -1 if representation["sign"] else 1
        magnitude = int(representation["mantissa_int"])
        return f"({sign_factor}) x {magnitude} = {representation['represented']:.8g}"

    sign_factor = -1 if representation["sign"] else 1
    mantissa_denominator = 2**fmt.mantissa_bits
    mantissa_int = int(representation["mantissa_int"])
    exponent = int(representation["exponent_unbiased"])
    significand = 1.0 + mantissa_int / mantissa_denominator
    return (
        f"({sign_factor}) x (1 + {mantissa_int}/{mantissa_denominator}) "
        f"x 2^{exponent} = ({sign_factor}) x {significand:.6g} "
        f"x {2.0**exponent:.6g} = {representation['represented']:.8g}"
    )


def nearby_float_values(fmt: NumberFormat, center, count=12):
    if fmt.kind == "int":
        lo = int(max(-127, np.floor(center) - count))
        hi = int(min(127, np.ceil(center) + count))
        return np.arange(lo, hi + 1, dtype=float)

    center = max(abs(center), float_range(fmt)[0])
    exponent = int(np.floor(np.log2(center)))
    exponents = range(exponent - 2, exponent + 3)
    vals = [0.0]
    for exp in exponents:
        for m in range(2**fmt.mantissa_bits):
            vals.append((1.0 + m / (2**fmt.mantissa_bits)) * (2.0**exp))
    vals = np.array(sorted(set(vals)))
    vals = vals[np.isfinite(vals)]
    return np.concatenate((-vals[::-1], vals))

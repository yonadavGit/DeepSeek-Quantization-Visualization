import numpy as np


def quantize_to_int8(x, qmax=127):
    x = np.asarray(x, dtype=float)
    alpha = float(np.max(np.abs(x))) if x.size else 0.0
    scale = qmax / alpha if alpha else 1.0
    scaled = x * scale
    q = np.clip(np.round(scaled), -qmax, qmax)
    x_hat = q / scale
    error = x_hat - x
    return {
        "x": x,
        "alpha": alpha,
        "scale": scale,
        "scaled": scaled,
        "q": q,
        "x_hat": x_hat,
        "error": error,
        "qmax": qmax,
    }


def int8_bits(q):
    q = int(q)
    sign = 1 if q < 0 else 0
    magnitude = min(abs(q), 127)
    return f"{sign} " + " ".join(format(magnitude, "07b"))


def quantization_formula(x, alpha, scaled, q):
    return f"round({x:g} * 127 / {alpha:g}) = round({scaled:.4g}) = {int(q)}"


def dequantization_formula(q, scale, x_hat):
    return f"{int(q)} / {scale:.6g} = {x_hat:.6g}"

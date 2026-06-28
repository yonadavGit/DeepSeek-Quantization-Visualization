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


def quantize_groupwise_to_int8(x, group_size, qmax=127):
    x = np.asarray(x, dtype=float)
    group_size = max(1, int(group_size))
    q = np.zeros_like(x)
    x_hat = np.zeros_like(x)
    scales = []
    alphas = []
    groups = np.zeros_like(x, dtype=int)

    for group_id, start in enumerate(range(0, len(x), group_size)):
        end = min(start + group_size, len(x))
        result = quantize_to_int8(x[start:end], qmax=qmax)
        q[start:end] = result["q"]
        x_hat[start:end] = result["x_hat"]
        scales.append(result["scale"])
        alphas.append(result["alpha"])
        groups[start:end] = group_id

    return {
        "x": x,
        "q": q,
        "x_hat": x_hat,
        "error": x_hat - x,
        "abs_error": np.abs(x_hat - x),
        "scales": np.array(scales),
        "alphas": np.array(alphas),
        "groups": groups,
        "group_size": group_size,
        "qmax": qmax,
    }


def quantize_blockwise_to_int8(matrix, block_size, qmax=127):
    matrix = np.asarray(matrix, dtype=float)
    block_size = max(1, int(block_size))
    q = np.zeros_like(matrix)
    reconstructed = np.zeros_like(matrix)
    block_rows = int(np.ceil(matrix.shape[0] / block_size))
    block_cols = int(np.ceil(matrix.shape[1] / block_size))
    scales = np.zeros((block_rows, block_cols))
    alphas = np.zeros((block_rows, block_cols))

    for bi, row_start in enumerate(range(0, matrix.shape[0], block_size)):
        for bj, col_start in enumerate(range(0, matrix.shape[1], block_size)):
            row_end = min(row_start + block_size, matrix.shape[0])
            col_end = min(col_start + block_size, matrix.shape[1])
            result = quantize_to_int8(matrix[row_start:row_end, col_start:col_end], qmax=qmax)
            q[row_start:row_end, col_start:col_end] = result["q"]
            reconstructed[row_start:row_end, col_start:col_end] = result["x_hat"]
            scales[bi, bj] = result["scale"]
            alphas[bi, bj] = result["alpha"]

    return {
        "x": matrix,
        "q": q,
        "x_hat": reconstructed,
        "error": reconstructed - matrix,
        "abs_error": np.abs(reconstructed - matrix),
        "scales": scales,
        "alphas": alphas,
        "block_size": block_size,
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

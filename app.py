import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import (
    PAGES,
    all_bit_layouts_figure,
    bit_layout_figure,
    formula,
    inject_css,
    number_line_figure,
    quantization_mapping_figure,
)
from quantization import (
    dequantization_formula,
    int8_bits,
    quantization_formula,
    quantize_blockwise_to_int8,
    quantize_groupwise_to_int8,
    quantize_to_int8,
)
from quant_formats import FORMATS, bit_segments, decoded_formula, float_range, nearby_float_values, represent_value


st.set_page_config(page_title="DeepSeek Quantization Visualizer", page_icon="8", layout="wide")
inject_css()


def joined_bits(fmt, rep):
    return " ".join(bit for _, bit, _ in bit_segments(fmt, rep))


PRECISION_COLORS = {
    "FP32": "#2f7d51",
    "BF16": "#4776a8",
    "FP8": "#d91b61",
    "FP8 E4M3": "#d91b61",
}


def chip(name):
    color = PRECISION_COLORS[name]
    return f'<span class="precision-chip" style="background:{color}">{name}</span>'


def quant_stepper(max_step):
    key = "quantization_step"
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = max(0, min(st.session_state[key], max_step))
    c1, c2, c3, c4 = st.columns([1, 1, 1, 5])
    with c1:
        if st.button("Previous", key="quant_prev", disabled=st.session_state[key] <= 0):
            st.session_state[key] -= 1
    with c2:
        if st.button("Next", key="quant_next", disabled=st.session_state[key] >= max_step):
            st.session_state[key] += 1
    with c3:
        if st.button("Reset", key="quant_reset"):
            st.session_state[key] = 0
    with c4:
        st.progress((st.session_state[key] + 1) / (max_step + 1))
    return st.session_state[key]


def render_quantization_panel():
    st.header("Actual Quantization: Scaling FP32 Values into INT8 Codes")
    st.write(
        "This mirrors the chapter's scaling diagram: find the largest absolute value alpha, "
        "map the original FP32 range `[-alpha, +alpha]` onto the INT8 range `[-127, +127]`, "
        "round to integer codes, then invert the scale to reconstruct approximate FP32 values."
    )

    vector_text = st.text_input(
        "Example FP32 vector",
        value="5.47, 3.08, -7.59, 0, -1.95, -4.57, 10.8",
        help="Edit the vector and step through the scaling process again.",
    )
    try:
        x = [float(part.strip()) for part in vector_text.split(",") if part.strip()]
    except ValueError:
        st.error("Use comma-separated numbers, for example: 5.47, 3.08, -7.59, 0, -1.95, -4.57, 10.8")
        return
    if not x:
        st.error("Enter at least one value.")
        return

    result = quantize_to_int8(x)
    focus_labels = [f"x[{i}] = {value:g}" for i, value in enumerate(result["x"])]
    focus_idx = st.selectbox(
        "Focus calculation shown on the plot",
        options=list(range(len(focus_labels))),
        format_func=lambda idx: focus_labels[idx],
        index=min(len(focus_labels) - 1, 2),
    )
    steps = [
        "Start with the original FP32 vector and its bit representation.",
        "Find alpha = max(abs(x)), the highest absolute value.",
        "Compute scale = 127 / alpha to map the FP32 range to INT8.",
        "Map every FP32 value into the INT8 coordinate system.",
        "Round each scaled value to the nearest INT8 code.",
        "Apply the inverse transform: x_hat = q / scale.",
        "Compare the reconstructed value with the original.",
    ]
    step = quant_stepper(len(steps) - 1)
    st.caption(f"Step {step + 1}: {steps[step]}")

    formula("alpha = max(abs(x)); scale = 127 / alpha; q = round(x * scale); x_hat = q / scale; error = x_hat - x")

    m1, m2, m3 = st.columns(3)
    m1.metric("alpha", f"{result['alpha']:.6g}" if step >= 1 else "not chosen yet")
    m2.metric("scale", f"{result['scale']:.6g}" if step >= 2 else "not computed yet")
    m3.metric("target range", "INT8: -127..127")

    st.plotly_chart(
        quantization_mapping_figure(result, step, focus_idx),
        width="stretch",
        key=f"quant_map_{step}_{focus_idx}_{vector_text}",
    )

    rows = []
    for i, value in enumerate(result["x"]):
        fp32_rep = represent_value(value, FORMATS["FP32"])
        row = {
            "index": i,
            "original x": value,
            "FP32 bits": joined_bits(FORMATS["FP32"], fp32_rep),
        }
        if step >= 3:
            row["x * scale"] = result["scaled"][i]
        if step >= 4:
            row["q calculation"] = quantization_formula(
                value,
                result["alpha"],
                result["scaled"][i],
                result["q"][i],
            )
            row["INT8 q"] = int(result["q"][i])
            row["INT8 bits"] = int8_bits(result["q"][i])
        if step >= 5:
            row["dequant calculation"] = dequantization_formula(
                result["q"][i],
                result["scale"],
                result["x_hat"][i],
            )
            row["x_hat = q / scale"] = result["x_hat"][i]
        if step >= 6:
            row["error"] = result["error"][i]
            row["abs error"] = abs(result["error"][i])
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_number_formats():
    st.title("Pillar 1 Foundation: Number Formats at a Glance")
    st.write(
        "Before DeepSeek's five FP8 pillars, the chapter introduces the basic number formats. "
        "Use one decimal value and watch how the bits change across FP32, FP16, BF16, INT8, "
        "and FP8. Sign controls positive/negative, exponent controls dynamic range, and "
        "mantissa controls precision."
    )

    target = st.number_input(
        "Decimal value to encode in every format",
        value=3.14159,
        step=0.25,
        format="%.8f",
        help="Type any ordinary decimal value. Every card below recomputes its bits and decoded value.",
    )
    formula("floating point value = (-1)^sign x (1 + mantissa_integer / 2^mantissa_bits) x 2^(stored_exponent - bias)")
    st.markdown(
        "For a slower interactive explanation of sign, exponent, mantissa, and bias, see "
        "[Interactive Visualization of Floating Point IEEE 754](https://eibx.com/interactive-visualization-of-floating-point-ieee-754/)."
    )
    st.caption("INT8 is shown as a contrast: it has sign and integer magnitude bits only. Decimal detail comes from a separate scaling factor.")

    rows = []
    format_rows = []
    for fmt in FORMATS.values():
        rep = represent_value(target, fmt)
        format_rows.append((fmt, bit_segments(fmt, rep), rep["represented"]))
        rows.append(
            {
                "Format": fmt.name,
                "Layout": f"{fmt.sign_bits} sign | {fmt.exponent_bits} exp | {fmt.mantissa_bits} mantissa"
                if fmt.kind == "float"
                else "1 sign | 7 integer",
                "Represented": rep["represented"],
                "Abs error": abs(rep["represented"] - target),
                "Overflow": rep["overflow"],
                "Underflow": rep["underflow"],
            }
        )

    st.subheader("All Formats, Same Value")
    st.plotly_chart(all_bit_layouts_figure(format_rows), width="stretch", key=f"all_bits_{target}")

    st.subheader("Decoded Values and Errors")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("How the Bits Decode")
    formula_rows = []
    for fmt in FORMATS.values():
        rep = represent_value(target, fmt)
        formula_rows.append(
            {
                "Format": fmt.name,
                "Formula for current bits": decoded_formula(fmt, rep),
                "Used for": fmt.use,
            }
        )
    st.dataframe(pd.DataFrame(formula_rows), hide_index=True, width="stretch")

    with st.expander("Inspect one format in detail", expanded=False):
        selected_name = st.selectbox("Format", list(FORMATS.keys()), index=4)
        fmt = FORMATS[selected_name]
        rep = represent_value(target, fmt)
        safe_name = fmt.name.lower().replace(" ", "_")
        st.plotly_chart(
            bit_layout_figure(fmt, bit_segments(fmt, rep)),
            width="stretch",
            key=f"detail_bits_{safe_name}_{target}",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Decoded value", f"{rep['represented']:.8g}")
        c2.metric("Abs error", f"{abs(rep['represented'] - target):.3g}")
        if fmt.kind == "float":
            min_normal, max_normal = float_range(fmt)
            c3.metric("Range", f"{min_normal:.1e}..{max_normal:.1e}")
        else:
            c3.metric("Range", "-127..127")
        st.markdown(f'<div class="mini-formula">{decoded_formula(fmt, rep)}</div>', unsafe_allow_html=True)
        st.caption(fmt.note)
        values = nearby_float_values(fmt, target)
        st.plotly_chart(
            number_line_figure(fmt, values, target, rep["represented"]),
            width="stretch",
            key=f"detail_line_{safe_name}_{target}",
        )

    render_quantization_panel()


def render_pillar1_mixed_precision():
    st.title("Pillar 1: Mixed Precision Framework")
    st.write(
        "The first pillar is a precision policy for training. DeepSeek does not train "
        "everything in FP8. It chooses the format based on what each tensor is doing."
    )

    left, right = st.columns([3, 1])
    with right:
        st.markdown("### Precision Ladder")
        st.markdown(
            f"""
            <div class="precision-ladder">
                <div class="precision-ladder-item" style="--precision-color:{PRECISION_COLORS['FP32']}">
                    <b>{chip('FP32')}</b><br>
                    Highest precision used here. Master weights, optimizer state, and weight gradients.
                </div>
                <div class="precision-ladder-item" style="--precision-color:{PRECISION_COLORS['BF16']}">
                    <b>{chip('BF16')}</b><br>
                    Middle ground. Stored activations, gradients between layers, and sensitive modules.
                </div>
                <div class="precision-ladder-item" style="--precision-color:{PRECISION_COLORS['FP8']}">
                    <b>{chip('FP8')}</b><br>
                    Lowest precision in this pillar. Used for the large GEMM inputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with left:
        st.markdown(
            f"""
            <div class="precision-card">
                <h3>The Core Rule</h3>
                <p>
                Use {chip('FP8')} where the computation is huge and repetitive.
                Use {chip('BF16')} where values are passed between layers or the operation is sensitive.
                Use {chip('FP32')} where errors would accumulate over training.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="precision-card">
                <h3>1. Forward Pass</h3>
                <p>The layer computes <code>y = W x</code>.</p>
                <p>
                The input activation <code>x</code> is stored as {chip('BF16')}.
                The master weight <code>W_master</code> is kept in {chip('FP32')} or high precision.
                For the actual matrix multiply, both are converted on the fly to {chip('FP8')}.
                </p>
                <p>
                The expensive GEMM runs as <code>W_fp8 @ x_fp8</code>.
                The result is accumulated in {chip('FP32')} and then stored as {chip('BF16')}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="precision-card">
                <h3>2. Backward Pass: dgrad</h3>
                <p>This computes the gradient with respect to the input:</p>
                <p><code>dL/dx = dL/dz @ W^T</code></p>
                <p>
                The incoming gradient <code>dL/dz</code> is stored as {chip('BF16')}, then converted to {chip('FP8')}.
                The weight matrix is also converted to {chip('FP8')} for the GEMM.
                The result <code>dL/dx</code> is accumulated in {chip('FP32')} and stored as {chip('BF16')}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="precision-card">
                <h3>3. Backward Pass: wgrad</h3>
                <p>This computes the weight gradient:</p>
                <p><code>dL/dW = x^T @ dL/dz</code></p>
                <p>
                The inputs to this GEMM use {chip('FP8')}, but the output <code>dW</code> is accumulated
                and stored in {chip('FP32')}.
                </p>
                <p>
                This is the key stability choice: <code>dW</code> directly changes the model weights,
                so DeepSeek does not store it in {chip('FP8')}.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="precision-card">
                <h3>4. Optimizer Update</h3>
                <p>The permanent model state is updated in high precision:</p>
                <p><code>W_master_new = W_master_old - learning_rate * dW</code></p>
                <p>
                <code>W_master</code>, <code>dW</code>, and optimizer states such as AdamW momentum and variance
                stay in {chip('FP32')}. After the update, the next forward pass can again cast weights down to
                {chip('FP8')} for fast compute.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="precision-card">
                <h3>Sensitive Modules That Bypass FP8</h3>
                <p>
                The PDF also says several Transformer components are kept in {chip('BF16')}:
                embeddings, the output head, MoE gating, normalization layers, and attention
                softmax/context calculation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Precision Assignments from the Chapter")
    st.dataframe(
        pd.DataFrame(
            [
                ["Master weights", "FP32", "Permanent model state; updated by optimizer."],
                ["Optimizer states", "FP32", "Momentum/variance need stable accumulation."],
                ["Fprop GEMM inputs", "FP8", "Fast matrix multiplication path."],
                ["Dgrad GEMM inputs", "FP8", "Backprop input-gradient GEMM uses FP8 inputs."],
                ["Wgrad result", "FP32", "Weight gradients are accumulated and stored at high precision."],
                ["Inter-layer activations", "BF16", "Stored between transformer layers."],
                ["Embeddings / output head / MoE gate / norm / attention softmax-context", "BF16", "Sensitive modules bypass FP8."],
            ],
            columns=["Component", "Precision", "Reason"],
        ),
        hide_index=True,
        width="stretch",
    )


def vector_reconstruction_figure(x, global_hat, fine_hat, groups):
    idx = np.arange(len(x))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx - 0.25, y=x, width=0.25, name="Original", marker_color="#4776a8"))
    fig.add_trace(go.Bar(x=idx, y=global_hat, width=0.25, name="Global x_hat", marker_color="#d91b61"))
    fig.add_trace(go.Bar(x=idx + 0.25, y=fine_hat, width=0.25, name="Fine-grained x_hat", marker_color="#2f7d51"))
    for group in np.unique(groups):
        positions = idx[groups == group]
        fig.add_vrect(
            x0=positions[0] - 0.5,
            x1=positions[-1] + 0.5,
            fillcolor="#2f7d51" if group % 2 == 0 else "#4776a8",
            opacity=0.06,
            line_width=0,
        )
    fig.update_layout(
        title="Original vs reconstructed values",
        xaxis_title="Vector index",
        yaxis_title="Value",
        barmode="group",
        height=420,
    )
    return fig


def vector_error_figure(global_error, fine_error):
    idx = np.arange(len(global_error))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx - 0.16, y=np.abs(global_error), width=0.32, name="Global abs error", marker_color="#d91b61"))
    fig.add_trace(go.Bar(x=idx + 0.16, y=np.abs(fine_error), width=0.32, name="Fine-grained abs error", marker_color="#2f7d51"))
    fig.update_layout(
        title="Reconstruction loss per element",
        xaxis_title="Vector index",
        yaxis_title="|x_hat - x|",
        height=340,
    )
    return fig


def matrix_heatmap(matrix, title, block_size=None, colorscale="RdBu"):
    text = np.vectorize(lambda value: f"{value:.2g}")(matrix)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale=colorscale,
            colorbar=dict(len=0.75),
        )
    )
    if block_size:
        rows, cols = matrix.shape
        for r in range(block_size, rows, block_size):
            fig.add_shape(type="line", x0=-0.5, x1=cols - 0.5, y0=r - 0.5, y1=r - 0.5, line=dict(color="#172033", width=2))
        for c in range(block_size, cols, block_size):
            fig.add_shape(type="line", x0=c - 0.5, x1=c - 0.5, y0=-0.5, y1=rows - 0.5, line=dict(color="#172033", width=2))
    fig.update_layout(title=title, height=390, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def low_precision_round(value, step):
    if value == 0:
        return 0.0
    quantum = max(step, abs(value) * step)
    return np.round(value / quantum) * quantum


def simulate_accumulation(k, promotion_interval, round_step):
    rng = np.random.default_rng(12)
    w = np.abs(rng.normal(0.4, 0.12, size=k))
    x = np.abs(rng.normal(0.4, 0.12, size=k))
    products = w * x
    true_cum = np.cumsum(products)

    low_total = 0.0
    chunk_total = 0.0
    fp32_total = 0.0
    low_curve = []
    promoted_curve = []
    promotion_marks = []

    for i, product in enumerate(products, start=1):
        low_total = low_precision_round(low_total + product, round_step * 8)
        chunk_total = low_precision_round(chunk_total + product, round_step)

        if i % promotion_interval == 0 or i == k:
            fp32_total += float(chunk_total)
            chunk_total = 0.0
            promotion_marks.append(i - 1)

        low_curve.append(low_total)
        promoted_curve.append(fp32_total + float(chunk_total))

    return {
        "products": products,
        "true": true_cum,
        "low": np.array(low_curve),
        "promoted": np.array(promoted_curve),
        "promotion_marks": promotion_marks,
    }


def accumulation_figure(sim, promotion_interval):
    idx = np.arange(len(sim["true"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=sim["true"], name="True FP32 sum", line=dict(color="#4776a8", width=3)))
    fig.add_trace(go.Scatter(x=idx, y=sim["low"], name="Low-precision accumulator only", line=dict(color="#d91b61", width=2)))
    fig.add_trace(go.Scatter(x=idx, y=sim["promoted"], name=f"Promote every Nc={promotion_interval}", line=dict(color="#2f7d51", width=3)))
    for mark in sim["promotion_marks"]:
        fig.add_vline(x=mark, line_dash="dot", line_color="#9aa4b2", opacity=0.5)
    fig.update_layout(
        title="Running dot-product sum",
        xaxis_title="Product index i",
        yaxis_title="Cumulative sum",
        height=460,
    )
    return fig


def chunk_accumulation(products, start, end, round_step):
    total = 0.0
    rows = []
    for i in range(start, end):
        before = total
        total = low_precision_round(total + products[i], round_step)
        rows.append([i, products[i], before, total])
    return total, rows


def accumulation_hardware_figure(sim, promotion_interval, chunk_index, round_step):
    products = sim["products"]
    k = len(products)
    start = chunk_index * promotion_interval
    end = min(start + promotion_interval, k)
    low_partial, _ = chunk_accumulation(products, start, end, round_step)
    fp32_before = sim["promoted"][start - 1] if start > 0 else 0.0
    fp32_after = sim["promoted"][end - 1]
    is_promotion = (end - 1) in sim["promotion_marks"]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.04, x1=0.62, y0=0.52, y1=0.94, fillcolor="#f8edf3", line=dict(color="#cc2f66", width=2))
    fig.add_shape(type="rect", x0=0.70, x1=0.96, y0=0.52, y1=0.94, fillcolor="#eef7f1", line=dict(color="#2f7d51", width=2))
    fig.add_annotation(x=0.07, y=0.90, text="<b>Tensor Core</b>", showarrow=False, xanchor="left", font=dict(size=16))
    fig.add_annotation(x=0.73, y=0.90, text="<b>CUDA Core</b>", showarrow=False, xanchor="left", font=dict(size=16))

    # GEMM inputs and WGMMA chunk.
    fig.add_shape(type="rect", x0=0.08, x1=0.24, y0=0.72, y1=0.84, fillcolor="#cc2f66", line=dict(color="#cc2f66"))
    fig.add_shape(type="rect", x0=0.28, x1=0.44, y0=0.72, y1=0.84, fillcolor="#cc2f66", line=dict(color="#cc2f66"))
    fig.add_annotation(x=0.16, y=0.78, text="<b>W chunk</b><br>FP8", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=0.36, y=0.78, text="<b>X chunk</b><br>FP8", showarrow=False, font=dict(color="white", size=12))
    fig.add_annotation(x=0.26, y=0.66, text=f"WGMMA chunk {chunk_index + 1}<br>products {start}..{end - 1}", showarrow=False, font=dict(size=13))

    # Low precision accumulator.
    fig.add_shape(type="rect", x0=0.46, x1=0.58, y0=0.66, y1=0.86, fillcolor="#fff8f3", line=dict(color="#e0855b", width=2))
    fig.add_annotation(x=0.52, y=0.76, text=f"<b>Low Prec Acc</b><br>{low_partial:.5g}", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=0.46, y=0.78, ax=0.24, ay=0.78, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2)
    fig.add_annotation(x=0.46, y=0.74, ax=0.44, ay=0.78, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2)

    # CUDA FP32 register.
    fig.add_shape(type="rect", x0=0.76, x1=0.92, y0=0.66, y1=0.86, fillcolor="#e8f5ec", line=dict(color="#2f7d51", width=2))
    fig.add_annotation(
        x=0.84,
        y=0.76,
        text=f"<b>FP32 Register</b><br>before {fp32_before:.5g}<br>after {fp32_after:.5g}",
        showarrow=False,
        font=dict(size=12),
    )

    if is_promotion:
        fig.add_annotation(
            x=0.76,
            y=0.76,
            ax=0.58,
            ay=0.76,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowwidth=4,
            arrowcolor="#2f7d51",
            text="promote partial sum",
            font=dict(size=12, color="#2f7d51"),
        )
    else:
        fig.add_annotation(x=0.67, y=0.76, text="no promotion yet", showarrow=False, font=dict(size=12, color="#7b8294"))

    fig.add_shape(type="rect", x0=0.04, x1=0.96, y0=0.12, y1=0.38, fillcolor="#fbfcfe", line=dict(color="#d8dee8"))
    fig.add_annotation(
        x=0.50,
        y=0.29,
        text=(
            f"Chunk sum is accumulated quickly in low precision on the Tensor Core.<br>"
            f"At Nc={promotion_interval}, the partial sum is delegated to the CUDA Core FP32 register."
        ),
        showarrow=False,
        font=dict(size=14),
    )
    fig.add_annotation(
        x=0.50,
        y=0.18,
        text=f"Final true FP32 sum: {sim['true'][-1]:.6g} | final promoted sum: {sim['promoted'][-1]:.6g}",
        showarrow=False,
        font=dict(size=13),
    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    return fig


def format_chunk_sum(products, start, end, max_terms=8):
    shown = products[start : min(end, start + max_terms)]
    terms = " + ".join(f"{value:.4g}" for value in shown)
    if end - start > max_terms:
        terms += " + ..."
    return terms


def render_pillar3_accumulation():
    st.title("Pillar 3: Increasing Accumulation Precision")
    st.write(
        "Pillar 3 protects long FP8 dot products. Tensor Cores are fast for FP8 multiplication, "
        "but their low-precision accumulator can lose information across thousands of additions. "
        "DeepSeek periodically promotes partial sums to CUDA Core FP32 registers."
    )

    st.subheader("Tiny CUDA GPU Primer")
    st.markdown(
        """
        <div class="gpu-primer">
            <p>A GPU has different execution units. Pillar 3 is about moving the running sum
            between the unit that is fastest and the unit that is more numerically stable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            """
            <div class="gpu-card" style="--gpu-color:#cc2f66">
                <h3>Tensor Cores</h3>
                <span class="gpu-chip">fast matrix math</span>
                <p>Specialized hardware for high-throughput GEMM chunks such as FP8 x FP8.</p>
                <p><b>Good at:</b> speed.</p>
                <p><b>Weakness:</b> low-precision accumulation over long sums.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="gpu-card" style="--gpu-color:#e0855b">
                <h3>Low Prec Acc</h3>
                <span class="gpu-chip">temporary sum</span>
                <p>Small accumulator inside the Tensor Core that holds the running sum for a chunk.</p>
                <p><b>Good at:</b> short partial sums.</p>
                <p><b>Weakness:</b> precision loss if used too long.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="gpu-card" style="--gpu-color:#2f7d51">
                <h3>CUDA Cores</h3>
                <span class="gpu-chip">FP32 register</span>
                <p>General-purpose GPU cores that can accumulate promoted partial sums in FP32.</p>
                <p><b>Good at:</b> numerical stability.</p>
                <p><b>Tradeoff:</b> slower than Tensor Cores for raw GEMM throughput.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    formula("Y_ij = sum_k W_ik * X_kj")
    formula("Tensor Core: fast FP8 products + low precision accumulator; CUDA Core: periodic FP32 accumulation")

    st.markdown(
        f"""
        <div class="flow-card">
            <h3>The Flow</h3>
            <div class="flow-step"><span class="tag" style="background:#d91b61">FP8</span> FP8 GEMM inputs</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#cc2f66">Tensor Core</span> Tensor Core WGMMA chunk</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#e0855b">temporary</span> Low Precision Accumulator</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-note">every Nc operations</div>
            <p>
            <b>Nc</b> means the chunk size / promotion interval: how many FP8 product terms are accumulated
            before the partial sum is moved to the CUDA Core. The PDF uses <code>Nc = 128</code>.
            It is written like a symbol rather than expanded as a formal acronym; read it as
            <b>N-sub-c</b>, the chunk interval.
            </p>
            <div class="flow-step"><span class="tag" style="background:#2f7d51">promotion</span> Promote partial sum to CUDA Core</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#2f7d51">FP32</span> FP32 Register</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#4776a8">output</span> Stable output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pillar4_mantissa_over_exponents():
    st.title("Pillar 4: Mantissa Over Exponents")
    st.write(
        "Pillar 4 is about DeepSeek's FP8 format choice. FP8 has only 8 bits, so choosing more "
        "exponent bits gives more dynamic range, while choosing more mantissa bits gives more precision."
    )

    target = st.number_input(
        "Example value for bit visualization",
        value=3.14159,
        step=0.25,
        format="%.6f",
        key="pillar4_value",
    )

    format_rows = []
    summary_rows = []
    for fmt_name in ["FP8 E4M3", "FP8 E5M2"]:
        fmt = FORMATS[fmt_name]
        rep = represent_value(target, fmt)
        format_rows.append((fmt, bit_segments(fmt, rep), rep["represented"]))
        summary_rows.append(
            {
                "Format": fmt.name,
                "Exponent bits": fmt.exponent_bits,
                "Mantissa bits": fmt.mantissa_bits,
                "Tradeoff": "more precision, less range" if fmt.name == "FP8 E4M3" else "more range, less precision",
                "Represented value": rep["represented"],
                "Abs error": abs(rep["represented"] - target),
            }
        )

    st.subheader("The Two FP8 Choices")
    st.plotly_chart(all_bit_layouts_figure(format_rows), width="stretch", key=f"pillar4_bits_{target}")

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.markdown(
        f"""
        <div class="precision-card">
            <h3>What DeepSeek Chooses</h3>
            <p>
            Conventional mixed-precision training often used <b>E4M3</b> for the forward pass and
            <b>E5M2</b> for backward gradients, because gradients can have larger outliers.
            </p>
            <p>
            DeepSeek instead uses <b>E4M3 uniformly</b>. Pillar 2's fine-grained quantization already
            handles local dynamic range, so DeepSeek can spend the FP8 bits on an extra mantissa bit
            and preserve more precision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pillar5_online_quantization():
    st.title("Pillar 5: Online Quantization")
    st.write(
        "Pillar 5 is about when the scale factor is computed. A tensor can only be stored in "
        "FP8 accurately if the scale matches the values being quantized. DeepSeek avoids stale "
        "historical scales by computing the scale from the current tensor in real time."
    )

    st.caption(
        "`R_format` means the largest usable magnitude of the target low-precision format. "
        "For an INT8 toy example this would be 127; for DeepSeek's pillar this should be read "
        "as the usable FP8 range after choosing the FP8 format."
    )
    formula("alpha_current = max(abs(current tensor)); scale_current = R_format / alpha_current; x_scaled = x * scale_current")

    delayed, online = st.columns(2)
    with delayed:
        st.markdown(
            f"""
            <div class="precision-card">
                <h3>Delayed Quantization</h3>
                <p>
                Conventional delayed quantization estimates the next scale from previous batches
                or earlier iterations.
                </p>
                <p><code>scale_delayed[t] = R_format / alpha_history[t-1]</code></p>
                <p>
                If the current tensor suddenly gets larger, the stale scale can overflow.
                If the current tensor gets smaller, the old scale wastes the available
                {chip('FP8')} range and loses precision.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with online:
        st.markdown(
            f"""
            <div class="precision-card">
                <h3>Online Quantization</h3>
                <p>
                DeepSeek computes the maximum absolute value from the current tensor itself,
                then immediately quantizes that same tensor.
                </p>
                <p><code>scale_online[t] = R_format / max(abs(x_current[t]))</code></p>
                <p>
                The scale follows the real data distribution at this moment, so the
                {chip('FP8')} codes are used more effectively and the quantized GEMM inputs
                are more stable.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("DeepSeek Flow")
    st.markdown(
        f"""
        <div class="flow-card">
            <div class="flow-step"><span class="tag" style="background:#4776a8">current</span> Current activation or weight tensor</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#2f7d51">max</span> Find <code>alpha = max(abs(tensor))</code></div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#2f7d51">scale</span> Compute <code>scale = R_format / alpha</code></div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#d91b61">FP8</span> Quantize immediately using the fresh scale</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step"><span class="tag" style="background:#d91b61">GEMM</span> Use fresh {chip('FP8')} inputs for the matrix multiply</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why This Pillar Matters")
    st.dataframe(
        pd.DataFrame(
            [
                [
                    "Current max is larger than history",
                    "Old scale may map values outside the FP8 range",
                    "Compute alpha from the current tensor before quantizing",
                ],
                [
                    "Current max is smaller than history",
                    "Old scale underuses FP8 codes, so nearby values collapse together",
                    "Use the current smaller alpha to spread values across the FP8 range",
                ],
                [
                    "Extra work",
                    "A max-reduction is required before quantization",
                    "The PDF treats this cost as small compared with the following GEMM",
                ],
            ],
            columns=["Situation", "Delayed scaling problem", "Online quantization fix"],
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        """
        <div class="precision-card">
            <h3>One-Sentence Summary</h3>
            <p>
            Online quantization means DeepSeek does not wait for the next batch to learn the right scale:
            it measures the current tensor, computes the current scale, and quantizes immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pillar2_fine_grained():
    st.title("Pillar 2: Fine-Grained Quantization")
    st.write(
        "Pillar 2 is about making the scale factor local. A global scale can be dominated by one "
        "outlier; fine-grained quantization uses separate scales for activation groups and weight blocks."
    )

    st.subheader("Weight Matrix: Global Scale vs Block Scale")
    m1, m2, m3 = st.columns(3)
    with m1:
        matrix_size = st.select_slider("Matrix size", options=[8, 16], value=8, key="p2_matrix_size")
    with m2:
        block_size = st.select_slider("Weight block size", options=[1, 2, 4, 8], value=4, key="p2_block_size")
    with m3:
        outlier_strength = st.select_slider("Matrix outlier strength", options=[3.0, 8.0, 20.0, 50.0], value=20.0, key="p2_matrix_outlier")

    rng = np.random.default_rng(7)
    weights = rng.normal(0, 1.0, size=(matrix_size, matrix_size))
    weights[: matrix_size // 2, : matrix_size // 2] *= 0.15
    weights[-2:, -2:] += outlier_strength
    global_matrix = quantize_to_int8(weights)
    fine_matrix = quantize_blockwise_to_int8(weights, block_size)

    formula(
        "Global matrix quantization: "
        "alpha_global = max(abs(W)); "
        "scale_global = 127 / alpha_global; "
        "Q = round(W * scale_global); "
        "W_hat = Q / scale_global; "
        "loss = abs(W_hat - W)"
    )
    formula(
        "Blockwise matrix quantization: "
        "for each block B_ij, "
        "alpha_ij = max(abs(B_ij)); "
        "scale_ij = 127 / alpha_ij; "
        "Q_ij = round(B_ij * scale_ij); "
        "B_hat_ij = Q_ij / scale_ij"
    )

    matrix_metrics = st.columns(3)
    matrix_metrics[0].metric("Global alpha", f"{global_matrix['alpha']:.4g}")
    matrix_metrics[1].metric("Global scale", f"{global_matrix['scale']:.4g}")
    matrix_metrics[2].metric("Blocks", f"{fine_matrix['scales'].shape[0]} x {fine_matrix['scales'].shape[1]}")

    error_metrics = st.columns(3)
    error_metrics[0].metric("Global mean abs error", f"{np.mean(np.abs(global_matrix['error'])):.4g}")
    error_metrics[1].metric("Blockwise mean abs error", f"{np.mean(fine_matrix['abs_error']):.4g}")
    error_metrics[2].metric(
        "Block scale range",
        f"{np.min(fine_matrix['scales']):.4g} .. {np.max(fine_matrix['scales']):.4g}",
    )

    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(matrix_heatmap(weights, "Original weight matrix", block_size), width="stretch")
    with h2:
        st.plotly_chart(matrix_heatmap(fine_matrix["scales"], "One scale per block", None, "Viridis"), width="stretch")
    h3, h4 = st.columns(2)
    with h3:
        st.plotly_chart(matrix_heatmap(np.abs(global_matrix["error"]), "Global reconstruction loss", None, "Reds"), width="stretch")
    with h4:
        st.plotly_chart(matrix_heatmap(fine_matrix["abs_error"], "Blockwise reconstruction loss", block_size, "Reds"), width="stretch")


def render_placeholder(title):
    st.title(title)
    st.info(
        "This view is intentionally a placeholder in the rebuilt app. "
        "The first implemented screen is the interactive number-format view; next we can build this pillar around the same step-by-step style."
    )


st.sidebar.title("DeepSeek Quantization")
page = st.sidebar.radio("Views", PAGES)

if page == "Number Formats":
    render_number_formats()
elif page == "Pillar 1: Mixed Precision":
    render_pillar1_mixed_precision()
elif page == "Pillar 2: Fine-Grained Quantization":
    render_pillar2_fine_grained()
elif page == "Pillar 3: Accumulation Precision":
    render_pillar3_accumulation()
elif page == "Pillar 4: Mantissa Over Exponents":
    render_pillar4_mantissa_over_exponents()
elif page == "Pillar 5: Online Quantization":
    render_pillar5_online_quantization()
else:
    render_placeholder(page)

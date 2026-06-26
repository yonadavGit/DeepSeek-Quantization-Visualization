import pandas as pd
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
from quantization import dequantization_formula, int8_bits, quantization_formula, quantize_to_int8
from quant_formats import FORMATS, bit_segments, decoded_formula, float_range, nearby_float_values, represent_value


st.set_page_config(page_title="DeepSeek Quantization Visualizer", page_icon="8", layout="wide")
inject_css()


def joined_bits(fmt, rep):
    return " ".join(bit for _, bit, _ in bit_segments(fmt, rep))


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


def render_placeholder(title):
    st.title(title)
    st.info(
        "This view is intentionally a placeholder in the rebuilt app. "
        "The first implemented screen is the interactive number-format view; next we can build this pillar around the same step-by-step style."
    )


st.sidebar.title("DeepSeek Quantization")
page = st.sidebar.radio("Views", PAGES)

if page in ("Number Formats", "Pillar 1: Mixed Precision"):
    render_number_formats()
else:
    render_placeholder(page)

import plotly.graph_objects as go
import numpy as np
import streamlit as st


PAGES = [
    "Number Formats",
    "Pillar 1: Mixed Precision",
    "Pillar 2: Fine-Grained Quantization",
    "Pillar 3: Accumulation Precision",
    "Pillar 4: Mantissa Over Exponents",
    "Pillar 5: Online Quantization",
]


def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        .format-note {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 0.85rem;
            background: #fbfcfe;
            min-height: 132px;
        }
        .format-card {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 0.9rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }
        .format-card h3 {
            margin-top: 0;
            margin-bottom: 0.2rem;
        }
        .mini-formula {
            border-left: 4px solid #e0855b;
            background: #fff8f3;
            padding: 0.55rem 0.75rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }
        .formula {
            border-left: 4px solid #2f6f73;
            background: #f3f8f8;
            padding: 0.65rem 0.9rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def formula(text):
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def bit_layout_figure(fmt, segments):
    fig = go.Figure()
    for i, (name, bit, color) in enumerate(segments):
        fig.add_shape(
            type="rect",
            x0=i,
            x1=i + 1,
            y0=0,
            y1=1,
            line=dict(color="white", width=1),
            fillcolor=color,
        )
        font_size = 11 if fmt.total_bits <= 16 else 8
        fig.add_annotation(
            x=i + 0.5,
            y=0.55,
            text=bit,
            showarrow=False,
            font=dict(color="white", size=font_size),
        )

    start = 0
    last = None
    for i, (name, _, color) in enumerate(segments + [("END", "", "")]):
        if last is None:
            last = name
        if name != last:
            fig.add_annotation(
                x=(start + i) / 2,
                y=-0.22,
                text=last,
                showarrow=False,
                font=dict(size=11, color="#172033"),
            )
            start = i
            last = name

    fig.update_xaxes(visible=False, range=[0, fmt.total_bits])
    fig.update_yaxes(visible=False, range=[-0.38, 1.08])
    fig.update_layout(
        height=155 if fmt.total_bits <= 16 else 185,
        margin=dict(l=10, r=10, t=10, b=32),
        plot_bgcolor="white",
    )
    return fig


def number_line_figure(fmt, values, target, represented):
    fig = go.Figure()
    visible = values[np.abs(values - target) <= max(1.0, abs(target) * 0.75)]
    if len(visible) < 3:
        visible = values
    fig.add_trace(
        go.Scatter(
            x=visible,
            y=[0] * len(visible),
            mode="markers",
            marker=dict(size=8, color="#9aa4b2"),
            name=f"{fmt.name} representable values",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target],
            y=[0.18],
            mode="markers+text",
            marker=dict(size=14, color="#172033", symbol="x"),
            text=["target"],
            textposition="top center",
            name="target value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[represented],
            y=[-0.18],
            mode="markers+text",
            marker=dict(size=14, color="#e0855b"),
            text=["represented"],
            textposition="bottom center",
            name="stored value",
        )
    )
    fig.update_yaxes(visible=False, range=[-0.55, 0.55])
    fig.update_xaxes(title="value")
    fig.update_layout(height=270, margin=dict(l=10, r=10, t=25, b=35))
    return fig


def all_bit_layouts_figure(format_rows):
    max_bits = max(fmt.total_bits for fmt, _, _ in format_rows)
    row_height = 1.25
    fig = go.Figure()

    for row_idx, (fmt, segments, represented) in enumerate(format_rows):
        y = (len(format_rows) - row_idx - 1) * row_height
        fig.add_annotation(
            x=-0.6,
            y=y + 0.5,
            text=f"<b>{fmt.name}</b><br><span style='font-size:10px'>{represented:.6g}</span>",
            showarrow=False,
            xanchor="right",
            align="right",
            font=dict(size=12, color="#172033"),
        )
        for i, (name, bit, color) in enumerate(segments):
            fig.add_shape(
                type="rect",
                x0=i,
                x1=i + 1,
                y0=y,
                y1=y + 1,
                line=dict(color="white", width=1),
                fillcolor=color,
            )
            fig.add_annotation(
                x=i + 0.5,
                y=y + 0.54,
                text=bit,
                showarrow=False,
                font=dict(color="white", size=8 if fmt.total_bits == 32 else 11),
            )

        start = 0
        last = None
        for i, (name, _, _) in enumerate(segments + [("END", "", "")]):
            if last is None:
                last = name
            if name != last:
                fig.add_annotation(
                    x=(start + i) / 2,
                    y=y - 0.16,
                    text=last,
                    showarrow=False,
                    font=dict(size=9, color="#394150"),
                )
                start = i
                last = name

    fig.update_xaxes(visible=False, range=[-4.5, max_bits + 0.5])
    fig.update_yaxes(visible=False, range=[-0.35, len(format_rows) * row_height])
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
    )
    return fig


def quantization_mapping_figure(result, step, focus_idx=0):
    x = result["x"]
    q = result["q"]
    x_hat = result["x_hat"]
    alpha = result["alpha"]
    qmax = result["qmax"]
    scale = result["scale"]
    x_pos = x / alpha * qmax if alpha else x
    xhat_pos = x_hat / alpha * qmax if alpha else x_hat

    fig = go.Figure()
    fp32_y = 2.0
    int8_y = 1.0
    deq_y = 0.0
    colors = ["#7b559d"] * len(x)
    max_idx = int(np.argmax(np.abs(x)))
    focus_idx = int(np.clip(focus_idx, 0, len(x) - 1))
    if step >= 1:
        colors[max_idx] = "#28a6cf"
    if step >= 3:
        colors[focus_idx] = "#172033"

    fig.add_shape(type="line", x0=-qmax, x1=qmax, y0=fp32_y, y1=fp32_y, line=dict(color="#d9b7db", width=16))
    fig.add_shape(type="line", x0=-qmax, x1=qmax, y0=int8_y, y1=int8_y, line=dict(color="#f3b7cc", width=16))
    if step >= 5:
        fig.add_shape(type="line", x0=-qmax, x1=qmax, y0=deq_y, y1=deq_y, line=dict(color="#f7d9c8", width=16))

    fig.add_annotation(x=-qmax - 7, y=fp32_y, text="<b>FP32 x</b>", showarrow=False, xanchor="right", font=dict(size=13))
    fig.add_annotation(x=-qmax - 7, y=int8_y, text="<b>INT8 q</b>", showarrow=False, xanchor="right", font=dict(size=13))
    if step >= 5:
        fig.add_annotation(x=-qmax - 7, y=deq_y, text="<b>dequant x_hat</b>", showarrow=False, xanchor="right", font=dict(size=13))

    fig.add_annotation(x=-qmax, y=fp32_y + 0.18, text="-alpha", showarrow=False, font=dict(color="#28a6cf", size=13))
    fig.add_annotation(x=qmax, y=fp32_y + 0.18, text="+alpha", showarrow=False, font=dict(color="#28a6cf", size=13))
    fig.add_annotation(x=-qmax, y=int8_y - 0.22, text="-127", showarrow=False, font=dict(color="#d91b61", size=13))
    fig.add_annotation(x=0, y=int8_y - 0.22, text="0", showarrow=False, font=dict(color="#172033", size=13))
    fig.add_annotation(x=qmax, y=int8_y - 0.22, text="127", showarrow=False, font=dict(color="#d91b61", size=13))

    if step >= 0:
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=[fp32_y] * len(x),
                mode="markers+text",
                marker=dict(size=15, color=colors),
                text=[f"{v:g}" for v in x],
                textposition="top center",
                name="FP32 values",
            )
        )

    if step >= 1:
        fig.add_annotation(
            x=x_pos[max_idx],
            y=fp32_y + 0.42,
            text=f"highest absolute value alpha = {alpha:g}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#28a6cf",
            font=dict(color="#172033", size=13),
        )

    if step >= 2:
        fig.add_annotation(
            x=0,
            y=1.55,
            text=f"scale = 127 / {alpha:g} = {scale:.4g}",
            showarrow=False,
            font=dict(size=15, color="#172033"),
            bgcolor="#f3f8f8",
            bordercolor="#2f6f73",
            borderwidth=1,
        )

    if step >= 3:
        for xi, qi in zip(x_pos, q):
            fig.add_annotation(
                x=qi,
                y=int8_y,
                ax=xi,
                ay=fp32_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowwidth=1.5,
                arrowcolor="#565d6b",
            )
        fig.add_trace(
            go.Scatter(
                x=q,
                y=[int8_y] * len(q),
                mode="markers+text",
                marker=dict(
                    size=[19 if i == focus_idx else 15 for i in range(len(q))],
                    color=["#172033" if i == focus_idx else "#d91b61" for i in range(len(q))],
                ),
                text=[f"{v:.0f}" for v in q],
                textposition="bottom center",
                name="INT8 codes",
            )
        )
        if step == 3:
            equation = f"{x[focus_idx]:g} x {scale:.4g} = {result['scaled'][focus_idx]:.4g}"
        else:
            equation = (
                f"q = round({x[focus_idx]:g} x 127 / {alpha:g})"
                f"<br>= round({result['scaled'][focus_idx]:.4g}) = {int(q[focus_idx])}"
            )
        fig.add_annotation(
            x=q[focus_idx],
            y=1.32,
            text=equation,
            showarrow=True,
            arrowhead=2,
            arrowcolor="#172033",
            bgcolor="#fff8f3",
            bordercolor="#e0855b",
            borderwidth=1,
            font=dict(size=13, color="#172033"),
        )

    if step >= 5:
        for qi, xi_hat_pos in zip(q, xhat_pos):
            fig.add_annotation(
                x=xi_hat_pos,
                y=deq_y,
                ax=qi,
                ay=int8_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowwidth=1.5,
                arrowcolor="#e0855b",
            )
        fig.add_trace(
            go.Scatter(
                x=xhat_pos,
                y=[deq_y] * len(x_hat),
                mode="markers+text",
                marker=dict(
                    size=[17 if i == focus_idx else 12 for i in range(len(x_hat))],
                    color=["#172033" if i == focus_idx else "#e0855b" for i in range(len(x_hat))],
                    symbol="diamond",
                ),
                text=[f"{v:.3g}" for v in x_hat],
                textposition="bottom center",
                name="reconstructed",
            )
        )
        fig.add_annotation(
            x=xhat_pos[focus_idx],
            y=0.36,
            text=f"x_hat = {int(q[focus_idx])} / {scale:.4g} = {x_hat[focus_idx]:.4g}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e0855b",
            bgcolor="#fff8f3",
            bordercolor="#e0855b",
            borderwidth=1,
            font=dict(size=13, color="#172033"),
        )

    if step >= 6:
        fig.add_annotation(
            x=x_pos[focus_idx],
            y=fp32_y - 0.32,
            ax=xhat_pos[focus_idx],
            ay=deq_y + 0.05,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor="#b84545",
        )
        fig.add_annotation(
            x=xhat_pos[focus_idx],
            y=-0.34,
            text=f"error = {x_hat[focus_idx]:.4g} - {x[focus_idx]:g} = {x_hat[focus_idx] - x[focus_idx]:.4g}",
            showarrow=False,
            bgcolor="#fff1f1",
            bordercolor="#b84545",
            borderwidth=1,
            font=dict(size=13, color="#172033"),
        )

    fig.update_xaxes(range=[-qmax * 1.24, qmax * 1.18], title="visual position after range mapping")
    fig.update_yaxes(visible=False, range=[-0.62, 2.58])
    fig.update_layout(
        height=560,
        margin=dict(l=15, r=15, t=25, b=45),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig

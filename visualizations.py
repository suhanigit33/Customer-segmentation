import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = px.colors.qualitative.Bold


def scatter_2d(df: pd.DataFrame, labels: np.ndarray,
               x_col: str, y_col: str, seg_names: dict) -> go.Figure:
    df2 = df.copy()
    df2["Segment"] = [seg_names.get(l, f"Noise ({l})") if l >= 0 else "Noise" for l in labels]
    df2["_color_key"] = [str(l) for l in labels]

    fig = px.scatter(
        df2, x=x_col, y=y_col, color="Segment",
        color_discrete_sequence=PALETTE,
        hover_data={c: True for c in df2.columns if not c.startswith("_")},
        title=f"{x_col} vs {y_col}",
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.4, color="white")))
    fig.update_layout(legend_title_text="Segment", height=450)
    return fig



def elbow_plot(elbow_data: dict, selected_k: int) -> go.Figure:
    ks = elbow_data["k"]
    inertias = elbow_data["inertia"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias, mode="lines+markers",
        line=dict(color="#378ADD", width=2),
        marker=dict(
            size=[14 if k == selected_k else 8 for k in ks],
            color=["#D85A30" if k == selected_k else "#378ADD" for k in ks],
            line=dict(width=1, color="white"),
        ),
        name="Inertia"
    ))
    fig.update_layout(
        title="Elbow Method — Inertia vs K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (WCSS)",
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    return fig



def silhouette_plot(elbow_data: dict, selected_k: int) -> go.Figure:
    ks = elbow_data["k"]
    sils = elbow_data["silhouette"]

    fig = go.Figure(go.Bar(
        x=ks, y=sils,
        marker_color=["#1D9E75" if k == selected_k else "rgba(29,158,117,0.4)" for k in ks],
        marker_line_width=0,
        text=[f"{s:.2f}" if s else "" for s in sils],
        textposition="outside",
    ))
    fig.update_layout(
        title="Silhouette Score vs K",
        xaxis_title="K", yaxis_title="Silhouette Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    return fig



def radar_chart(profiles: pd.DataFrame, feature_cols: list[str], seg_names: dict) -> go.Figure:
    """Normalise features 0-100 per column then plot radar."""
    norm = profiles[feature_cols].copy()
    for col in feature_cols:
        mn, mx = norm[col].min(), norm[col].max()
        if mx > mn:
            norm[col] = (norm[col] - mn) / (mx - mn) * 100
        else:
            norm[col] = 50.0

    fig = go.Figure()
    for _, row in profiles.iterrows():
        seg_id = int(row["_Segment"])
        name = seg_names.get(seg_id, f"Segment {seg_id}")
        vals = [norm.loc[row.name, c] for c in feature_cols]
        vals += [vals[0]]  # close the polygon
        cats = feature_cols + [feature_cols[0]]
        color = PALETTE[seg_id % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=name,
            fill="toself",
            line=dict(color=color, width=2),
            marker=dict(color=color),
            opacity=0.75,
        ))

    fig.update_layout(
    polar=dict(
        bgcolor="#020617",
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor="#334155",
            tickfont=dict(color="#E2E8F0"),
        ),
        angularaxis=dict(
            gridcolor="#334155",
            tickfont=dict(color="#E2E8F0"),
        ),
    ),
    paper_bgcolor="#020617",
    plot_bgcolor="#020617",
    title=dict(
        text="Feature Profile by Segment (normalised 0–100)",
        font=dict(color="#F1F5F9", size=18),
    ),
    height=550,  # 🔥 bigger
    margin=dict(l=40, r=40, t=60, b=40),
    showlegend=True,
    legend=dict(
        title="Segment",
        font=dict(color="#E2E8F0"),
        bgcolor="#334155",
        bordercolor="#E2E8F0",
        borderwidth=1,
    )
    )
    return fig


# ── Bar: segment sizes ─────────────────────────────────────────────────────────

def segment_size_bar(profiles: pd.DataFrame, seg_names: dict) -> go.Figure:
    names = [seg_names.get(int(r["_Segment"]), f"Segment {int(r['_Segment'])}") for _, r in profiles.iterrows()]
    counts = profiles["Count"].tolist()
    colors = [PALETTE[int(r["_Segment"]) % len(PALETTE)] for _, r in profiles.iterrows()]

    fig = go.Figure(go.Bar(
        x=names, y=counts,
        marker_color=colors,
        marker_line_width=0,
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        title="Customers per Segment",
        xaxis_title="Segment", yaxis_title="Count",
        template="plotly_white",
        height=320,
        showlegend=False,
    )
    return fig


# ── Heatmap ────────────────────────────────────────────────────────────────────

def feature_heatmap(profiles: pd.DataFrame, feature_cols: list[str], seg_names: dict) -> go.Figure:
    norm = profiles[feature_cols].copy()
    for col in feature_cols:
        mn, mx = norm[col].min(), norm[col].max()
        if mx > mn:
            norm[col] = (norm[col] - mn) / (mx - mn)

    y_labels = [seg_names.get(int(r["_Segment"]), f"Seg {int(r['_Segment'])}") for _, r in profiles.iterrows()]

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=feature_cols,
        y=y_labels,
        colorscale="Blues",
        showscale=True,
        text=profiles[feature_cols].round(1).values,
        texttemplate="%{text}",
        textfont_size=11,
    ))
    fig.update_layout(
        title="Feature Heatmap by Segment (normalised)",
        template="plotly_white",
        height=320,
    )
    return fig

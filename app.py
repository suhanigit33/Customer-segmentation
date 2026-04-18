"""
Customer Segmentation Dashboard
Run: streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

from data_generator import generate_customer_data
from preprocessing import preprocess, get_feature_names
from clustering import (
    run_kmeans, run_dbscan, run_hierarchical,
    elbow_curve, segment_profiles,
)
from visualizations import (
    scatter_2d, elbow_plot, silhouette_plot,
    radar_chart, segment_size_bar, feature_heatmap,
)

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .stMetric {
        background: #1E293B !important;
        border-radius: 12px;
        padding: 15px !important;
        color: #E2E8F0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    .stMetric label {
        color: #94A3B8 !important;
        font-size: 14px;
    }

    .stMetric div {
        color: #F8FAFC !important;
        font-weight: bold;
    }

    h1 { font-size: 1.8rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Controls")

    st.subheader("Data")
    data_source = st.radio("Source", ["Generate synthetic", "Upload CSV"])
    n_samples = st.slider("Sample size", 100, 1000, 300, 50,
                          disabled=(data_source == "Upload CSV"))

    st.divider()

    st.subheader("Algorithm")
    algo = st.selectbox("Clustering algorithm", ["K-Means", "DBSCAN", "Hierarchical"])

    if algo == "K-Means":
        k = st.slider("Number of clusters (K)", 2, 10, 4)
        show_elbow = st.checkbox("Compute elbow curve", value=True)

    elif algo == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 0.1, 3.0, 0.8, 0.05)
        min_samples = st.slider("Min samples", 2, 20, 5)
        k = None
        show_elbow = False

    else:  
        k = st.slider("Number of clusters (K)", 2, 10, 4)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        show_elbow = False

    st.divider()

    st.subheader("Preprocessing")
    scaler_type = st.radio("Scaler", ["standard", "minmax"])

    st.divider()

    st.subheader("Visualisation")
    feat_cols = get_feature_names()
    x_feat = st.selectbox("Scatter X-axis", feat_cols, index=1)
    y_feat = st.selectbox("Scatter Y-axis", feat_cols, index=2)

    st.divider()

    st.subheader("Segment names")
    seg_names = {}
    max_k = k if k else 8
    for i in range(max_k):
        default = ["High Value", "Budget Savers", "Young Explorers",
                   "Loyal Seniors", "At-Risk", "VIP", "New Customers", "Churn Risk"][i % 8]
        seg_names[i] = st.text_input(f"Cluster {i}", default, key=f"seg_{i}")

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        missing = [c for c in feat_cols if c not in df_raw.columns]
        if missing:
            st.error(f"CSV is missing columns: {missing}")
            st.stop()
    else:
        st.info("👈 Upload a CSV or switch to 'Generate synthetic'.")
        st.stop()
else:
    df_raw = generate_customer_data(n_samples)

X_scaled, scaler = preprocess(df_raw, scaler_type)

if algo == "K-Means":
    result = run_kmeans(X_scaled, k)
elif algo == "DBSCAN":
    result = run_dbscan(X_scaled, eps, min_samples)
else:
    result = run_hierarchical(X_scaled, k, linkage)

labels = result.labels
profiles = segment_profiles(df_raw, labels, feat_cols)

st.title("🐯 Customer Segmentation Dashboard")
st.caption(f"Algorithm: **{algo}** · Scaler: **{scaler_type}** · Dataset: **{len(df_raw):,} customers**")

 
col1, col2, col3, col4 = st.columns(4)
noise_count = int(np.sum(labels == -1))
sil = f"{result.silhouette:.3f}" if result.silhouette is not None else "N/A"
db  = f"{result.davies_bouldin:.3f}" if result.davies_bouldin is not None else "N/A"

col1.metric("Total customers",  f"{len(df_raw):,}")
col2.metric("Segments found",   str(result.n_clusters))
col3.metric("Silhouette score", sil)
col4.metric("Davies-Bouldin",   db)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Scatter Plot",
    "📈 Elbow / Silhouette",
    "🕸️ Radar Profile",
    "🔥 Heatmap",
    "📋 Segment Table",
    "🗂️ Raw Data",
])

with tab1:
    st.plotly_chart(
        scatter_2d(df_raw, labels, x_feat, y_feat, seg_names),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(segment_size_bar(profiles, seg_names), use_container_width=True)
    with c2:
        if result.inertia:
            st.metric("KMeans Inertia (WCSS)", f"{result.inertia:,.1f}")
        if result.calinski_harabasz:
            st.metric("Calinski-Harabasz index", f"{result.calinski_harabasz:,.1f}",
                      help="A metric that evaluates\n"
                            "clustering quality by\n"
                           "comparing between-cluster\n"
                           "separation to within-cluster compactness—higher\n"
                           "values indicate better-defined\n"
                            "clusters."
                        )

with tab2:
    if algo == "K-Means" and show_elbow:
        with st.spinner("Computing elbow curve for K = 2 … 10 …"):
            elbow_data = elbow_curve(X_scaled, range(2, 11))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(elbow_plot(elbow_data, k), use_container_width=True)
        with c2:
            st.plotly_chart(silhouette_plot(elbow_data, k), use_container_width=True)

        # Elbow table
        elbow_df = pd.DataFrame({
            "K": elbow_data["k"],
            "Inertia": [round(v, 1) for v in elbow_data["inertia"]],
            "Silhouette": [round(v, 4) if v else None for v in elbow_data["silhouette"]],
        })
        st.dataframe(elbow_df, hide_index=True, use_container_width=True)
    else:
        st.info("Elbow / silhouette curves are available for K-Means only. "
                "Enable 'Compute elbow curve' in the sidebar.")

with tab3:
    profiles_named = profiles.rename(columns={"_Segment": "_Segment"})
    st.plotly_chart(
        radar_chart(profiles, feat_cols, seg_names),
        use_container_width=True,
    )

with tab4:
    st.plotly_chart(
        feature_heatmap(profiles, feat_cols, seg_names),
        use_container_width=True,
    )

with tab5:
    display_profiles = profiles.copy()
    display_profiles.insert(
        0, "Segment Name",
        display_profiles["_Segment"].apply(lambda x: seg_names.get(int(x), f"Segment {int(x)}"))
    )
    display_profiles = display_profiles.drop(columns=["_Segment"])
    st.dataframe(display_profiles, hide_index=True, use_container_width=True)

    st.subheader("💡 Strategy recommendations")
    strategies = {
        0: ("🎯 Upsell & Cross-sell", "High-value customers — offer premium tiers, loyalty rewards, and early access."),
        1: ("💼 Retention campaigns", "Price-sensitive segment — focus on discounts, value bundles, and satisfaction surveys."),
        2: ("🚀 Engagement & growth", "Active but young — gamification, referral programmes, and social proof."),
        3: ("🏅 Loyalty programme", "Long-tenured, moderate spend — nurture with exclusive perks and personalised offers."),
    }
    cols = st.columns(min(4, result.n_clusters))
    for i, col in enumerate(cols):
        with col:
            title, desc = strategies.get(i, (f"Segment {i}", "Investigate this segment further."))
            st.info(f"**{seg_names.get(i, title)}**\n\n{desc}")

with tab6:
    df_export = df_raw.copy()
    df_export["Segment ID"] = labels
    df_export["Segment Name"] = [seg_names.get(int(l), "Noise") for l in labels]

    st.dataframe(df_export, use_container_width=True, height=400)

    csv_bytes = df_export.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download CSV",
        data=csv_bytes,
        file_name="customer_segments.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Built with Streamlit · scikit-learn · Plotly")

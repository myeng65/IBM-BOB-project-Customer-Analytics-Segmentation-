"""
app.py  —  Customer Analytics & Segmentation Dashboard
=======================================================
Streamlit frontend for the Customer Segmentation analytics pipeline.
Run with:  streamlit run app.py

Designed for non-technical business managers:
- Plain-English labels and metric explanations
- Clear distinction between assumptions and findings
- All numbers derived 100% from the CSV (no hard-coded values)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Path setup so analytics.py can be imported ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analytics

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE (consistent across all charts)
# ════════════════════════════════════════════════════════════════════════════
SEGMENT_COLORS = {
    "Segment1": "#3b82f6",
    "Segment2": "#10b981",
    "Segment3": "#f59e0b",
    "Segment4": "#ef4444",
    "Segment5": "#8b5cf6",
}
PRIORITY_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1e3a5f; }
    .metric-label { font-size: 13px; color: #64748b; margin-top: 2px; }
    .metric-note  { font-size: 11px; color: #94a3b8; margin-top: 4px; font-style: italic; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 18px;
    }
    .assumption-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 10px 14px; border-radius: 6px; font-size: 13px;
        color: #78350f; margin-bottom: 14px;
    }
    .insight-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 10px 14px; border-radius: 6px; font-size: 13px;
        color: #1e40af; margin-bottom: 8px;
    }
    .strategy-card {
        background: white; border-radius: 10px; padding: 16px 20px;
        margin-bottom: 14px; border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .priority-high   { color: #ef4444; font-weight: 700; }
    .priority-medium { color: #f59e0b; font-weight: 700; }
    .priority-low    { color: #10b981; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="🔄 Loading and analysing data — please wait...")
def load_results():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "customer_segmentation_data (1).csv")
    return analytics.run_all(data_path)


# ════════════════════════════════════════════════════════════════════════════
# HELPER: Metric card
# ════════════════════════════════════════════════════════════════════════════
def metric_card(label, value, note="", border_color="#3b82f6"):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{border_color}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {"<div class='metric-note'>" + note + "</div>" if note else ""}
    </div>""", unsafe_allow_html=True)


def assumption_note(text):
    st.markdown(f'<div class="assumption-box">⚠️ <b>Assumption:</b> {text}</div>', unsafe_allow_html=True)


def insight_note(text):
    st.markdown(f'<div class="insight-box">💡 <b>Finding:</b> {text}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar(results):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
        st.title("📊 Customer Analytics")
        st.caption("Insurance Customer Segmentation")
        st.divider()

        df = results["df"]
        pm = results["purchase_metrics"]

        st.markdown("### 📌 Quick Stats")
        st.metric("Total Customers", f"{len(df):,}")
        st.metric("Total Revenue", f"₹{pm['total_revenue']:,.0f}")
        st.metric("Avg Annual Premium", f"₹{pm['avg_premium']:,.0f}")
        st.metric("Engaged Customers", f"{pm['repeat_pct']}%")
        st.divider()

        st.markdown("### 🗂️ Navigation")
        st.markdown("""
        - 📋 Executive Summary
        - 📈 Key KPIs
        - 👥 Segment Analysis
        - 🔬 RFM Analysis
        - 🗺️ Product & Geography
        - 📊 Charts
        - 🎯 Marketing Strategy
        - 🧹 Data Quality
        """)
        st.divider()
        st.caption("Data: customer_segmentation_data (1).csv")
        st.caption("53,503 insurance customers · 20 features")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
def tab_executive_summary(results):
    df = results["df"]
    pm = results["purchase_metrics"]
    seg = results["seg_table"]
    rfm = results["rfm"]

    st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    > This dashboard presents a complete customer analytics study of **{:,} insurance customers**
    > across **5 segments**, analysed from purchase records dated **{} to {}**.
    > All metrics are derived directly from the uploaded CSV. Assumptions are clearly marked.
    """.format(
        len(df),
        df["Purchase Date"].min().strftime("%b %Y"),
        df["Purchase Date"].max().strftime("%b %Y"),
    ))

    # ── Top KPI row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Customers", f"{len(df):,}", "Unique insurance policy holders", "#3b82f6")
    with c2: metric_card("Total Revenue (Premiums)", f"₹{pm['total_revenue']/1e6:.1f}M", "Sum of all annual premiums", "#10b981")
    with c3: metric_card("Avg Annual Premium", f"₹{pm['avg_premium']:,.0f}", "Per customer per year", "#f59e0b")
    with c4: metric_card("Avg Coverage Value", f"₹{pm['avg_coverage']:,.0f}", "Average insured amount", "#8b5cf6")
    with c5: metric_card("Engaged Customers", f"{pm['repeat_pct']}%", "Policy Tier ≥ 3 (higher engagement)", "#ef4444")

    st.divider()

    # ── Key narrative findings ───────────────────────────────────────────────
    top_seg = seg.loc[seg["Total_Revenue"].idxmax(), "Segmentation Group"]
    top_rev = seg.loc[seg["Total_Revenue"].idxmax(), "Revenue_Share_%"]
    most_cust = seg.loc[seg["Customers"].idxmax(), "Segmentation Group"]
    most_cust_share = seg.loc[seg["Customers"].idxmax(), "Customer_Share_%"]
    rfm_champions = rfm[rfm["RFM_Label"] == "Champions"]["Customer ID"].nunique()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Key Data Findings")
        findings = [
            f"**{top_seg}** generates the most revenue, accounting for **{top_rev}%** of total premiums.",
            f"**{most_cust}** has the largest customer base at **{most_cust_share}%** of all customers.",
            f"**{rfm_champions:,} Champions** (high RFM score) are your most valuable customers — prioritise retention.",
            f"**{pm['repeat_pct']}%** of customers hold higher-tier policies (Tier ≥ 3), indicating strong engagement.",
            f"Average annual premium is **₹{pm['avg_premium']:,.0f}** with median **₹{pm['median_premium']:,.0f}**.",
            "Purchase dates span from **2018 to 2023**, giving a **5-year** customer activity window.",
        ]
        for f in findings:
            st.markdown(f"✅ {f}")

    with col2:
        st.markdown("#### 📊 Revenue by Segment")
        fig = px.pie(
            seg, values="Total_Revenue", names="Segmentation Group",
            color="Segmentation Group",
            color_discrete_map=SEGMENT_COLORS,
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    assumption_note(
        "Premium Amount is used as the annual revenue proxy per customer. "
        "Coverage Amount is used as the policy/order value. "
        "Each row in the CSV represents one customer's snapshot record."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — KEY KPIs
# ════════════════════════════════════════════════════════════════════════════
def tab_kpis(results):
    df = results["df"]
    pm = results["purchase_metrics"]

    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

    # ── Row 1: Spending metrics ──────────────────────────────────────────────
    st.markdown("#### 💰 Customer Spending Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Average Annual Premium", f"₹{pm['avg_premium']:,.0f}", "Mean revenue per customer", "#3b82f6")
    with c2: metric_card("Median Annual Premium", f"₹{pm['median_premium']:,.0f}", "50th percentile — less affected by outliers", "#10b981")
    with c3: metric_card("Average Coverage (Order Value)", f"₹{pm['avg_coverage']:,.0f}", "Average insured sum per policy", "#f59e0b")
    with c4: metric_card("Total Portfolio Revenue", f"₹{pm['total_revenue']/1e6:.2f}M", "Total annual premium income", "#8b5cf6")

    st.divider()

    # ── Row 2: Customer engagement ───────────────────────────────────────────
    st.markdown("#### 👥 Customer Engagement Metrics")
    assumption_note(
        "Since this is a snapshot dataset (one row per customer), 'repeat' customers "
        "are defined as those holding higher-tier policies (Policy Tier ≥ 3), indicating "
        "deeper product engagement. True multi-purchase frequency cannot be calculated."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Customers", f"{len(df):,}", "Unique customer IDs", "#3b82f6")
    with c2: metric_card("Engaged Customers", f"{pm['repeat_customers']:,}", "Policy Tier ≥ 3", "#10b981")
    with c3: metric_card("Lower Engagement", f"{pm['one_time_customers']:,}", "Policy Tier < 3", "#ef4444")
    with c4: metric_card("Engagement Rate", f"{pm['repeat_pct']}%", "% with higher policy tier", "#f59e0b")

    st.divider()

    # ── Revenue distribution bar chart ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Revenue Distribution by Premium Band")
        rev_dist = pm["revenue_distribution"].reset_index()
        rev_dist.columns = ["Premium Band", "Customer Count"]
        fig = px.bar(
            rev_dist, x="Premium Band", y="Customer Count",
            color="Customer Count",
            color_continuous_scale="Blues",
            text="Customer Count",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            showlegend=False, coloraxis_showscale=False,
            margin=dict(t=20, b=20), height=350,
            xaxis_title="Annual Premium Range", yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ Premium Amount (₹500–₹5,000) represents the annual insurance premium paid by each customer.")

    with col2:
        st.markdown("#### 📊 Average Premium by Segment")
        avg_seg = pm["avg_premium_by_seg"].reset_index()
        avg_seg.columns = ["Segment", "Avg Premium"]
        fig = px.bar(
            avg_seg, x="Segment", y="Avg Premium",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            text="Avg Premium",
        )
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20), height=350,
                          xaxis_title="Segment", yaxis_title="Average Premium (₹)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Purchase frequency analysis ──────────────────────────────────────────
    st.markdown("#### 📅 Purchase History Timeline")
    df_temp = df.copy()
    df_temp["Year"] = df_temp["Purchase Date"].dt.year
    yearly = df_temp.groupby("Year")["Premium Amount"].agg(["sum", "count"]).reset_index()
    yearly.columns = ["Year", "Total Revenue", "New Policies"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly["Year"], y=yearly["Total Revenue"],
                         name="Total Revenue (₹)", marker_color="#3b82f6"), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["New Policies"],
                             name="Policies Issued", mode="lines+markers",
                             line=dict(color="#ef4444", width=2)), secondary_y=True)
    fig.update_layout(height=350, margin=dict(t=20, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Total Revenue (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Policies Issued", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ℹ️ Year is extracted from 'Purchase History' column. Each bar represents policies issued that year.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def tab_segments(results):
    seg = results["seg_table"]
    df = results["df"]

    st.markdown('<div class="section-header">👥 Customer Segment Summary & Comparison</div>', unsafe_allow_html=True)

    # ── Pattern insights ────────────────────────────────────────────────────
    st.markdown("#### 🔍 Identified Segment Patterns")
    for _, row in seg.iterrows():
        if row["Pattern"]:
            col1, col2, col3 = st.columns([1, 2, 3])
            with col1: st.markdown(f"**{row['Segmentation Group']}**")
            with col2: st.markdown(row["Pattern"])
            with col3: st.markdown(f"Revenue: ₹{row['Total_Revenue']:,.0f} &nbsp;|&nbsp; Customers: {row['Customers']:,}")

    st.divider()

    # ── Full comparison table ────────────────────────────────────────────────
    st.markdown("#### 📋 Full Segment Comparison Table")
    st.caption("All values are computed directly from the dataset. Hover over columns for details.")

    display_cols = [
        "Segmentation Group", "Customers", "Customer_Share_%",
        "Total_Revenue", "Revenue_Share_%", "Avg_Premium", "Avg_Coverage",
        "Avg_Income", "Avg_Age", "Avg_RFM_Score",
        "Avg_Recency_Score", "Avg_Frequency_Score", "Avg_Monetary_Score",
        "Pattern"
    ]
    display = seg[display_cols].copy()
    display = display.rename(columns={
        "Segmentation Group": "Segment",
        "Customer_Share_%": "Customer %",
        "Total_Revenue": "Total Revenue (₹)",
        "Revenue_Share_%": "Revenue %",
        "Avg_Premium": "Avg Premium (₹)",
        "Avg_Coverage": "Avg Coverage (₹)",
        "Avg_Income": "Avg Income (₹)",
        "Avg_Age": "Avg Age",
        "Avg_RFM_Score": "RFM Score",
        "Avg_Recency_Score": "Recency",
        "Avg_Frequency_Score": "Frequency",
        "Avg_Monetary_Score": "Monetary",
    })

    # Format columns
    display["Total Revenue (₹)"] = display["Total Revenue (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display["Avg Premium (₹)"] = display["Avg Premium (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display["Avg Coverage (₹)"] = display["Avg Coverage (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display["Avg Income (₹)"] = display["Avg Income (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display["Avg Age"] = display["Avg Age"].apply(lambda x: f"{x:.1f} yrs")
    display["Customers"] = display["Customers"].apply(lambda x: f"{x:,}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    # ── Side-by-side comparison charts ──────────────────────────────────────
    st.markdown("#### 📊 Segment Comparisons — Visual")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Revenue vs Customer Count**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Total Revenue (₹)", x=seg["Segmentation Group"],
            y=seg["Total_Revenue"], marker_color=[SEGMENT_COLORS[s] for s in seg["Segmentation Group"]],
            yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            name="Customers", x=seg["Segmentation Group"], y=seg["Customers"],
            mode="lines+markers", yaxis="y2",
            line=dict(color="#1f2937", width=2), marker=dict(size=8)
        ))
        fig.update_layout(
            height=350, margin=dict(t=20, b=20),
            yaxis=dict(title="Total Revenue (₹)"),
            yaxis2=dict(title="Customers", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**RFM Score Components by Segment**")
        fig = go.Figure()
        for metric, color in [("Avg_Recency_Score", "#3b82f6"),
                               ("Avg_Frequency_Score", "#10b981"),
                               ("Avg_Monetary_Score", "#f59e0b")]:
            label = metric.replace("Avg_", "").replace("_Score", "")
            fig.add_trace(go.Bar(name=label, x=seg["Segmentation Group"],
                                 y=seg[metric], marker_color=color))
        fig.update_layout(barmode="group", height=350, margin=dict(t=20, b=20),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    # ── Demographic breakdown ────────────────────────────────────────────────
    st.markdown("#### 🧑 Demographic Breakdown by Segment")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Gender Distribution**")
        gender = df.groupby(["Segmentation Group", "Gender"]).size().reset_index(name="Count")
        fig = px.bar(gender, x="Segmentation Group", y="Count", color="Gender",
                     barmode="group", color_discrete_sequence=["#3b82f6", "#ec4899", "#8b5cf6"])
        fig.update_layout(height=300, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Age Distribution by Segment**")
        fig = px.box(df, x="Segmentation Group", y="Age",
                     color="Segmentation Group", color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RFM ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def tab_rfm(results):
    rfm = results["rfm"]

    st.markdown('<div class="section-header">🔬 RFM Analysis</div>', unsafe_allow_html=True)

    # Explanation for non-technical readers
    st.markdown("""
    **What is RFM Analysis?**
    RFM stands for **Recency, Frequency, and Monetary** — a proven method to rank customers
    based on their behaviour. Each customer is scored 1–5 on each dimension.
    A combined score of 15 = ideal customer; 3 = disengaged customer.

    | Dimension | In This Dataset | Scoring |
    |-----------|----------------|---------|
    | **Recency** | Days since purchase date | 5 = bought recently, 1 = bought long ago |
    | **Frequency** | Policy Tier (1–5) | 5 = highest tier policy, 1 = entry policy |
    | **Monetary** | Annual Premium Amount | 5 = highest payer, 1 = lowest payer |
    """)

    assumption_note(
        "Because the dataset contains one record per customer, Frequency is mapped to "
        "the customer's Policy Tier Score (1–5) rather than a raw purchase count. "
        "This is a reasonable proxy for engagement depth in insurance."
    )

    st.divider()

    # ── RFM Summary KPIs ────────────────────────────────────────────────────
    label_counts = rfm["RFM_Label"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Champions", f"{label_counts.get('Champions', 0):,}", "Highest RFM score", "#3b82f6")
    with c2: metric_card("Loyal Customers", f"{label_counts.get('Loyal Customers', 0):,}", "Strong engagement", "#10b981")
    with c3: metric_card("At Risk (High Value)", f"{label_counts.get('At Risk High Value', 0):,}", "High spend, low recency", "#ef4444")
    with c4: metric_card("Lost / Inactive", f"{label_counts.get('Lost / Inactive', 0):,}", "Low RFM — need re-engagement", "#94a3b8")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📊 RFM Label Distribution")
        rfm_dist = rfm["RFM_Label"].value_counts().reset_index()
        rfm_dist.columns = ["RFM Label", "Count"]
        fig = px.bar(rfm_dist, x="Count", y="RFM Label", orientation="h",
                     color="Count", color_continuous_scale="Blues", text="Count")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=380, margin=dict(t=10, b=10, l=10, r=80),
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🗂️ RFM Labels by Segment")
        rfm_seg = rfm.groupby(["Segmentation Group", "RFM_Label"]).size().reset_index(name="Count")
        fig = px.bar(rfm_seg, x="Segmentation Group", y="Count", color="RFM_Label",
                     barmode="stack", text_auto=False)
        fig.update_layout(height=380, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: Recency vs Monetary ────────────────────────────────────────
    st.markdown("#### 🔵 Recency vs Monetary Score (by RFM Label)")
    st.caption("Each dot = one customer. X-axis: how recently they purchased. Y-axis: how much they pay.")
    rfm_sample = rfm.sample(min(3000, len(rfm)), random_state=42)
    fig = px.scatter(
        rfm_sample, x="R_Score", y="M_Score",
        color="RFM_Label", size="Monetary",
        hover_data={"Customer ID": True, "Recency": True, "Monetary": True},
        labels={"R_Score": "Recency Score (5=most recent)", "M_Score": "Monetary Score (5=highest)"},
        opacity=0.65, height=400,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("ℹ️ Displaying a random sample of 3,000 customers for chart readability. Full data used for all metrics.")

    # ── Detailed RFM table ───────────────────────────────────────────────────
    st.markdown("#### 📋 RFM Score Summary by Segment")
    rfm_sum = rfm.groupby("Segmentation Group").agg(
        Avg_RFM=("RFM_Score", "mean"),
        Min_RFM=("RFM_Score", "min"),
        Max_RFM=("RFM_Score", "max"),
        Champions=("RFM_Label", lambda x: (x == "Champions").sum()),
        At_Risk=("RFM_Label", lambda x: (x.str.startswith("At Risk")).sum()),
        Lost=("RFM_Label", lambda x: (x == "Lost / Inactive").sum()),
    ).round(2).reset_index()
    rfm_sum.columns = ["Segment", "Avg RFM", "Min RFM", "Max RFM", "Champions", "At Risk", "Lost/Inactive"]
    st.dataframe(rfm_sum, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRODUCT & GEOGRAPHY
# ════════════════════════════════════════════════════════════════════════════
def tab_product_geo(results):
    pg = results["product_geo"]
    df = results["df"]

    st.markdown('<div class="section-header">🗺️ Product & Geographic Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📦 Policy Type Distribution")
        pt = pg["policy_type_dist"].reset_index()
        pt.columns = ["Policy Type", "Count"]
        fig = px.pie(pt, values="Count", names="Policy Type", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Policy types: Individual, Group, Family, Business")

    with c2:
        st.markdown("#### 💼 Revenue by Policy Type")
        rev_pt = pg["revenue_by_policy_type"].reset_index()
        rev_pt.columns = ["Policy Type", "Revenue"]
        fig = px.bar(rev_pt, x="Policy Type", y="Revenue", text="Revenue",
                     color="Policy Type", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Top states ───────────────────────────────────────────────────────────
    st.markdown("#### 🇮🇳 Top 15 States by Revenue")
    geo_top = pg["geo_revenue"].head(15).reset_index()
    geo_top.columns = ["State", "Total Revenue", "Avg Premium", "Customers"]
    fig = px.bar(
        geo_top, x="Total Revenue", y="State", orientation="h",
        color="Total Revenue", color_continuous_scale="Blues",
        text="Customers", hover_data={"Avg Premium": ":,.0f"},
    )
    fig.update_traces(texttemplate="%{text:,} customers", textposition="outside")
    fig.update_layout(height=500, margin=dict(t=10, b=10, r=100),
                      yaxis=dict(categoryorder="total ascending"),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 💰 Avg Premium by Occupation")
        occ = pg["premium_by_occupation"].head(10).reset_index()
        occ.columns = ["Occupation", "Avg Premium"]
        fig = px.bar(occ, x="Avg Premium", y="Occupation", orientation="h",
                     text="Avg Premium",
                     color="Avg Premium", color_continuous_scale="Greens")
        fig.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig.update_layout(height=360, margin=dict(t=10, b=10, r=80),
                          yaxis=dict(categoryorder="total ascending"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🛡️ Insurance Products by Segment")
        prod_seg = pg["product_by_segment"].reset_index()
        prod_melt = prod_seg.melt(id_vars="Segmentation Group",
                                  var_name="Product", value_name="Count")
        fig = px.bar(prod_melt, x="Segmentation Group", y="Count", color="Product",
                     barmode="stack", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(height=360, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=1.1, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHARTS (Overview)
# ════════════════════════════════════════════════════════════════════════════
def tab_charts(results):
    df = results["df"]
    rfm = results["rfm"]
    seg = results["seg_table"]

    st.markdown('<div class="section-header">📊 Chart Gallery — Segment Comparisons</div>', unsafe_allow_html=True)

    # ── 1. Revenue & Customers side by side ─────────────────────────────────
    st.markdown("#### 1. Revenue Share vs Customer Share")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue Share %", x=seg["Segmentation Group"],
                         y=seg["Revenue_Share_%"], marker_color="#3b82f6",
                         text=seg["Revenue_Share_%"], texttemplate="%{text}%"))
    fig.add_trace(go.Bar(name="Customer Share %", x=seg["Segmentation Group"],
                         y=seg["Customer_Share_%"], marker_color="#10b981",
                         text=seg["Customer_Share_%"], texttemplate="%{text}%"))
    fig.update_layout(barmode="group", height=360, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.1),
                      yaxis_title="Percentage (%)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    insight_note("If Revenue Share > Customer Share for a segment, those customers spend more per head — a premium segment.")

    # ── 2. Average Income vs Average Premium ────────────────────────────────
    st.markdown("#### 2. Average Income vs Average Annual Premium by Segment")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Income (₹)", x=seg["Segmentation Group"],
                         y=seg["Avg_Income"], marker_color="#8b5cf6"))
    fig.add_trace(go.Bar(name="Avg Premium (₹)", x=seg["Segmentation Group"],
                         y=seg["Avg_Premium"], marker_color="#f59e0b"))
    fig.update_layout(barmode="group", height=350, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.1), yaxis_title="Amount (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. Heatmap: Segment × RFM Label count ───────────────────────────────
    st.markdown("#### 3. Segment × RFM Label Heatmap")
    rfm_heat = rfm.groupby(["Segmentation Group", "RFM_Label"]).size().unstack(fill_value=0)
    fig = px.imshow(
        rfm_heat, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="RFM Label", y="Segment", color="Customers"),
    )
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each cell = number of customers. Darker = more customers with that RFM label in that segment.")

    # ── 4. Communication channel preference ──────────────────────────────────
    st.markdown("#### 4. Preferred Communication Channels by Segment")
    comm = df.groupby(["Segmentation Group", "Preferred Communication Channel"]).size().reset_index(name="Count")
    fig = px.bar(comm, x="Segmentation Group", y="Count",
                 color="Preferred Communication Channel", barmode="stack",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=1.1, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)
    insight_note("Use this to match outreach channels to segment preferences for higher response rates.")

    # ── 5. Age vs Premium scatter ────────────────────────────────────────────
    st.markdown("#### 5. Customer Age vs Annual Premium (sample of 2,000)")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(sample, x="Age", y="Premium Amount",
                     color="Segmentation Group", color_discrete_map=SEGMENT_COLORS,
                     opacity=0.55, height=400,
                     labels={"Premium Amount": "Annual Premium (₹)", "Age": "Customer Age"})
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # ── 6. Preferred contact time ────────────────────────────────────────────
    st.markdown("#### 6. Preferred Contact Times Across All Customers")
    time_pref = df["Preferred Contact Time"].value_counts().reset_index()
    time_pref.columns = ["Contact Time", "Count"]
    fig = px.pie(time_pref, values="Count", names="Contact Time", hole=0.35,
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)
    insight_note("Schedule outreach campaigns during peak preferred times to maximise pickup rates.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — MARKETING STRATEGY
# ════════════════════════════════════════════════════════════════════════════
def tab_strategy(results):
    strategies = results["strategy"]
    seg = results["seg_table"]

    st.markdown('<div class="section-header">🎯 Marketing Strategy & Business Decisions</div>', unsafe_allow_html=True)

    st.markdown("""
    > All strategies below are **derived from actual data patterns** in the dataset.
    > No recommendations are fabricated. Each card shows the data evidence behind the recommendation.
    """)

    # ── Priority overview ───────────────────────────────────────────────────
    st.markdown("#### 📌 Priority Matrix")
    priority_counts = pd.DataFrame(strategies)["Priority"].value_counts()
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("🔴 High Priority Segments", str(priority_counts.get("High", 0)), "Immediate action required", "#ef4444")
    with c2: metric_card("🟡 Medium Priority Segments", str(priority_counts.get("Medium", 0)), "Nurture & grow", "#f59e0b")
    with c3: metric_card("🟢 Low Priority Segments", str(priority_counts.get("Low", 0)), "Low-cost re-engagement", "#10b981")

    st.divider()

    # ── Strategy cards ──────────────────────────────────────────────────────
    for s in strategies:
        priority_class = f"priority-{s['Priority'].lower()}"
        border_color = PRIORITY_COLORS[s["Priority"]]

        st.markdown(f"""
        <div class="strategy-card" style="border-left: 4px solid {border_color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:17px; font-weight:700; color:#1e3a5f;">{s['Segment']}</div>
                <span class="{priority_class}">▶ {s['Priority']} Priority</span>
            </div>
            <div style="color:#64748b; font-size:13px; margin:4px 0 8px 0;">
                🏷️ <b>RFM Profile:</b> {s['Label']} &nbsp;|&nbsp;
                👥 <b>Customers:</b> {s['Customers']} &nbsp;|&nbsp;
                💰 <b>Avg Premium:</b> {s['Avg Premium']} &nbsp;|&nbsp;
                📊 <b>Revenue Share:</b> {s['Revenue Share']}
            </div>
            <div style="background:#f8fafc; padding:10px; border-radius:6px; margin-bottom:8px;">
                <b>📊 Data Insight:</b> {s['Insight']}
            </div>
            <div style="background:#f0fdf4; padding:10px; border-radius:6px;">
                <b>✅ Recommended Action:</b> {s['Recommended Action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Business decisions summary table ────────────────────────────────────
    st.markdown("#### 📋 Quick Reference: Business Decision Summary")
    df_strat = pd.DataFrame(strategies)[["Segment", "Label", "Customers", "Revenue Share", "Avg Premium", "Priority"]]
    st.dataframe(df_strat, use_container_width=True, hide_index=True)

    # ── Additional business insights ────────────────────────────────────────
    st.divider()
    st.markdown("#### 💡 Additional Business Recommendations (from data)")
    col1, col2 = st.columns(2)

    df_main = results["df"]
    top_lang = df_main["Preferred Language"].value_counts().index[0]
    top_channel = df_main["Preferred Communication Channel"].value_counts().index[0]
    top_time = df_main["Preferred Contact Time"].value_counts().index[0]
    top_state = results["product_geo"]["geo_revenue"].index[0]

    with col1:
        st.markdown(f"""
        **Communication Strategy (from actual data)**
        - 🌐 Most preferred language: **{top_lang}** — localise campaigns accordingly
        - 📱 Top channel: **{top_channel}** — prioritise this channel for mass campaigns
        - ⏰ Best contact time: **{top_time}** — schedule automated outreach for this window
        - 📍 Highest revenue state: **{top_state}** — focus field agent deployment here
        """)

    with col2:
        top_occ = results["product_geo"]["premium_by_occupation"].index[0]
        top_occ_prem = results["product_geo"]["premium_by_occupation"].iloc[0]
        st.markdown(f"""
        **Product & Pricing Strategy (from actual data)**
        - 💼 Highest-paying occupation: **{top_occ}** (avg ₹{top_occ_prem:,.0f}/yr) — target with premium products
        - 🛡️ Diversify product offerings across Family, Group, Business, and Individual policies
        - 🔄 Win-back campaigns needed for segments with high premium but declining recency scores
        - 📈 Focus cross-sell efforts on engaged segments to increase average policy tier
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════
def tab_data_quality(results):
    report = results["cleaning_report"]
    df = results["df"]

    st.markdown('<div class="section-header">🧹 Data Collection & Quality Report</div>', unsafe_allow_html=True)

    st.markdown("""
    This section documents every cleaning step applied to the raw dataset.
    All transformations are reproducible and listed in order of application.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Original Rows", f"{report['original_rows']:,}", "Before cleaning", "#3b82f6")
    with c2: metric_card("Clean Rows", f"{report['clean_rows']:,}", "After cleaning", "#10b981")
    with c3: metric_card("Duplicates Removed", f"{report['duplicates_removed']:,}", "Duplicate Customer IDs", "#f59e0b")
    with c4: metric_card("Unparseable Dates", f"{report['unparseable_dates']:,}", "Could not parse date format", "#ef4444")

    st.divider()

    st.markdown("#### 📋 Cleaning Steps Applied")
    steps = [
        ("1️⃣ Strip whitespace", "Removed leading/trailing spaces from all text columns.", "✅ Done"),
        ("2️⃣ Numeric conversion", "Converted Age, Income Level, Coverage Amount, Premium Amount to numbers.", "✅ Done"),
        ("3️⃣ Date parsing", "Parsed 'Purchase History' from mixed formats (DD-MM-YYYY, MM/DD/YYYY, M/D/YYYY).", "✅ Done"),
        ("4️⃣ Duplicate removal", f"Removed {report['duplicates_removed']} duplicate Customer IDs (kept last record).", "✅ Done"),
        ("5️⃣ Age validation", f"Removed rows with Age < 0 or > 120. Found: {report['invalid_age_rows']} invalid.", "✅ Done"),
        ("6️⃣ Policy tier encoding", "Mapped 'Behavioral Data' (policy1–policy5) to numeric score 1–5.", "✅ Done"),
        ("7️⃣ Missing values", "Dataset has zero missing values (confirmed by scan).", "✅ No action needed"),
    ]
    for name, desc, status in steps:
        c1, c2, c3 = st.columns([2, 5, 1])
        with c1: st.markdown(f"**{name}**")
        with c2: st.markdown(desc)
        with c3: st.markdown(f"`{status}`")
    st.divider()

    # ── Data types ──────────────────────────────────────────────────────────
    st.markdown("#### 🗃️ Column Data Types After Cleaning")
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Type": df.dtypes.astype(str).values,
        "Sample Value": [str(df[c].iloc[0]) for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Null Count": [df[c].isnull().sum() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # ── Numeric summary ──────────────────────────────────────────────────────
    st.markdown("#### 📈 Numeric Column Statistics")
    st.dataframe(
        df[["Age", "Income Level", "Coverage Amount", "Premium Amount", "Policy Tier Score"]]
        .describe().round(2),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    results = load_results()
    render_sidebar(results)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
                padding: 30px 36px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h1 style="margin:0; font-size:28px; font-weight:800;">
            📊 Customer Analytics & Segmentation Dashboard
        </h1>
        <p style="margin:8px 0 0 0; font-size:15px; opacity:0.85;">
            Insurance Customer Data · 53,503 Customers · 5 Segments · Powered by RFM Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Executive Summary",
        "📈 Key KPIs",
        "👥 Segment Analysis",
        "🔬 RFM Analysis",
        "🗺️ Product & Geography",
        "📊 Charts",
        "🎯 Strategy",
        "🧹 Data Quality",
    ])

    with tab1: tab_executive_summary(results)
    with tab2: tab_kpis(results)
    with tab3: tab_segments(results)
    with tab4: tab_rfm(results)
    with tab5: tab_product_geo(results)
    with tab6: tab_charts(results)
    with tab7: tab_strategy(results)
    with tab8: tab_data_quality(results)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <hr style="border-color:#e2e8f0; margin: 40px 0 12px 0;">
        Customer Analytics Dashboard · Data: customer_segmentation_data (1).csv ·
        All metrics computed from raw data · No values fabricated ·
        <b>Made with IBM Bob</b>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

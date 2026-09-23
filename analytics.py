"""
analytics.py
============
Core analytics engine for Customer Segmentation project.
All calculations are performed here and returned as plain Python dicts / DataFrames
so the Streamlit UI stays purely presentational.

Dataset: customer_segmentation_data (1).csv
- 53,503 customers, 20 columns
- Insurance domain (Premium Amount = revenue proxy, Coverage Amount = order value proxy)
- RFM is derived from: Purchase History (Recency), Behavioral Data policy tier (Frequency proxy),
  Premium Amount (Monetary)
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Dataset path
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "customer_segmentation_data (1).csv")


# ============================================================================
# 1.  LOAD
# ============================================================================
def load_data(path=DATA_PATH):
    """Load the CSV and return a raw DataFrame."""
    return pd.read_csv(path)


# ============================================================================
# 2.  CLEAN
# ============================================================================
def _parse_date(val):
    """Try multiple date formats and return a Timestamp or NaT."""
    for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return pd.NaT


def clean_data(df):
    """
    Clean the dataset and return (cleaned_df, cleaning_report).

    Cleaning steps:
    1. Strip whitespace from all string columns.
    2. Convert numeric columns to correct dtypes.
    3. Parse 'Purchase History' (mixed date formats) into datetime.
    4. Remove duplicates on Customer ID.
    5. Remove out-of-range age values (< 0 or > 120).
    6. Map Behavioral Data policy tier to a numeric score (1-5).
    """
    report = {}
    original_rows = len(df)

    # 1. Strip whitespace
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2. Numeric types
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Income Level"] = pd.to_numeric(df["Income Level"], errors="coerce")
    df["Coverage Amount"] = pd.to_numeric(df["Coverage Amount"], errors="coerce")
    df["Premium Amount"] = pd.to_numeric(df["Premium Amount"], errors="coerce")

    # 3. Parse purchase date (mixed formats)
    df["Purchase Date"] = df["Purchase History"].apply(_parse_date)
    unparseable = int(df["Purchase Date"].isna().sum())
    report["unparseable_dates"] = unparseable
    df = df.dropna(subset=["Purchase Date"])

    # 4. Duplicates on Customer ID
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["Customer ID"], keep="last")
    report["duplicates_removed"] = before_dedup - len(df)

    # 5. Age range
    invalid_age = int(df[(df["Age"] < 0) | (df["Age"] > 120)].shape[0])
    report["invalid_age_rows"] = invalid_age
    df = df[(df["Age"] >= 0) & (df["Age"] <= 120)]

    # 6. Policy tier score (1-5)
    policy_map = {"policy1": 1, "policy2": 2, "policy3": 3, "policy4": 4, "policy5": 5}
    df["Policy Tier Score"] = df["Behavioral Data"].map(policy_map).fillna(3).astype(int)
    df["Product Tier Score"] = df["Insurance Products Owned"].map(policy_map).fillna(3).astype(int)

    report["original_rows"] = original_rows
    report["clean_rows"] = len(df)
    report["missing_values"] = df.isnull().sum().to_dict()

    return df.reset_index(drop=True), report


# ============================================================================
# 3.  CUSTOMER SPENDING & PURCHASE METRICS
# ============================================================================
def compute_purchase_metrics(df):
    """
    Returns purchase frequency, avg spending, avg order value, repeat vs one-time,
    and revenue distribution.

    ASSUMPTION: 'Premium Amount' = annual revenue per customer.
    'Coverage Amount' = insurance policy order value.
    A customer with Policy Tier Score >= 3 is considered 'engaged/repeat' because
    higher-tier policies indicate a deeper product relationship.
    """
    metrics = {}
    metrics["avg_premium"] = round(float(df["Premium Amount"].mean()), 2)
    metrics["avg_coverage"] = round(float(df["Coverage Amount"].mean()), 2)
    metrics["total_revenue"] = round(float(df["Premium Amount"].sum()), 2)
    metrics["median_premium"] = round(float(df["Premium Amount"].median()), 2)

    repeat = df[df["Policy Tier Score"] >= 3]
    one_time = df[df["Policy Tier Score"] < 3]
    metrics["repeat_customers"] = int(len(repeat))
    metrics["one_time_customers"] = int(len(one_time))
    metrics["repeat_pct"] = round(len(repeat) / len(df) * 100, 1)

    metrics["revenue_by_segment"] = (
        df.groupby("Segmentation Group")["Premium Amount"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
    )
    metrics["avg_premium_by_seg"] = (
        df.groupby("Segmentation Group")["Premium Amount"].mean().round(2)
    )
    metrics["count_by_segment"] = df["Segmentation Group"].value_counts()

    # Revenue distribution buckets
    bins = [0, 1000, 2000, 3000, 4000, 5001]
    labels = ["500-1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k"]
    df["Premium Bucket"] = pd.cut(df["Premium Amount"], bins=bins, labels=labels)
    metrics["revenue_distribution"] = df["Premium Bucket"].value_counts().sort_index()

    return metrics


# ============================================================================
# 4.  RFM ANALYSIS
# ============================================================================
def compute_rfm(df):
    """
    RFM (Recency, Frequency, Monetary) Analysis.

    Definitions mapped to insurance dataset:
    - Recency   : Days since last purchase date. Lower = better. Scored 5 (recent) to 1 (old).
    - Frequency : Policy Tier Score (1-5). Higher tier = more engaged customer.
    - Monetary  : Premium Amount. Higher = more revenue. Scored in quintiles 1-5.

    ASSUMPTION: Each row = one customer's most recent policy record.
    """
    snapshot_date = df["Purchase Date"].max() + pd.Timedelta(days=1)

    rfm = df[["Customer ID", "Segmentation Group", "Purchase Date",
              "Policy Tier Score", "Premium Amount"]].copy()
    rfm["Recency"] = (snapshot_date - rfm["Purchase Date"]).dt.days
    rfm["Frequency"] = rfm["Policy Tier Score"]
    rfm["Monetary"] = rfm["Premium Amount"]

    # Score 1-5 (5 = best)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = rfm["Frequency"]  # already 1-5
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def rfm_label(row):
        score = row["RFM_Score"]
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]
        if score >= 13:
            return "Champions"
        elif score >= 10 and r >= 3:
            return "Loyal Customers"
        elif r >= 4 and score >= 9:
            return "Potential Loyalists"
        elif r >= 4 and score < 9:
            return "Recent Customers"
        elif r <= 2 and f >= 4 and m >= 4:
            return "At Risk High Value"
        elif r <= 2 and score >= 8:
            return "At Risk"
        elif r <= 2 and score < 8:
            return "Lost / Inactive"
        elif f <= 2 and m <= 2:
            return "Low Value"
        else:
            return "Need Attention"

    rfm["RFM_Label"] = rfm.apply(rfm_label, axis=1)
    return rfm


# ============================================================================
# 5.  PRODUCT & GEOGRAPHIC ANALYSIS
# ============================================================================
def compute_product_geo(df):
    """Product mix and geographic revenue breakdown."""
    results = {}
    results["policy_type_dist"] = df["Policy Type"].value_counts()
    results["product_by_segment"] = (
        df.groupby(["Segmentation Group", "Insurance Products Owned"])
        .size()
        .unstack(fill_value=0)
    )
    geo_rev = (
        df.groupby("Geographic Information")["Premium Amount"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "Total Revenue", "mean": "Avg Premium", "count": "Customers"})
        .sort_values("Total Revenue", ascending=False)
    )
    results["geo_revenue"] = geo_rev
    results["premium_by_occupation"] = (
        df.groupby("Occupation")["Premium Amount"].mean().sort_values(ascending=False).round(2)
    )
    results["revenue_by_policy_type"] = (
        df.groupby("Policy Type")["Premium Amount"].sum().sort_values(ascending=False).round(2)
    )
    return results


# ============================================================================
# 6.  SEGMENT COMPARISON TABLE
# ============================================================================
def build_segment_comparison(df, rfm):
    """
    One row per segment with:
    - Customer count and share
    - Revenue metrics
    - RFM averages
    - Demographics
    - Pattern flags
    """
    seg = df.groupby("Segmentation Group").agg(
        Customers=("Customer ID", "count"),
        Total_Revenue=("Premium Amount", "sum"),
        Avg_Premium=("Premium Amount", "mean"),
        Avg_Coverage=("Coverage Amount", "mean"),
        Avg_Income=("Income Level", "mean"),
        Avg_Age=("Age", "mean"),
    ).round(2)

    seg["Revenue_Share_%"] = (seg["Total_Revenue"] / seg["Total_Revenue"].sum() * 100).round(1)
    seg["Customer_Share_%"] = (seg["Customers"] / seg["Customers"].sum() * 100).round(1)

    rfm_avg = rfm.groupby("Segmentation Group")[["R_Score", "F_Score", "M_Score", "RFM_Score"]].mean().round(2)
    rfm_avg.columns = ["Avg_Recency_Score", "Avg_Frequency_Score", "Avg_Monetary_Score", "Avg_RFM_Score"]
    seg = seg.join(rfm_avg)

    max_rev_seg = seg["Total_Revenue"].idxmax()
    max_cust_seg = seg["Customers"].idxmax()

    seg["Pattern"] = ""
    seg.loc[max_rev_seg, "Pattern"] = "Most Revenue"
    if max_cust_seg == max_rev_seg:
        seg.loc[max_cust_seg, "Pattern"] += " | Most Customers"
    else:
        seg.loc[max_cust_seg, "Pattern"] = "Most Customers"

    high_spend_low_rec = seg[
        (seg["Avg_Premium"] > seg["Avg_Premium"].median()) &
        (seg["Avg_Recency_Score"] < seg["Avg_Recency_Score"].median())
    ]
    for idx in high_spend_low_rec.index:
        current = seg.loc[idx, "Pattern"]
        seg.loc[idx, "Pattern"] = (current + " | High Spend, Declining Activity").lstrip(" | ")

    newest_seg = seg["Avg_Recency_Score"].idxmax()
    current = seg.loc[newest_seg, "Pattern"]
    seg.loc[newest_seg, "Pattern"] = (current + " | Most Recently Acquired").lstrip(" | ")

    return seg.reset_index()


# ============================================================================
# 7.  MARKETING STRATEGY
# ============================================================================
def generate_marketing_strategy(seg_table, rfm):
    """
    Strategy cards based on RELATIVE ranking within the dataset.

    NOTE: The pre-labelled segments in this dataset have very similar average RFM scores
    (all approximately 8.7-8.9 out of 15) because the source segments are balanced.
    Strategy is therefore based on relative rankings: revenue rank, customer size,
    average premium, and recency rank. All insights reference actual computed values.
    """
    strategies = []

    t = seg_table.copy().reset_index(drop=True)
    t["rev_rank"]  = t["Total_Revenue"].rank(ascending=False).astype(int)
    t["prem_rank"] = t["Avg_Premium"].rank(ascending=False).astype(int)
    t["cust_rank"] = t["Customers"].rank(ascending=False).astype(int)
    t["rec_rank"]  = t["Avg_Recency_Score"].rank(ascending=False).astype(int)  # 1 = best recency

    for _, row in t.iterrows():
        seg           = row["Segmentation Group"]
        avg_premium   = float(row["Avg_Premium"])
        avg_recency   = float(row["Avg_Recency_Score"])
        customers     = int(row["Customers"])
        revenue_share = float(row["Revenue_Share_%"])
        cust_share    = float(row["Customer_Share_%"])
        total_rev     = float(row["Total_Revenue"])
        rev_rank      = int(row["rev_rank"])
        prem_rank     = int(row["prem_rank"])
        cust_rank     = int(row["cust_rank"])
        rec_rank      = int(row["rec_rank"])

        if rev_rank == 1:
            label    = "Revenue Champion"
            insight  = (
                f"{seg} is the #1 revenue-generating segment: Rs{total_rev:,.0f} total "
                f"({revenue_share}% of portfolio), {customers:,} customers, "
                f"avg premium Rs{avg_premium:,.0f}/yr. Recency rank: #{rec_rank}/5."
            )
            action   = (
                "Protect and grow this segment. Launch a VIP loyalty programme, dedicated "
                "relationship managers, and priority claims service. Identify the top 20% "
                "by premium for referral incentives and white-glove treatment."
            )
            priority = "High"

        elif cust_rank == 1:
            gap   = revenue_share - cust_share
            label = "Volume Leader - Upsell Opportunity" if gap < 0 else "Volume Leader"
            insight = (
                f"{seg} is the largest segment: {customers:,} customers ({cust_share}% of base). "
                f"Revenue share is {revenue_share}% -- "
                f"{'under-spending by' if gap < 0 else 'contributing'} {abs(gap):.1f}pp "
                f"vs customer share. Avg premium Rs{avg_premium:,.0f}/yr, recency rank #{rec_rank}/5."
            )
            action = (
                "Upsell campaign: offer policy tier upgrades and coverage add-ons. "
                "A 10% premium lift in this segment alone would significantly boost portfolio revenue. "
                "Use preferred communication channels and languages for personalised outreach."
                if gap < 0 else
                "Maintain high engagement with loyalty rewards and auto-renewal incentives. "
                "Cross-sell complementary products. Focus on retention to protect revenue."
            )
            priority = "High"

        elif prem_rank == 1 and rec_rank >= 3:
            label   = "High Value, Declining Activity"
            insight = (
                f"{seg} has the highest avg premium (Rs{avg_premium:,.0f}/yr, rank #1) "
                f"but recency rank is #{rec_rank}/5 -- this high-value group shows signs of "
                f"reduced engagement. Revenue share: {revenue_share}%."
            )
            action  = (
                "Win-back campaign: personalised renewal discounts, proactive outreach 60 days "
                "before policy expiry, and premium upgrade bundles. Losing these customers "
                "has an outsized negative impact on total revenue."
            )
            priority = "High"

        elif rev_rank <= 3 and rec_rank <= 2:
            label   = "Active Growth Segment"
            insight = (
                f"{seg} ranks #{rev_rank} by revenue (Rs{total_rev:,.0f}, {revenue_share}%) "
                f"with strong recency (rank #{rec_rank}/5). Avg premium Rs{avg_premium:,.0f}/yr."
            )
            action  = (
                "Cross-sell additional insurance products (health, life, vehicle). "
                "Introduce referral programmes. A/B test personalised vs bulk outreach "
                "to optimise conversion rates."
            )
            priority = "Medium"

        else:
            label   = "Nurture and Develop"
            insight = (
                f"{seg} contributes {revenue_share}% of revenue ({customers:,} customers). "
                f"Avg premium Rs{avg_premium:,.0f}/yr, recency rank #{rec_rank}/5. "
                f"Revenue and engagement are currently at or below the portfolio median."
            )
            action  = (
                "Send educational content on policy benefits and claims success stories. "
                "Offer first-renewal discounts and entry-level bundles. "
                "Use low-cost channels (email/SMS) to keep customer acquisition cost low "
                "while building product familiarity."
            )
            priority = "Medium"

        strategies.append({
            "Segment": seg,
            "Label": label,
            "Customers": f"{customers:,}",
            "Revenue Share": f"{revenue_share}%",
            "Avg Premium": f"Rs{avg_premium:,.0f}",
            "Insight": insight,
            "Recommended Action": action,
            "Priority": priority,
        })

    return sorted(strategies, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x["Priority"]])


# ============================================================================
# MASTER RUNNER
# ============================================================================
def run_all(path=DATA_PATH):
    """Run the full pipeline and return all results as a dict of named artefacts."""
    df_raw = load_data(path)
    df, cleaning_report = clean_data(df_raw.copy())
    purchase_metrics = compute_purchase_metrics(df)
    rfm = compute_rfm(df)
    product_geo = compute_product_geo(df)
    seg_table = build_segment_comparison(df, rfm)
    strategy = generate_marketing_strategy(seg_table, rfm)

    return {
        "df": df,
        "cleaning_report": cleaning_report,
        "purchase_metrics": purchase_metrics,
        "rfm": rfm,
        "product_geo": product_geo,
        "seg_table": seg_table,
        "strategy": strategy,
    }

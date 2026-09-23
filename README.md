# Customer Analytics & Segmentation Dashboard

A complete Python data analytics project for insurance customer segmentation,
built on top of `customer_segmentation_data (1).csv` (53,503 customers, 20 columns).
## Project Structure

```
customer_analytics/analytics.py      # Core analytics engine — all calculations
  app.py
# Stream lit dashboard frontend requirements.txt
# Python dependencies README.md
# This file
## What the Dashboard Contains

| Tab | Contents |
|-----|----------|
| Executive Summary | Key findings, revenue pie chart, top metrics |
| Key KPIs | Avg premium, coverage, engagement, timeline |
| Segment Analysis | Comparison table, patterns, demographics |
| RFM Analysis | Recency/Frequency/Monetary scoring, heatmaps |
| Product & Geography | Policy types, top states, occupations |
| Charts | Revenue share, RFM heatmap, channel preferences |
| Marketing Strategy | Data-driven strategy cards per segment |
| Data Quality | Cleaning steps, dtypes, statistics |

## Analytics Methods

### Purchase Metrics
- **Revenue proxy**: Premium Amount (annual insurance premium)
- **Order value proxy**: Coverage Amount (sum insured)
- **Engaged customers**: Policy Tier Score >= 3 (ASSUMPTION: higher tier = deeper relationship)
**ASSUMPTION**: Frequency uses Policy Tier as proxy because the dataset is a
cross-sectional snapshot (one record per customer, not multi-transaction history).





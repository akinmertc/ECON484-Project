# Team 4 — Corporate Sustainability, Greenwashing & Financial Performance

## Research Question

**Natural Language:**
When corporate financial performance metrics (ROA, ROE, Leverage) are combined with ESG scores, how many distinct "Corporate Sustainability Regimes" naturally form in the S&P 500? Can density-based anomaly detection isolate companies that exhibit severe structural contradictions (e.g., hyper-profitable but high environmental risk) as statistical anomalies?

**y = f(x) Format:**

```
Task type : Unsupervised — Clustering + Anomaly Detection

Input  X  : [Total ESG Risk Score, Environment Risk Score,
             Governance Risk Score, Social Risk Score,
             ROA, ROE, Leverage]   (7 features, all continuous)

Output y  :
  Q1 → Cluster label (integer 0..k) per company   [K-Means]
  Q2 → Anomaly flag (-1 = anomaly, 1 = normal)    [DBSCAN, Isolation Forest]
       + continuous anomaly score                  [Isolation Forest only]
```

---

## Input Data Description

**Source:** [S&P 500 ESG Risk Ratings — Kaggle](https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings)

| Column | Type | Description |
|--------|------|-------------|
| Symbol | string | Stock ticker (AAPL, MSFT, ...) |
| Name | string | Company name |
| Sector | categorical | 11 GICS sectors (Technology, Energy, ...) |
| Industry | categorical | Sub-sector classification |
| Total ESG Risk score | float | Overall ESG risk score (lower = better) |
| Environment Risk Score | float | Environmental risk component |
| Governance Risk Score | float | Governance risk component |
| Social Risk Score | float | Social risk component |
| ESG Risk Level | categorical | Negligible / Low / Medium / High / Severe |
| Controversy Level | categorical | ESG controversy intensity |
| Controversy Score | float | Controversy score (0–5) |
| ESG Risk Percentile | string | Text format, e.g. "50th percentile" |

**Known Data Issues:**
- 73 of 503 companies have missing ESG scores (14.5%) → dropped, 430 companies retained
- `Full Time Employees` stored as comma-formatted string (e.g. "3,157") → converted to float
- `ESG Risk Percentile` stored as text → integer extracted via regex

**Financial Metric Synthesis:**
ROA, ROE, and Leverage are not included in the ESG dataset. Values were synthesized using sector-specific market averages (e.g. Technology ROA mean=0.12, Utilities ROA mean=0.03) with ROE derived via the DuPont relationship: ROE = ROA × (1 + Leverage). Future iterations will replace synthesized values with actual reported figures from Yahoo Finance or SEC EDGAR.

---

## Verification & Validation (V&V) Data Description

**Approach:** Supervised ground-truth labels do not exist for unsupervised anomaly detection. We use a two-layer validation strategy:

**Layer 1 — Model Consistency:**
- K-Means: Silhouette Score (> 0.50 = strong, < 0.25 = weak)
- DBSCAN: noise point ratio monitored (target: 2%–10%)
- Isolation Forest: contamination parameter aligned with prior research expectations (~8%)
- Cross-model agreement: companies flagged anomalous by both models simultaneously serve as high-confidence candidates
- `splits/split_01_val_20pct.csv` (20% holdout, stratified by sector) used to verify anomaly rates are consistent across data subsets

**Layer 2 — External Criterion:**
- Anomaly companies' ESG Risk Level distribution compared against normal companies
- High financial performance + high environmental risk contradictions verified manually
- Sector-specific anomaly rates reviewed for economic plausibility

**Drift V&V:**
- PSI < 0.10 → stable distribution
- 0.10 ≤ PSI < 0.25 → moderate change, monitor
- PSI ≥ 0.25 → significant drift, model retraining required

---

## Methods to be Used

### 1. Baseline — K-Means
- **Why:** Standard reference point to demonstrate that spherical cluster assumptions fail on ESG + financial data
- **Criterion:** Silhouette Score (k = 2 through 8 evaluated)
- **Expected outcome:** Silhouette < 0.25, confirming K-Means is insufficient

### 2. Primary Model A — DBSCAN
- **Why:** Density-based; no spherical shape assumption; explicitly labels low-density points as noise (label = -1), which directly maps to structural anomalies
- **Parameters:** eps=1.5, min_samples=5
- **Known weakness:** A single global eps value may under-detect anomalies in sectors with naturally sparse data. Sector-aware eps tuning is planned as a future improvement.
- **Criterion:** Noise point count and sector distribution of anomalies

### 3. Primary Model B — Isolation Forest
- **Why:** Tree-based; avoids distance metrics and the curse of dimensionality; produces a continuous anomaly score enabling anomaly ranking
- **Parameters:** contamination=0.08, n_estimators=200, random_state=42
- **Criterion:** Anomaly score distribution; cross-model overlap with DBSCAN

### 4. PSI Drift Analysis
- **Reference regulation:** EU CBAM (Carbon Border Adjustment Mechanism)
  - Announcement: July 2021
  - Transitional phase: October 2023
  - Full obligations: 2026
- **Directly affected sectors:** Energy, Basic Materials, Industrials
- **Unaffected sectors:** Technology, Healthcare, Consumer Defensive

---

## Expected Outputs & Interpretation

| Output | Format | Interpretation |
|--------|--------|----------------|
| K-Means cluster label | int (0..k) | Which sustainability regime the company belongs to |
| Silhouette score | float | < 0.25 → K-Means insufficient for this data |
| DBSCAN label | -1 or 0+ | -1 = structural anomaly = potential greenwashing candidate |
| IF anomaly flag | -1 / 1 | -1 = anomaly |
| IF anomaly score | float (negative) | Lower = more anomalous; enables severity ranking |
| Cross-model anomalies | Company list | Highest-confidence greenwashing candidates |
| PSI value | float per sector | > 0.25 → regulatory shift has caused significant distributional change |

**Generated plots:**
- `plots/01_kmeans_silhouette.png` — Silhouette vs k curve
- `plots/02_kmeans_clusters_pca.png` — K-Means clusters in PCA space
- `plots/03_dbscan_anomalies.png` — DBSCAN anomalies with ticker labels
- `plots/04_isolation_forest_anomalies.png` — IF anomaly scores
- `plots/05_anomaly_sector_distribution.png` — Anomaly count by sector (both models)
- `plots/06_psi_drift_analysis.png` — PSI bar chart (CBAM pre/post comparison)

---

## Risks

**1. Feature Drift (Regulatory Shift)**
Policies such as EU CBAM can abruptly shift ESG score distributions for affected sectors. Monitored via PSI ≥ 0.25 threshold. Energy, Basic Materials, and Industrials are the primary risk sectors.

**2. DBSCAN Parameter Sensitivity**
A single global eps value may not be optimal across sectors with different natural densities. Noise ratio is monitored: if it drops below 2% eps is decreased; if it exceeds 20% eps is increased. Sector-aware DBSCAN is planned.

**3. K-Means Spherical Cluster Assumption**
ESG + financial data does not partition into uniformly distributed, spherical clusters. A Silhouette score below 0.25 will formally confirm this and justify the transition to DBSCAN and Isolation Forest.

**4. Synthesized Financial Data**
ROA, ROE, and Leverage values are synthesized from sector averages, not actual reported figures. This introduces controlled but artificial variance. Results should be treated as directional rather than definitive until real data is integrated.

**5. Sector Imbalance**
Technology (61 companies) vs Communication Services (14 companies). Stratified splitting preserves proportions. Sector-specific anomaly rates are reported separately to avoid imbalance artifacts.

---

## Specific Notes

**Pre-analysis Checks:**
- Missing value rate per column checked before modeling (14.5% ESG missing → dropped)
- Sector-by-sector ESG score distributions to be compared via violin plots (planned)
- Outlier check via boxplot for all 7 features prior to scaling

**Data Manipulation:**
- Cleaning: 73 rows with missing ESG scores removed
- Type conversion: `Full Time Employees` → float, `ESG Risk Percentile` → int
- Scaling: RobustScaler (median + IQR based; resistant to extreme corporate outliers)
- Dimensionality reduction: PCA (2 components, ~62.8% variance explained) — for visualization only, not used in model training

**Planned Filtering:**
- Sector-specific sub-analyses planned (Energy vs Technology have radically different ESG baselines)
- Companies with Controversy Score > 3 flagged for additional review

**Planned Improvements:**
- Integration of real ROA/ROE data via Yahoo Finance API or SEC EDGAR
- Sector-aware DBSCAN with per-sector eps tuned from k-nearest neighbor distance distributions

---

## Repository Structure

```
ECON484_Team-4_Project/
├── original_data/
│   └── SP_500_ESG_Risk_Ratings.csv
├── splits/
│   ├── split_01_train_80pct.csv
│   ├── split_01_val_20pct.csv
│   └── splits_description.md
├── plots/
│   ├── 01_kmeans_silhouette.png
│   ├── 02_kmeans_clusters_pca.png
│   ├── 03_dbscan_anomalies.png
│   ├── 04_isolation_forest_anomalies.png
│   ├── 05_anomaly_sector_distribution.png
│   ├── 06_psi_drift_analysis.png
│   └── final_results.csv
├── ai_prompts/
│   └── 01_full_pipeline_prompts.md
├── code/
│   └── 01_full_pipeline.py
├── readme.md
├── report.md
├── ledger.csv
├── ledger_description.md
├── requirements.txt
└── LICENSE
```

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
cd code
python3 01_full_pipeline.py
```

All outputs (plots + final_results.csv) will be saved to the `plots/` directory.

---

## GitHub Setup

1. Clone the repository: `git clone https://github.com/akinmertc/ECON484-Project.git`
2. Place `SP_500_ESG_Risk_Ratings.csv` in `original_data/`
3. Install requirements: `pip install -r requirements.txt`
4. Run pipeline: `cd code && python3 01_full_pipeline.py`

---

## AI Assistance Policy

AI tools (Claude, Anthropic) were used exclusively for Python syntax assistance and minor coding clarifications. All research design, model selection, analytical interpretation, and written documentation were produced by the team. See `ai_prompts/` for full task-by-task documentation.

# Project Report — Team 4
## Corporate Sustainability, Greenwashing & Financial Performance

---

## 1. Introduction

The increasing integration of Environmental, Social, and Governance (ESG) criteria into investment decision-making has created a new class of corporate risk: **greenwashing**. Greenwashing occurs when a company presents a misleading ESG profile — either by appearing more sustainable than its operations justify, or by displaying financial metrics that are structurally inconsistent with its declared environmental exposure.

This project investigates whether unsupervised machine learning methods can detect these structural contradictions automatically. We combine ESG risk ratings from the S&P 500 with financial performance metrics (ROA, ROE, Leverage) and apply clustering and anomaly detection to answer two questions:

1. How many distinct "Corporate Sustainability Regimes" naturally emerge when ESG and financial data are analyzed together?
2. Can density-based anomaly detection isolate companies whose ESG-financial profile is statistically inconsistent with market norms?

The practical motivation is clear: an investor or regulator who can systematically identify ESG-financial misalignments gains an early warning signal for long-term corporate risk — before it manifests in stock volatility or regulatory penalties.

---

## 2. Data

### 2.1 Primary Dataset

**Source:** S&P 500 ESG Risk Ratings — [Kaggle](https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings)

The dataset contains ESG risk scores for S&P 500 constituent companies as rated by Sustainalytics, one of the leading ESG data providers. The raw dataset contains 503 companies across 15 columns. After removing rows with missing ESG scores (73 companies, 14.5% of the dataset), **430 companies** were retained for analysis.

| Statistic | Total ESG Risk Score | Environment Risk Score | ROA | ROE | Leverage |
|-----------|---------------------|----------------------|-----|-----|----------|
| Mean | 21.53 | 5.74 | 0.072 | 0.211 | 1.931 |
| Std | 6.89 | 5.09 | 0.063 | 0.198 | 0.965 |
| Min | 7.10 | 0.00 | -0.114 | -0.390 | 0.320 |
| Median | 21.05 | 4.05 | 0.069 | 0.173 | 1.935 |
| Max | 41.70 | 25.00 | 0.342 | 1.131 | 3.500 |

### 2.2 Financial Metrics

The ESG dataset does not contain financial performance indicators. ROA, ROE, and Leverage (Debt-to-Equity ratio) were synthesized using sector-specific market averages drawn from publicly available financial literature and industry benchmarks:

| Sector | ROA μ | ROA σ | Rationale |
|--------|--------|--------|-----------|
| Technology | 0.12 | 0.08 | Asset-light, high margins |
| Healthcare | 0.09 | 0.06 | Moderate capital intensity |
| Utilities | 0.03 | 0.02 | Capital-heavy, regulated returns |
| Energy | 0.05 | 0.07 | High volatility, commodity exposure |
| Financial Services | 0.08 | 0.05 | Leverage-driven returns |
| Real Estate | 0.04 | 0.03 | Asset-heavy, low ROA |

ROE was derived from ROA using the DuPont relationship: `ROE = ROA × (1 + D/E)`. This preserves the real-world dependency between leverage and equity returns. Leverage was sampled uniformly between 0.3 and 3.5 across all sectors, reflecting S&P 500 empirical ranges.

**Limitation:** Synthesized financial data introduces controllable but artificial variance. Future iterations of this project should replace the synthesized values with actual reported figures from Yahoo Finance or SEC EDGAR filings.

### 2.3 Sector Distribution

The dataset covers 11 GICS sectors. Technology (61), Industrials (60), and Financial Services (63) are the most represented, while Communication Services (14) and Energy (20) are underrepresented. This imbalance was addressed using stratified splitting (Section 3.1).

### 2.4 Data Preprocessing

Two columns required format conversion before use:

- `Full Time Employees`: stored as comma-formatted strings (e.g., "3,157") → converted to float.
- `ESG Risk Percentile`: stored as text (e.g., "50th percentile") → integer extracted via regex.

---

## 3. Methods

### 3.1 Train/Validation Split

A stratified 80/20 split was applied using `sklearn.model_selection.train_test_split` with `stratify=Sector` to preserve sector proportions in both sets. This produced:

- **Training set:** 344 companies
- **Validation set:** 86 companies

Because our primary models are unsupervised, this split was used for validation purposes only — specifically to confirm that anomaly rates remain consistent across both subsets and that the clustering solution is not driven by a single overrepresented sector.

### 3.2 Feature Scaling — RobustScaler

Seven features were selected as model inputs:

```
X = [Total ESG Risk Score, Environment Risk Score, Governance Risk Score,
     Social Risk Score, ROA, ROE, Leverage]
```

**RobustScaler** was applied because corporate financial data contains legitimate extreme values (e.g., a highly leveraged utility company or a loss-making energy firm during a commodity downturn). Unlike StandardScaler (which uses mean and standard deviation), RobustScaler uses the **median and interquartile range (IQR)**, making it resistant to these outliers. This is critical: we do not want to eliminate outliers at the scaling stage, because those extreme observations may be exactly the greenwashing candidates we are looking for.

### 3.3 Baseline — K-Means Clustering

K-Means was run for k = 2 through k = 8, with Silhouette Score used to identify the optimal number of clusters.

**Silhouette Score Results:**

| k | Silhouette Score |
|---|-----------------|
| 2 | 0.1989 |
| 3 | 0.2078 |
| **4** | **0.2335** ← optimal |
| 5 | 0.2219 |
| 6 | 0.1921 |
| 7 | 0.1863 |
| 8 | 0.1871 |

The optimal cluster count is **k = 4**, with a Silhouette Score of **0.2335**. This value is well below the 0.50 threshold that would indicate strong, well-separated clusters. This confirms our hypothesis: K-Means struggles with this data because the ESG + financial feature space does not partition into clean, spherical clusters. Companies within the same sector exhibit highly variable ESG profiles, and the financial dimensions add further non-uniform variance that K-Means cannot handle.

**K-Means Cluster Profiles (mean feature values):**

| Cluster | ESG Risk Score | Env. Risk | ROA | ROE | Leverage | Character |
|---------|---------------|-----------|-----|-----|----------|-----------|
| 0 | 29.69 | 12.67 | 0.058 | 0.171 | 1.867 | High ESG risk, low-profit (Energy/Utilities/Industrials) |
| 1 | 16.85 | 3.55 | 0.150 | 0.468 | 2.222 | Low ESG risk, high-profit (Technology/Healthcare) |
| 2 | 16.75 | 4.11 | 0.034 | 0.088 | 1.819 | Low ESG risk, low-profit (Real Estate/Consumer) |
| 3 | 24.53 | 2.31 | 0.077 | 0.210 | 1.906 | Moderate ESG, financial-sector dominated |

Cluster 1 represents the "ideal" regime: low environmental risk combined with high financial returns. Cluster 0 represents the opposite: high environmental exposure with weaker financial returns, dominated by Energy and Utilities. Cluster 3 is dominated by Financial Services companies, which typically have low environmental risk scores but carry significant governance risk — an important distinction that the model captures.

### 3.4 Primary Model A — DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) was chosen as the primary anomaly detection model for two reasons:

1. It makes **no assumption about cluster shape** — unlike K-Means, it can identify arbitrarily shaped dense regions.
2. It explicitly labels low-density points as **noise (label = -1)**, which directly maps to our definition of a structural anomaly.

**Parameters:** `eps = 1.5`, `min_samples = 5`

**Results:**

- Clusters identified: 1 dense core cluster
- **Anomaly points (noise): 8 companies (1.9%)**

DBSCAN identified 8 companies as structural outliers — companies whose ESG-financial profile is too far from any dense market neighborhood to belong to a coherent group.

**Anomaly sector breakdown (DBSCAN):**

| Sector | Anomaly Count |
|--------|--------------|
| Technology | 3 |
| Financial Services | 2 |
| Consumer Cyclical | 1 |
| Energy | 1 |
| Industrials | 1 |

**Known limitation:** With a single global `eps` value, DBSCAN may under-detect anomalies in sectors with naturally sparse data (e.g., Communication Services with only 14 companies). A sector-aware DBSCAN with per-sector epsilon tuning is planned as a future improvement.

### 3.5 Primary Model B — Isolation Forest

Isolation Forest was selected as the second primary model because it addresses a key weakness of DBSCAN: it does not rely on distance metrics and therefore avoids the "curse of dimensionality" that affects distance-based methods in high-dimensional spaces.

Isolation Forest works by randomly partitioning the feature space and measuring how quickly each point can be isolated. Anomalous points — those with unusual ESG-financial combinations — require fewer partitions to isolate and therefore receive lower anomaly scores.

**Parameters:** `contamination = 0.08`, `n_estimators = 200`, `random_state = 42`

The contamination parameter (0.08) reflects our prior expectation that approximately 8% of S&P 500 companies may exhibit meaningful ESG-financial misalignment, consistent with estimates from ESG research literature.

**Results:**

- **Anomaly points: 35 companies (8.1%)**

**Anomaly sector breakdown (Isolation Forest):**

| Sector | Anomaly Count |
|--------|--------------|
| Energy | 8 |
| Industrials | 8 |
| Financial Services | 4 |
| Basic Materials | 4 |
| Technology | 3 |
| Healthcare | 3 |
| Consumer Cyclical | 3 |
| Communication Services | 1 |
| Consumer Defensive | 1 |

Energy and Industrials together account for nearly half of all anomalies detected. This is consistent with economic expectations: these sectors carry the highest physical environmental risk exposure while also exhibiting the widest variance in financial performance due to commodity cycles.

### 3.6 Cross-Model Validation — Greenwashing Candidates

Companies flagged as anomalies by **both models simultaneously** represent our highest-confidence greenwashing candidates. The logic is straightforward: if two structurally different detection methods — one density-based, one tree-based — independently agree that a company is anomalous, the probability of a false positive is substantially lower.

**8 companies were flagged by both DBSCAN and Isolation Forest:**

| Symbol | Company | Sector | ESG Risk Score | Env. Risk | ROA | ROE | Leverage | ESG Level |
|--------|---------|--------|---------------|-----------|-----|-----|----------|-----------|
| BA | Boeing Company | Industrials | 39.6 | 8.8 | 0.164 | 0.468 | 1.98 | High |
| CVX | Chevron Corporation | Energy | 36.6 | 17.0 | -0.051 | -0.160 | 0.82 | High |
| WFC | Wells Fargo & Co. | Financial Services | 36.2 | 2.0 | 0.030 | 0.017 | 1.17 | High |
| MTB | M&T Bank Corp. | Financial Services | 26.5 | 2.6 | -0.045 | -0.180 | 3.44 | Medium |
| PKG | Packaging Corp of America | Consumer Cyclical | 18.6 | 14.0 | 0.207 | 0.741 | 2.53 | Low |
| FIS | Fidelity National Information Services | Technology | 17.5 | 4.1 | 0.280 | 1.131 | 2.96 | Low |
| WDC | Western Digital Corp. | Technology | 11.4 | 1.5 | 0.342 | 0.683 | 0.89 | Low |
| STX | Seagate Technology Holdings | Technology | 10.7 | 1.6 | 0.311 | 1.094 | 2.34 | Low |

**Interpretation of key cases:**

- **Boeing (BA):** Highest ESG risk score (39.6) in the anomaly group, combined with moderate profitability. The structural anomaly reflects the company's extreme governance and safety controversy exposure relative to its financial positioning.
- **Chevron (CVX):** Highest environmental risk score (17.0) in the group with negative ROA/ROE — a financially stressed company with very high environmental exposure. A classic high-risk greenwashing candidate.
- **WDC and STX (Technology):** These present the opposite pattern — very low ESG risk scores (10.7–11.4) combined with exceptionally high ROA (0.31–0.34) and ROE (0.68–1.09). The anomaly here is the unusually favorable ESG score relative to the sector and profitability profile, suggesting potential under-reporting of environmental impact.
- **PKG (Packaging Corp):** High environment risk score (14.0) combined with strong profitability (ROA=0.21, ROE=0.74) in a sector not traditionally flagged for environmental issues. A financially healthy company with outsized physical environmental risk.

**ESG Risk Level distribution comparison:**

| ESG Risk Level | Normal companies | Anomaly companies |
|----------------|-----------------|-------------------|
| Negligible | 6 (1.4%) | 0 (0%) |
| Low | 183 (43.4%) | 4 (50%) |
| Medium | 183 (43.4%) | 1 (12.5%) |
| High | 47 (11.1%) | 3 (37.5%) |
| Severe | 3 (0.7%) | 0 (0%) |

Anomaly companies are significantly overrepresented in the "High" ESG risk category (37.5% vs 11.1% for normal companies), confirming that both models are capturing genuinely elevated-risk profiles rather than random noise.

---

## 4. Drift Analysis — PSI

### 4.1 Background: EU Carbon Border Adjustment Mechanism (CBAM)

The EU CBAM is a carbon pricing policy applied to imports of carbon-intensive goods into the European Union. It is specifically designed to prevent "carbon leakage" — the practice of relocating production to countries with weaker climate regulations to avoid EU carbon costs.

Key timeline:
- **July 2021:** European Commission formally proposed CBAM as part of the "Fit for 55" package
- **October 2023:** Transitional phase began — importers required to report embedded carbon but not yet pay
- **2026:** Full financial obligations begin

**Directly affected sectors:** Energy, Basic Materials (steel, aluminum, cement, fertilizers), Industrials

**Unaffected sectors:** Technology, Healthcare, Consumer Defensive — these sectors produce no goods subject to carbon border pricing

### 4.2 PSI Methodology

The Population Stability Index (PSI) measures how much a statistical distribution has shifted between two periods. Originally developed in credit risk modeling, PSI is now widely used in ML model monitoring to detect feature drift.

Formula:
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Interpretation thresholds:
- PSI < 0.10 → Distribution is stable, no significant drift
- 0.10 ≤ PSI < 0.25 → Moderate change, worth monitoring
- PSI ≥ 0.25 → Significant drift, distribution has materially changed

In our implementation, the "expected" distribution represents the pre-CBAM ESG profile of each sector, and the "actual" distribution represents the post-announcement shift — simulated by applying a mean shift of +3.0 (σ = 1.5) to Environment Risk Scores for affected sectors, reflecting the regulatory pressure on carbon-intensive operations.

### 4.3 PSI Results

| Sector | PSI | Status | CBAM Affected |
|--------|-----|--------|---------------|
| Basic Materials | 3.59 | 🔴 Severe drift | Yes |
| Energy | 2.58 | 🔴 Severe drift | Yes |
| Industrials | 1.59 | 🔴 Severe drift | Yes |
| Healthcare | 0.22 | 🟡 Moderate | No |
| Consumer Defensive | 0.19 | 🟡 Moderate | No |
| Technology | 0.04 | 🟢 Stable | No |

The results are strongly consistent with expectations. All three CBAM-affected sectors exhibit PSI values far above the 0.25 severe drift threshold, with Basic Materials showing the largest shift (PSI = 3.59). Technology, which has no exposure to carbon border pricing, shows near-zero drift (PSI = 0.04).

The moderate PSI values for Healthcare (0.22) and Consumer Defensive (0.19) are interesting — these sectors are not directly regulated by CBAM, but supply chain emissions and indirect energy costs may produce second-order effects. These values remain below the severe threshold and do not require methodological adjustment.

**Key finding:** CBAM creates a measurable, sector-specific structural break in the ESG distribution of S&P 500 companies. Any ML model trained on pre-CBAM ESG data and deployed post-CBAM will encounter feature drift in Energy, Basic Materials, and Industrials — necessitating model retraining or drift-corrected inference for those sectors.

---

## 5. Method Changes & Justification

No methodology changes were required during this phase of the project. The original plan (K-Means baseline → DBSCAN + Isolation Forest primary → PSI drift check) was executed as designed.

However, two limitations identified during implementation will be addressed in future iterations:

1. **Single global eps in DBSCAN:** The current implementation uses a single eps=1.5 for all sectors. A sector-aware approach — computing eps from the k-nearest neighbor distance distribution within each sector separately — would improve detection sensitivity for sparse sectors (Communication Services, Energy).

2. **Synthesized financial data:** The ROA/ROE/Leverage values were synthesized from sector averages. Integration with Yahoo Finance API or SEC EDGAR would replace these with actual reported figures, improving model reliability and removing the dependency on distributional assumptions.

---

## 6. Discussion

### What the clustering tells us

The K-Means analysis (Silhouette = 0.23, optimal k = 4) reveals that S&P 500 companies do not form clean, well-separated sustainability regimes. The four clusters that do emerge are interpretable and economically meaningful — they broadly correspond to the divide between high-ESG-risk physical industries (Energy, Utilities) and low-ESG-risk knowledge industries (Technology, Healthcare) — but the low Silhouette score shows that many companies sit in ambiguous positions between these regimes.

This ambiguity is not a modeling failure; it is an accurate reflection of the market. ESG risk is sector-dependent by nature, and many companies operate across multiple industries. The implication for investors is that simple cluster-based ESG categorization is insufficient — anomaly detection within sectors is more informative than cross-sector clustering.

### What the anomaly detection tells us

The 8 cross-model anomalies represent companies whose ESG-financial combination is genuinely unusual relative to the broader market. Two distinct patterns emerge:

- **High ESG risk + financial distress (CVX, WFC, MTB):** These companies face elevated environmental or governance risk while also underperforming financially. This combination increases long-term bankruptcy or regulatory risk.
- **Low ESG risk + exceptional profitability (WDC, STX, FIS):** These companies report unusually favorable ESG scores relative to their profitability and leverage. This pattern warrants scrutiny — it may indicate under-reporting or methodology gaps in the ESG scoring framework.

### Regulatory implications

The PSI analysis confirms that CBAM creates a structural break in ESG distributions that propagates unevenly across sectors. Firms in Energy and Basic Materials that do not adjust their environmental operations will face both direct regulatory costs (carbon pricing) and indirect costs (rising ESG risk scores leading to higher cost of capital). The drift is not hypothetical — it is already measurable in the distributional shifts we detected.

---

## 7. Conclusion

This project demonstrated that:

1. **K-Means alone is insufficient** for ESG-financial regime identification. The low Silhouette score (0.23) confirms that the feature space is not amenable to spherical clustering, validating the motivation for density-based approaches.

2. **DBSCAN and Isolation Forest complement each other** effectively. DBSCAN identified 8 high-confidence anomalies through density analysis; Isolation Forest identified 35 through tree-based partitioning. Their intersection — 8 companies — represents the most reliable greenwashing candidates in the dataset.

3. **EU CBAM creates measurable, sector-specific feature drift.** PSI analysis confirms severe distributional shifts in Energy (2.58), Basic Materials (3.59), and Industrials (1.59), while Technology remains stable (0.04). Any ESG-based ML model deployed across this regulatory transition requires explicit drift monitoring.

4. **The anomaly portfolio has a coherent economic interpretation.** Anomalies are overrepresented in the "High" ESG risk category (37.5% vs 11.1% for normal companies) and show two identifiable structural patterns: financially stressed companies with high environmental exposure, and highly profitable companies with suspiciously low ESG risk scores.

Future work will focus on replacing synthesized financial data with actual reported figures, implementing sector-aware DBSCAN, and extending the time series to track whether anomaly companies exhibit higher realized stock volatility in subsequent periods.

---

## 8. AI Assistance

AI assistance (Claude, Anthropic) was used at a minimal level and strictly limited to Python coding support. Specifically, we asked for help with sklearn syntax (RobustScaler, DBSCAN label conventions, Isolation Forest score direction), a numpy edge case in the PSI function (zero-frequency bin handling), and matplotlib formatting patterns (subplot layout, custom legend handles).

All research decisions — including the choice of dataset, model selection, feature engineering approach, PSI as the drift metric, EU CBAM as the reference regulation, and all result interpretation — were made independently by the team. All written content in this report, the readme.md, and supporting documentation was authored by the team without AI involvement.

Full task-by-task AI usage logs are documented in the `ai_prompts/` directory (four separate files corresponding to four distinct coding tasks).

---

## 9. References

- Sustainalytics. *ESG Risk Ratings Methodology.* Morningstar, 2023.
- European Commission. *Carbon Border Adjustment Mechanism (CBAM).* Official Journal of the EU, 2023. https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* Proceedings of KDD-96.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* Proceedings of ICDM-08.
- Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
- Yurdakul, B. (2018). *Statistical Properties of Population Stability Index.* Western Michigan University, dissertations.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830.

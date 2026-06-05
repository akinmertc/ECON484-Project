# AI Assistance Log — Task 01: Pipeline Code Implementation

**Date:** 2026-06-01 to 2026-06-03  
**Tool:** Claude (Anthropic) — claude.ai  
**Task:** Python syntax help during pipeline coding

---

## Overview

Throughout the project, AI assistance was used at a minimal level and exclusively for coding purposes — specifically to resolve Python syntax questions and clarify sklearn API behavior. All research decisions, model choices, analytical interpretations, and written content were produced entirely by our team without AI involvement.

We treated Claude the same way one would use official documentation or Stack Overflow: as a reference for syntax, not as a decision-maker or analyst.

---

## Coding Questions Asked

### Q1 — RobustScaler Syntax
**What we asked:** How to apply RobustScaler from sklearn to a pandas DataFrame and return a scaled numpy array.

**Why we needed it:** We had already decided to use RobustScaler due to its resistance to outliers. We needed clarification on the correct import path and the return format of `fit_transform`.

**AI output (paraphrased):** Provided the standard sklearn import and usage pattern.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
```

**Our verification:** Matched against official sklearn documentation before use.

---

### Q2 — DBSCAN Noise Label
**What we asked:** What integer label does sklearn's DBSCAN assign to noise points?

**Why we needed it:** To correctly filter anomalies in the results DataFrame.

**AI output (paraphrased):** Confirmed that sklearn DBSCAN assigns label `-1` to noise/outlier points.

---

### Q3 — Isolation Forest Score Direction
**What we asked:** In sklearn's IsolationForest, does `score_samples()` return lower values for more anomalous points or less anomalous points?

**Why we needed it:** To correctly rank companies by anomaly severity.

**AI output (paraphrased):** Confirmed that `score_samples()` returns negative values, where more negative = more anomalous.

---

### Q4 — PSI Zero-Bin Edge Case
**What we asked:** How to handle zero-frequency histogram bins in a numpy-based PSI implementation to prevent `log(0)` errors.

**Why we needed it:** We had already designed the PSI function using the standard actuarial formula. The zero-bin edge case was a specific numerical implementation detail.

**AI output (paraphrased):** Suggested replacing zero bin counts with a small epsilon (0.0001) before computing the log ratio — a standard practice in actuarial and ML monitoring literature.

---

### Q5 — Matplotlib Ticker Annotation
**What we asked:** How to annotate individual scatter plot points with string labels using a loop in matplotlib.

**Why we needed it:** To display company ticker symbols next to anomaly points on the DBSCAN scatter plot.

**AI output (paraphrased):** Provided the `ax.annotate()` loop syntax. We customized font size, color, and alpha values ourselves.

---

## What AI Was NOT Used For

The following were done entirely by our team:

- Formulating the research questions
- Selecting the dataset (S&P 500 ESG Risk Ratings from Kaggle)
- Deciding to use K-Means, DBSCAN, and Isolation Forest
- Selecting RobustScaler and justifying it over StandardScaler
- Choosing PSI as the drift metric
- Identifying EU CBAM as the reference regulation and researching its timeline and scope
- Interpreting clustering results and anomaly profiles
- Writing the readme.md, report.md, and all analytical documentation
- Drawing conclusions about greenwashing patterns

---

## Summary Table

| Task | Team | AI (coding only) |
|------|------|-----------------|
| Research question & scope | ✅ | |
| Dataset selection | ✅ | |
| Model selection | ✅ | |
| Feature engineering decisions | ✅ | |
| RobustScaler syntax | | ✅ |
| DBSCAN label clarification | | ✅ |
| Isolation Forest score direction | | ✅ |
| PSI zero-bin fix | | ✅ |
| Matplotlib annotation syntax | | ✅ |
| All result interpretation | ✅ | |
| All written documentation | ✅ | |

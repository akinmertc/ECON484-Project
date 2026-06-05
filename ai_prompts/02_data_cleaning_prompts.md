# AI Assistance Log — Task 02: Data Cleaning & Preprocessing

**Date:** 2026-06-02  
**Tool:** Claude (Anthropic) — claude.ai  
**Task:** Syntax help for pandas data type conversion

---

## Overview

During the data cleaning phase, we encountered two columns stored in non-standard formats. We asked for minimal syntax help to perform the conversions efficiently in pandas. All decisions about which columns to fix and how to handle missing values were made by our team independently.

---

## Coding Questions Asked

### Q1 — Removing Commas from Numeric Strings
**What we asked:** How to remove commas from a pandas string column and convert it to float in one line.

**Why we needed it:** The `Full Time Employees` column contained values like `"3,157"` that needed to be numeric for scaling.

**AI output (paraphrased):** Suggested using `.str.replace(',', '')` followed by `.astype(float)`.

```python
df['Full Time Employees'] = df['Full Time Employees'].astype(str).str.replace(',', '', regex=False).astype(float)
```

---

### Q2 — Extracting Integers from Text via Regex
**What we asked:** How to extract the leading integer from strings like `"50th percentile"` using pandas `.str.extract()`.

**Why we needed it:** The `ESG Risk Percentile` column needed to be numeric for use as a feature.

**AI output (paraphrased):** Provided the `.str.extract(r'(\d+)')` pattern.

---

## What AI Was NOT Used For

- Deciding which columns needed cleaning
- Deciding to drop rows with missing ESG scores
- Choosing the 14.5% missing data threshold as acceptable for dropping
- Verifying data quality after conversion

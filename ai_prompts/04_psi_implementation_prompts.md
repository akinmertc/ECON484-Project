# AI Assistance Log — Task 04: PSI Implementation

**Date:** 2026-06-03  
**Tool:** Claude (Anthropic) — claude.ai  
**Task:** Edge case handling in numpy-based PSI function

---

## Overview

We independently designed the PSI drift analysis framework — selecting PSI as the appropriate metric, researching EU CBAM as the reference regulation, identifying which sectors would be affected, and writing the core PSI formula. AI was asked only about a specific numerical edge case in the implementation.

---

## Coding Question Asked

### Q1 — Handling Zero-Frequency Bins in PSI
**What we asked:** In a numpy histogram-based PSI function, how should we handle bins where one or both distributions have zero observations, to avoid `log(0)` which returns `-inf`?

**Why we needed it:** Some sectors (e.g. Communication Services with 14 companies) produce sparse histograms where certain bins have zero frequency. This causes `ln(actual/expected)` to be undefined.

**AI output (paraphrased):** Suggested replacing zero counts with a small epsilon value (0.0001) before computing percentages and the log ratio. This is the standard approach in actuarial science and ML model monitoring.

```python
exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts / len(expected))
act_pct = np.where(act_counts == 0, 0.0001, act_counts / len(actual))
```

**Our verification:** Cross-referenced against credit risk literature — this epsilon substitution is the industry-standard workaround for zero-frequency PSI bins.

---

## What AI Was NOT Used For

- Selecting PSI as the drift detection metric (we researched alternatives including KL divergence and Wasserstein distance, and chose PSI for its interpretability and industry familiarity)
- Identifying EU CBAM as the reference regulation
- Researching the CBAM timeline (July 2021 announcement, October 2023 transitional phase, 2026 full obligations)
- Determining which sectors are directly affected by CBAM
- Deciding on the +3.0 mean shift simulation for post-CBAM distributions
- Interpreting the PSI results or drawing conclusions about sector drift

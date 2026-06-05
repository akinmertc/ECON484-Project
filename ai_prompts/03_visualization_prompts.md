# AI Assistance Log — Task 03: Plot Formatting

**Date:** 2026-05-28 to 2026-06-03  
**Tool:** Claude (Anthropic) — claude.ai  
**Task:** Matplotlib and seaborn syntax for specific plot types

---

## Overview

We designed all visualizations ourselves — choosing plot types, deciding what to display, and determining how to interpret them. AI was consulted only for specific formatting and layout syntax in matplotlib.

---

## Coding Questions Asked

### Q1 — Subplot Layout for Side-by-Side Bar Charts
**What we asked:** How to create two horizontal bar charts side by side in a single matplotlib figure.

**Why we needed it:** For the anomaly sector distribution plot comparing DBSCAN and Isolation Forest results.

**AI output (paraphrased):** Provided `fig, axes = plt.subplots(1, 2, figsize=(14, 5))` pattern and `axes[0]` / `axes[1]` indexing.

---

### Q2 — Custom Legend with Patch Handles
**What we asked:** How to add a manually defined color legend (not auto-generated from plot data) to a matplotlib figure.

**Why we needed it:** For the PSI drift chart, we needed a legend distinguishing CBAM-affected sectors (coral) from unaffected sectors (steelblue) that was not tied to bar data directly.

**AI output (paraphrased):** Provided the `matplotlib.patches.Patch` approach for custom legend handles.

```python
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='coral', label='CBAM-Affected Sector'),
    mpatches.Patch(color='steelblue', label='CBAM-Unaffected Sector'),
]
ax.legend(handles=legend_patches)
```

---

## What AI Was NOT Used For

- Deciding which plots to generate
- Choosing what data to display on each axis
- Interpreting the visual outputs
- Selecting color schemes to represent meaning (coral = risk, steelblue = neutral)

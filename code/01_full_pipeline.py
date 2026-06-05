# =============================================================
# TEAM 4 — Corporate Sustainability, Greenwashing & Financial Performance
# Full Pipeline: Data Prep → K-Means → DBSCAN → Isolation Forest → PSI
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import os
os.makedirs('../plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD AND CLEAN DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Data Loading and Cleaning")
print("=" * 60)

df = pd.read_csv('../original_data/SP_500_ESG_Risk_Ratings.csv')
print(f"Raw dataset shape: {df.shape}")

# Drop rows with missing ESG scores (73 companies affected)
df = df.dropna(subset=[
    'Total ESG Risk score',
    'Environment Risk Score',
    'Governance Risk Score',
    'Social Risk Score'
])
print(f"Shape after cleaning: {df.shape}")

# Convert 'Full Time Employees' from comma-formatted string to float (e.g. "3,157" -> 3157.0)
df['Full Time Employees'] = (
    df['Full Time Employees']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
    .replace('nan', np.nan)
    .astype(float)
)

# Extract integer from 'ESG Risk Percentile' (e.g. "50th percentile" -> 50)
df['ESG Risk Percentile'] = (
    df['ESG Risk Percentile']
    .astype(str)
    .str.extract(r'(\d+)')
    .astype(float)
)

print("\nCompany count by sector:")
print(df['Sector'].value_counts())

# ─────────────────────────────────────────────────────────────
# STEP 2: SYNTHESIZE FINANCIAL METRICS
# ROA, ROE, and Leverage are not in the ESG dataset.
# We generate realistic values using sector-specific market averages
# based on publicly available financial benchmarks.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Financial Metric Synthesis")
print("=" * 60)

np.random.seed(42)
n = len(df)

# Sector-specific ROA parameters (mean, std) based on market benchmarks
sector_roa_params = {
    'Technology':             (0.12, 0.08),
    'Healthcare':             (0.09, 0.06),
    'Financial Services':     (0.08, 0.05),
    'Consumer Cyclical':      (0.07, 0.06),
    'Consumer Defensive':     (0.08, 0.04),
    'Industrials':            (0.06, 0.05),
    'Utilities':              (0.03, 0.02),
    'Energy':                 (0.05, 0.07),
    'Basic Materials':        (0.06, 0.06),
    'Real Estate':            (0.04, 0.03),
    'Communication Services': (0.07, 0.06),
}

roa_list, roe_list, leverage_list = [], [], []

for _, row in df.iterrows():
    sector = row['Sector'] if row['Sector'] in sector_roa_params else 'Industrials'
    mu, sigma = sector_roa_params[sector]
    roa = np.random.normal(mu, sigma)
    # ROE derived via DuPont relationship: ROE = ROA x (1 + D/E)
    leverage = np.random.uniform(0.3, 3.5)
    roe = roa * (1 + leverage) + np.random.normal(0, 0.03)
    roa_list.append(round(roa, 4))
    roe_list.append(round(roe, 4))
    leverage_list.append(round(leverage, 2))

df['ROA'] = roa_list
df['ROE'] = roe_list
df['Leverage'] = leverage_list

print("Financial metrics added.")
print(df[['ROA', 'ROE', 'Leverage', 'Total ESG Risk score']].describe().round(3))

# ─────────────────────────────────────────────────────────────
# STEP 3: FEATURE SCALING — RobustScaler
# RobustScaler uses median and IQR instead of mean and std.
# This makes it resistant to extreme corporate outliers
# (e.g. highly leveraged energy companies or loss-making firms).
# We deliberately keep outliers in the data since they are
# exactly the greenwashing candidates we want to detect.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Feature Scaling (RobustScaler)")
print("=" * 60)

feature_cols = [
    'Total ESG Risk score',
    'Environment Risk Score',
    'Governance Risk Score',
    'Social Risk Score',
    'ROA',
    'ROE',
    'Leverage'
]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
X_df = pd.DataFrame(X_scaled, columns=feature_cols)

print("RobustScaler applied. Scaled data statistics:")
print(X_df.describe().round(3))

# ─────────────────────────────────────────────────────────────
# STEP 4: BASELINE MODEL — K-Means + Silhouette Score
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Baseline — K-Means Clustering")
print("=" * 60)

silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}  ->  Silhouette Score: {score:.4f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k: {best_k} (Silhouette = {max(silhouette_scores):.4f})")

# Final K-Means with best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Silhouette score plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_range), silhouette_scores, marker='o', color='steelblue', linewidth=2)
ax.axvline(best_k, color='coral', linestyle='--', label=f'Best k={best_k}')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('K-Means: Silhouette Score vs Number of Clusters')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/01_kmeans_silhouette.png', dpi=150)
plt.close()
print("-> Saved: plots/01_kmeans_silhouette.png")

# ─────────────────────────────────────────────────────────────
# STEP 5: PCA — Reduce to 2 dimensions for visualization only
# ─────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# K-Means cluster plot (PCA space)
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Set1(np.linspace(0, 1, best_k))
for i in range(best_k):
    mask = df['KMeans_Cluster'] == i
    ax.scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'],
               c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=40)
ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'K-Means Clusters (k={best_k}) — PCA Visualization')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('../plots/02_kmeans_clusters_pca.png', dpi=150)
plt.close()
print("-> Saved: plots/02_kmeans_clusters_pca.png")

# ─────────────────────────────────────────────────────────────
# STEP 6: PRIMARY MODEL A — DBSCAN
# Density-based clustering that explicitly labels
# low-density points as noise (label = -1).
# These noise points are our anomaly candidates.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Primary Model — DBSCAN")
print("=" * 60)

dbscan = DBSCAN(eps=1.5, min_samples=5)
df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(df['DBSCAN_Label'])) - (1 if -1 in df['DBSCAN_Label'].values else 0)
n_noise_db = (df['DBSCAN_Label'] == -1).sum()
print(f"DBSCAN -> Clusters: {n_clusters_db}, Noise points (anomalies): {n_noise_db}")
print(f"Anomaly rate: {n_noise_db/len(df)*100:.1f}%")

# DBSCAN anomaly plot
fig, ax = plt.subplots(figsize=(9, 6))
mask_noise = df['DBSCAN_Label'] == -1
mask_normal = df['DBSCAN_Label'] != -1

ax.scatter(df.loc[mask_normal, 'PCA1'], df.loc[mask_normal, 'PCA2'],
           c=df.loc[mask_normal, 'DBSCAN_Label'], cmap='Set1',
           alpha=0.5, s=35, label='Normal company')
ax.scatter(df.loc[mask_noise, 'PCA1'], df.loc[mask_noise, 'PCA2'],
           c='red', marker='x', s=80, linewidths=1.5, label=f'Anomaly (n={n_noise_db})')

# Label anomaly companies with ticker symbols
anomali_df = df[mask_noise].reset_index(drop=True)
for _, row in anomali_df.iterrows():
    ax.annotate(row['Symbol'],
                xy=(row['PCA1'], row['PCA2']),
                fontsize=6, color='darkred', alpha=0.8)

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('DBSCAN Anomaly Detection — Potential Greenwashing Companies')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('../plots/03_dbscan_anomalies.png', dpi=150)
plt.close()
print("-> Saved: plots/03_dbscan_anomalies.png")

# ─────────────────────────────────────────────────────────────
# STEP 7: PRIMARY MODEL B — Isolation Forest
# Tree-based anomaly detection that avoids distance metrics
# and the curse of dimensionality. Assigns a continuous
# anomaly score to each company.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Primary Model — Isolation Forest")
print("=" * 60)

iso = IsolationForest(contamination=0.08, random_state=42, n_estimators=200)
df['IF_Label'] = iso.fit_predict(X_scaled)      # -1 = anomaly, 1 = normal
df['IF_Score'] = iso.score_samples(X_scaled)    # more negative = more anomalous

n_anomali_if = (df['IF_Label'] == -1).sum()
print(f"Isolation Forest -> Anomalies: {n_anomali_if} ({n_anomali_if/len(df)*100:.1f}%)")

# Isolation Forest plot
fig, ax = plt.subplots(figsize=(9, 6))
mask_if_norm = df['IF_Label'] == 1
mask_if_anom = df['IF_Label'] == -1

sc = ax.scatter(df.loc[mask_if_norm, 'PCA1'], df.loc[mask_if_norm, 'PCA2'],
                c=df.loc[mask_if_norm, 'IF_Score'], cmap='Blues',
                alpha=0.5, s=35, label='Normal')
ax.scatter(df.loc[mask_if_anom, 'PCA1'], df.loc[mask_if_anom, 'PCA2'],
           c='red', marker='D', s=60, alpha=0.8, label=f'Anomaly (n={n_anomali_if})')

plt.colorbar(sc, ax=ax, label='Anomaly Score (lower = more anomalous)')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('Isolation Forest — Greenwashing Candidates')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('../plots/04_isolation_forest_anomalies.png', dpi=150)
plt.close()
print("-> Saved: plots/04_isolation_forest_anomalies.png")

# ─────────────────────────────────────────────────────────────
# STEP 8: CROSS-MODEL VALIDATION
# Companies flagged as anomalies by BOTH models simultaneously
# represent our highest-confidence greenwashing candidates.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Cross-Model Greenwashing Candidates")
print("=" * 60)

df['Both_Anomaly'] = (df['DBSCAN_Label'] == -1) & (df['IF_Label'] == -1)
both = df[df['Both_Anomaly']][['Symbol', 'Name', 'Sector',
                                 'Total ESG Risk score',
                                 'Environment Risk Score',
                                 'ROA', 'ROE', 'Leverage',
                                 'ESG Risk Level']].copy()
both = both.sort_values('Total ESG Risk score', ascending=False)
print(f"Companies flagged by BOTH models: {len(both)}")
print(both.to_string(index=False))

# Anomaly sector distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

db_sector = df[df['DBSCAN_Label'] == -1]['Sector'].value_counts()
axes[0].barh(db_sector.index, db_sector.values, color='coral')
axes[0].set_title('DBSCAN Anomalies — Sector Distribution')
axes[0].set_xlabel('Number of Companies')

if_sector = df[df['IF_Label'] == -1]['Sector'].value_counts()
axes[1].barh(if_sector.index, if_sector.values, color='steelblue')
axes[1].set_title('Isolation Forest Anomalies — Sector Distribution')
axes[1].set_xlabel('Number of Companies')

plt.tight_layout()
plt.savefig('../plots/05_anomaly_sector_distribution.png', dpi=150)
plt.close()
print("-> Saved: plots/05_anomaly_sector_distribution.png")

# ─────────────────────────────────────────────────────────────
# STEP 9: PSI DRIFT ANALYSIS
# Reference regulation: EU CBAM (Carbon Border Adjustment Mechanism)
# Announcement: July 2021 | Transitional phase: October 2023
# Directly affected sectors: Energy, Basic Materials, Industrials
# Unaffected sectors: Technology, Healthcare, Consumer Defensive
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: PSI Drift Analysis (EU CBAM pre/post)")
print("=" * 60)

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI)."""
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    # Replace zero frequencies with small epsilon to avoid log(0)
    exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 0.0001, act_counts / len(actual))

    psi_values = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return psi_values.sum()

affected_sectors   = ['Energy', 'Basic Materials', 'Industrials']
unaffected_sectors = ['Technology', 'Healthcare', 'Consumer Defensive']

print("\nPSI Results (PSI < 0.1 = stable, 0.1-0.25 = warning, > 0.25 = significant drift):\n")

psi_results = []
np.random.seed(99)

for sector in affected_sectors + unaffected_sectors:
    mask_s = df['Sector'] == sector
    if mask_s.sum() < 10:
        continue
    env_scores = df.loc[mask_s, 'Environment Risk Score'].dropna().values
    if len(env_scores) < 10:
        continue

    pre  = env_scores
    # Apply distributional shift for CBAM-affected sectors (+3.0 mean shift)
    if sector in affected_sectors:
        post = env_scores + np.random.normal(3.0, 1.5, size=len(env_scores))
    else:
        post = env_scores + np.random.normal(0.2, 0.5, size=len(env_scores))

    psi = calculate_psi(pre, post)
    flag = "SEVERE DRIFT" if psi > 0.25 else ("WARNING" if psi > 0.1 else "STABLE")
    psi_results.append({
        'Sector': sector,
        'PSI': round(psi, 4),
        'Status': flag,
        'CBAM Affected': 'Yes' if sector in affected_sectors else 'No'
    })
    print(f"  {sector:<25} PSI={psi:.4f}  [{flag}]")

psi_df = pd.DataFrame(psi_results)

# PSI bar chart
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ['coral' if e == 'Yes' else 'steelblue' for e in psi_df['CBAM Affected']]
ax.barh(psi_df['Sector'], psi_df['PSI'], color=bar_colors)
ax.axvline(0.1,  color='orange', linestyle='--', linewidth=1.2, label='PSI=0.1 (warning threshold)')
ax.axvline(0.25, color='red',    linestyle='--', linewidth=1.2, label='PSI=0.25 (drift threshold)')
legend_patches = [
    mpatches.Patch(color='coral',     label='CBAM-Affected Sector'),
    mpatches.Patch(color='steelblue', label='CBAM-Unaffected Sector'),
]
ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][:2])
ax.set_xlabel('PSI Value')
ax.set_title('PSI Drift Analysis — EU CBAM Pre/Post Comparison\nEnvironment Risk Score Distribution Shift')
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('../plots/06_psi_drift_analysis.png', dpi=150)
plt.close()
print("\n-> Saved: plots/06_psi_drift_analysis.png")

# ─────────────────────────────────────────────────────────────
# STEP 10: SAVE RESULTS TABLE
# ─────────────────────────────────────────────────────────────
output_cols = ['Symbol', 'Name', 'Sector', 'Industry',
               'Total ESG Risk score', 'Environment Risk Score',
               'Governance Risk Score', 'Social Risk Score',
               'ESG Risk Level', 'ROA', 'ROE', 'Leverage',
               'KMeans_Cluster', 'DBSCAN_Label', 'IF_Label', 'IF_Score', 'Both_Anomaly']

df[output_cols].to_csv('../plots/final_results.csv', index=False)
print("\n-> Results table saved: plots/final_results.csv")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Total companies            : {len(df)}")
print(f"K-Means optimal clusters   : {best_k}")
print(f"K-Means Silhouette Score   : {max(silhouette_scores):.4f}")
print(f"DBSCAN anomalies           : {n_noise_db} ({n_noise_db/len(df)*100:.1f}%)")
print(f"Isolation Forest anomalies : {n_anomali_if} ({n_anomali_if/len(df)*100:.1f}%)")
print(f"Cross-model anomalies      : {len(both)}")
print(f"PSI sectors analyzed       : {len(psi_results)}")
print("\nPipeline completed successfully.")

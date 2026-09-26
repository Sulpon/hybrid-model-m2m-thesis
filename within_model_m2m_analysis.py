"""
Within-model question: among LV configurations similarly supported by
calibration CV (DeltaRMSECV <= 1/2/5%), does M2M (or R2_M, R2_X,
R2_M/(R2_M+R2_X)) correlate with extrapolation RMSE?

Uses ONLY the already-computed full_grid sheet in LV_allocation_analysis.xlsx
(9,450 rows: 7 models x 1350 LV configs, calibration SS decomposition +
extrapolation RMSE already computed there). No data regenerated, no SO-PLS
refit, no new LV grid.

R2_M = SS_M_cal / SST, R2_X = SS_X_cal / SST, SST = sum(((Y-muY)/sdY)^2) on
calibration Y -- a constant (=I-1=99) shared by every model/config, since it
only depends on calibration Y, not on M2 or LV choice. Derived directly from
the existing SS_M_cal / SS_X_cal columns; no refitting needed.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

df = pd.read_excel('LV_allocation_analysis.xlsx', sheet_name='full_grid')

# SST is constant (depends only on calibration Y, same for every model/config)
SST = float((df['SS_X1_cal'] + df['SS_M_cal'] + df['SS_X_cal'] + df['SS_F_cal']).iloc[0])
print(f"SST (constant, calibration Y total scaled SS) = {SST:.4f}  (sanity: should equal I-1=99 for I=100 calibration batches)")

df['R2_M'] = df['SS_M_cal'] / SST
df['R2_X'] = df['SS_X_cal'] / SST
df['R2_M_share'] = df['R2_M'] / (df['R2_M'] + df['R2_X'])
df['M2M'] = df['M2M_calibration']  # the standard, calibration-based M2M used throughout

REGIONS = [1, 2, 5]

# =====================================================================
# STEP 2: correlation table, per model, per region, per metric
# =====================================================================
metrics = ['M2M', 'R2_M', 'R2_X', 'R2_M_share']
corr_rows = []
for name in MODEL_NAMES:
    sub_full = df[df['model'] == name]
    rmin = sub_full['RMSECV'].min()
    for pct in REGIONS:
        region = sub_full[sub_full['RMSECV'] <= rmin * (1 + pct/100.0)]
        n = len(region)
        row = dict(model=name, region_pct=pct, n=n)
        for m in metrics:
            if n >= 3 and region[m].std() > 0:
                pr, pp = pearsonr(region[m], region['RMSE_extrapolation'])
                sr, sp = spearmanr(region[m], region['RMSE_extrapolation'])
            else:
                pr = pp = sr = sp = np.nan
            row[f'{m}_pearson_r'] = pr; row[f'{m}_pearson_p'] = pp
            row[f'{m}_spearman_rho'] = sr; row[f'{m}_spearman_p'] = sp
        corr_rows.append(row)
corr_df = pd.DataFrame(corr_rows)

print("\n===== CORRELATION TABLE: M2M vs extrapolation RMSE, within each model, by CV-equivalent region =====")
print(corr_df[['model','region_pct','n','M2M_pearson_r','M2M_pearson_p','M2M_spearman_rho','M2M_spearman_p']].to_string(index=False))

print("\n===== CORRELATION TABLE: R2_M, R2_X, R2_M_share vs extrapolation RMSE =====")
for m in ['R2_M', 'R2_X', 'R2_M_share']:
    print(f"\n--- {m} ---")
    print(corr_df[['model','region_pct','n',f'{m}_pearson_r',f'{m}_pearson_p',f'{m}_spearman_rho',f'{m}_spearman_p']].to_string(index=False))

# =====================================================================
# STEP 3/4: quintile analysis by M2M, within each region
# =====================================================================
quint_rows = []
for name in MODEL_NAMES:
    sub_full = df[df['model'] == name]
    rmin = sub_full['RMSECV'].min()
    for pct in REGIONS:
        region = sub_full[sub_full['RMSECV'] <= rmin * (1 + pct/100.0)].copy()
        n = len(region)
        if n < 5:
            quint_rows.append(dict(model=name, region_pct=pct, quintile='TOO_FEW_CONFIGS', n=n,
                                    median_M2M=np.nan, median_extrapRMSE=np.nan, mean_extrapRMSE=np.nan,
                                    sd_extrapRMSE=np.nan, min_extrapRMSE=np.nan, max_extrapRMSE=np.nan))
            continue
        try:
            region['quintile'] = pd.qcut(region['M2M'], 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        except ValueError:
            region['quintile'] = pd.qcut(region['M2M'].rank(method='first'), min(5, n), labels=False, duplicates='drop')
        for q, grp in region.groupby('quintile', observed=True):
            if len(grp) == 0:
                continue
            quint_rows.append(dict(model=name, region_pct=pct, quintile=str(q), n=len(grp),
                                    median_M2M=grp['M2M'].median(), median_extrapRMSE=grp['RMSE_extrapolation'].median(),
                                    mean_extrapRMSE=grp['RMSE_extrapolation'].mean(), sd_extrapRMSE=grp['RMSE_extrapolation'].std(ddof=1) if len(grp)>1 else 0.0,
                                    min_extrapRMSE=grp['RMSE_extrapolation'].min(), max_extrapRMSE=grp['RMSE_extrapolation'].max()))
quint_df = pd.DataFrame(quint_rows)

print("\n===== QUINTILE ANALYSIS (M2M quintiles vs extrapolation RMSE), region = 5% =====")
print(quint_df[quint_df['region_pct']==5].to_string(index=False))
print("\n===== QUINTILE ANALYSIS, region = 2% =====")
print(quint_df[quint_df['region_pct']==2].to_string(index=False))
print("\n===== QUINTILE ANALYSIS, region = 1% (small-n regions flagged) =====")
print(quint_df[quint_df['region_pct']==1].to_string(index=False))

# =====================================================================
# STEP 6/7: direction of relationship + consistency across models
# =====================================================================
print("\n===== DIRECTION OF RELATIONSHIP (M2M vs extrapolation RMSE), 5% region =====")
direction_rows = []
for _, row in corr_df[corr_df['region_pct']==5].iterrows():
    r, p, rho, ps = row['M2M_pearson_r'], row['M2M_pearson_p'], row['M2M_spearman_rho'], row['M2M_spearman_p']
    if np.isnan(r):
        direction = 'insufficient data'
    elif p > 0.05 and ps > 0.05:
        direction = 'no significant relationship'
    elif r > 0:
        direction = 'higher M2M -> HIGHER extrap RMSE (worse)'
    else:
        direction = 'higher M2M -> LOWER extrap RMSE (better)'
    direction_rows.append(dict(model=row['model'], pearson_r=r, pearson_p=p, spearman_rho=rho, spearman_p=ps, direction=direction))
    print(f"{row['model']:<16} r={r:6.3f} (p={p:.3f})  rho={rho:6.3f} (p={ps:.3f})  -> {direction}")

directions = [d['direction'] for d in direction_rows if d['direction'] != 'insufficient data']
n_positive = sum('HIGHER extrap RMSE' in d for d in directions)
n_negative = sum('LOWER extrap RMSE' in d for d in directions)
n_none = sum('no significant' in d for d in directions)
print(f"\nAcross {len(directions)} models with usable data: {n_positive} positive (worse), {n_negative} negative (better), {n_none} not significant.")
print("CONSISTENT" if n_positive == 0 or n_negative == 0 else "NOT CONSISTENT -- direction of the relationship differs by model")

# =====================================================================
# SAVE
# =====================================================================
with pd.ExcelWriter('within_model_M2M_analysis.xlsx') as w:
    corr_df.to_excel(w, sheet_name='correlations', index=False)
    quint_df.to_excel(w, sheet_name='quintiles', index=False)
    pd.DataFrame(direction_rows).to_excel(w, sheet_name='direction_summary', index=False)
print("\nSaved within_model_M2M_analysis.xlsx")

# =====================================================================
# PLOTS
# =====================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_metric_vs_extrap(metric, xlabel, fname, title):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    axes = axes.ravel()
    for idx, name in enumerate(MODEL_NAMES):
        sub_full = df[df['model'] == name]
        rmin = sub_full['RMSECV'].min()
        region = sub_full[sub_full['RMSECV'] <= rmin * 1.05]
        ax = axes[idx]
        ax.scatter(region[metric], region['RMSE_extrapolation'], s=12, alpha=0.5, color='#3E7C8C')
        r, p = pearsonr(region[metric], region['RMSE_extrapolation'])
        if p < 0.10 and len(region) >= 5:
            z = np.polyfit(region[metric], region['RMSE_extrapolation'], 1)
            xs = np.linspace(region[metric].min(), region[metric].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color='red', linewidth=1.5)
        sig = '*' if p < 0.05 else ('~' if p < 0.10 else 'ns')
        ax.set_title(f"{name}\nr={r:.2f} (p={p:.3f}, {sig})  n={len(region)}", fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel('RMSE extrapolation'); ax.grid(alpha=0.3)
    axes[7].axis('off')
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout(); plt.savefig(fname, dpi=130, facecolor='white')
    print(f"Saved {fname}")

plot_metric_vs_extrap('M2M', 'M2M (calibration)', 'M2M_vs_extrapRMSE_within_model_5pct.png',
                       'M2M vs extrapolation RMSE, within-model, dRMSECV<=5% region (trend line if p<0.10)')
plot_metric_vs_extrap('R2_M_share', 'R2_M / (R2_M + R2_X)', 'R2Mshare_vs_extrapRMSE_within_model_5pct.png',
                       'R2_M/(R2_M+R2_X) vs extrapolation RMSE, within-model, dRMSECV<=5% region')

print("\nDONE.")

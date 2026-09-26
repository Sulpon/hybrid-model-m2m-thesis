"""
Step 5-10: extend the EXISTING SO-PLS grid (same helpers/orthogonalization
order as so_pls_experiment.py, same A_MAX/B_MAX/C_MAX) to predict the new
large test sets at every (a,b,c), merge with the already-computed
RMSECV/Q2 (from SO_PLS_results.xlsx, NOT recomputed -- same CV surface,
reused as-is), then run the M2M distribution / flat-region / quality
association analysis.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, pearsonr
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

# ---------------------------------------------------------------------------
# Load blocks (calibration + large test sets)
# ---------------------------------------------------------------------------
X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
CAL_X2_COLS = list(X2_cal.columns)
CAL_X1_COLS = list(X1_cal.columns)

X1_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='X1')[CAL_X1_COLS].values
Y_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='X2')[CAL_X2_COLS].values

X1_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='X1')[CAL_X1_COLS].values
Y_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='X2')[CAL_X2_COLS].values

X1c = X1_cal[CAL_X1_COLS].values
X2c = X2_cal.values


def load_M2_cal(model_name):
    return pd.read_excel(f"M2_{model_name}.xlsx", sheet_name='calibration')[CAL_X2_COLS].values


def load_M2_large(model_name):
    xl = pd.ExcelFile(f"M2_{model_name}_large.xlsx")
    m2w = pd.read_excel(xl, sheet_name='within_large')[CAL_X2_COLS].values
    m2e = pd.read_excel(xl, sheet_name='extrap_large')[CAL_X2_COLS].values
    return m2w, m2e


# ---------------------------------------------------------------------------
# Same core helpers as so_pls_experiment.py
# ---------------------------------------------------------------------------
def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n

def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    k = np.sqrt(B.shape[1])
    return mu, sd, k

def _scale_apply(B, mu, sd, k):
    return (B - mu) / sd / k

def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g


def full_grid_extended(X1, M2, X2, Y, ext_sets):
    """ext_sets: {name: (X1n, M2n, X2n, Yn)}. Returns R2,SSX1,SSM,SSX,SSF,M2M
    arrays (A,B,C) from a single in-sample calibration fit, plus RMSE arrays
    per external set, all computed with the SAME frozen calibration scaling."""
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2); m2s = _scale_apply(M2, muM2, sdM2, k2)
    muX2, sdX2, k3 = _scale_fit(X2); x2s = _scale_apply(X2, muX2, sdX2, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY) / sdy
    sst = float(np.sum(ys ** 2))

    ext_scaled = {name: (_scale_apply(x1n, muX1, sdX1, k1), _scale_apply(m2n, muM2, sdM2, k2),
                          _scale_apply(x2n, muX2, sdX2, k3), yn)
                  for name, (x1n, m2n, x2n, yn) in ext_sets.items()}

    R2 = np.zeros((A_MAX, B_MAX, C_MAX)); SSX1 = np.zeros_like(R2); SSM = np.zeros_like(R2)
    SSX = np.zeros_like(R2); SSF = np.zeros_like(R2)
    RMSE_ext = {name: np.zeros((A_MAX, B_MAX, C_MAX)) for name in ext_sets}

    p1, T1, Q1, af = _fit(x1s, ys, A_MAX)
    T1_ext = {name: p1.transform(v[0]) for name, v in ext_scaled.items()}
    for a in range(1, A_MAX + 1):
        ae = min(a, af); Ta = T1[:, :ae]
        ssx1 = float(np.sum((Ta @ Q1[:, :ae].T) ** 2))
        m2r, gM = _resid(Ta, m2s); x2_a, gX = _resid(Ta, x2s); Ytr_a, gY = _resid(Ta, ys)
        Ta_ext = {name: T1_ext[name][:, :ae] for name in ext_sets}
        m2r_ext = {name: ext_scaled[name][1] - Ta_ext[name] @ gM for name in ext_sets}
        x2_a_ext = {name: ext_scaled[name][2] - Ta_ext[name] @ gX for name in ext_sets}
        yh_a_ext = {name: Ta_ext[name] @ Q1[:, :ae].T for name in ext_sets}

        p2, T2m, Q2, bf = _fit(m2r, Ytr_a, B_MAX)
        T2_ext = {name: p2.transform(m2r_ext[name]) for name in ext_sets}
        for b in range(1, B_MAX + 1):
            be = min(b, bf); Tb = T2m[:, :be]
            ssm = float(np.sum((Tb @ Q2[:, :be].T) ** 2))
            x2_b, gX2 = _resid(Tb, x2_a); Ytr_b, gY2 = _resid(Tb, Ytr_a)
            Tb_ext = {name: T2_ext[name][:, :be] for name in ext_sets}
            x2_b_ext = {name: x2_a_ext[name] - Tb_ext[name] @ gX2 for name in ext_sets}
            yh_ab_ext = {name: yh_a_ext[name] + Tb_ext[name] @ Q2[:, :be].T for name in ext_sets}

            p3, T3, Q3, cf = _fit(x2_b, Ytr_b, C_MAX)
            T3_ext = {name: p3.transform(x2_b_ext[name]) for name in ext_sets}
            for c in range(1, C_MAX + 1):
                ce = min(c, cf); Tc = T3[:, :ce]
                ssx = float(np.sum((Tc @ Q3[:, :ce].T) ** 2))
                ssf = sst - ssx1 - ssm - ssx
                R2[a-1,b-1,c-1] = 1 - ssf/sst
                SSX1[a-1,b-1,c-1] = ssx1; SSM[a-1,b-1,c-1] = ssm; SSX[a-1,b-1,c-1] = ssx; SSF[a-1,b-1,c-1] = ssf
                for name in ext_sets:
                    Tc_ext = T3_ext[name][:, :ce]
                    yhat = (yh_ab_ext[name] + Tc_ext @ Q3[:, :ce].T) * sdy + muY
                    yn = ext_sets[name][3]
                    RMSE_ext[name][a-1,b-1,c-1] = np.sqrt(np.mean((yhat.ravel() - yn.ravel()) ** 2))
    M2M = SSM / np.maximum(SSX, 1e-9)
    return R2, SSX1, SSM, SSX, SSF, M2M, RMSE_ext


# ---------------------------------------------------------------------------
# STEP 5: run extended grid for all 7 models, merge with existing RMSECV/Q2
# ---------------------------------------------------------------------------
prev = pd.read_excel('SO_PLS_results.xlsx', sheet_name='full_grid')  # RMSECV, Q2 reused as-is (NOT recomputed)

all_rows = []
for name in MODEL_NAMES:
    print(f"=== {name} ===")
    M2c = load_M2_cal(name)
    M2w, M2e = load_M2_large(name)
    ext_sets = {'within_large': (X1_w, M2w, X2_w, Y_w), 'extrap_large': (X1_e, M2e, X2_e, Y_e)}
    R2s, SSX1s, SSMs, SSXs, SSFs, M2Ms, RMSE_ext = full_grid_extended(X1c, M2c, X2c, Y_cal, ext_sets)

    prev_m = prev[prev['model'] == name].set_index(['LV_X1', 'LV_M2', 'LV_X2'])
    for a in range(A_MAX):
        for b in range(B_MAX):
            for c in range(C_MAX):
                key = (a+1, b+1, c+1)
                rmsecv = prev_m.loc[key, 'RMSECV'] if key in prev_m.index else np.nan
                q2 = prev_m.loc[key, 'Q2'] if key in prev_m.index else np.nan
                all_rows.append(dict(model=name, LV_X1=a+1, LV_M2=b+1, LV_X2=c+1,
                                      RMSECV=rmsecv, Q2=q2, R2=R2s[a,b,c],
                                      SS_X1=SSX1s[a,b,c], SS_M=SSMs[a,b,c], SS_X=SSXs[a,b,c], SS_F=SSFs[a,b,c],
                                      M2M=M2Ms[a,b,c], RMSE_within_large=RMSE_ext['within_large'][a,b,c],
                                      RMSE_extrap_large=RMSE_ext['extrap_large'][a,b,c]))
    i = np.unravel_index(np.array(prev_m['RMSECV']).reshape(A_MAX,B_MAX,C_MAX).argmin() if False else 0, (1,))  # placeholder unused
    print(f"  grid extended: {A_MAX*B_MAX*C_MAX} configs")

full_df = pd.DataFrame(all_rows)
print(f"\nMerged grid: {len(full_df)} rows, any NaN RMSECV/Q2 (grid mismatch check): "
      f"{full_df[['RMSECV','Q2']].isna().any().any()}")

# ---------------------------------------------------------------------------
# STEP 6 + 7: distribution stats, full grid AND flat regions (dRMSECV <=1/2/5%)
# ---------------------------------------------------------------------------
def dist_stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, min=np.nan, max=np.nan, median=np.nan, P10=np.nan, P25=np.nan, P50=np.nan,
                    P75=np.nan, P90=np.nan, range=np.nan, sd=np.nan, skewness=np.nan)
    return dict(n=len(v), min=v.min(), max=v.max(), median=np.median(v),
                P10=np.percentile(v,10), P25=np.percentile(v,25), P50=np.percentile(v,50),
                P75=np.percentile(v,75), P90=np.percentile(v,90), range=v.max()-v.min(),
                sd=v.std(ddof=1) if len(v)>1 else 0.0, skewness=float(skew(v)) if len(v)>2 else np.nan)

region_rows = []
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    rmin = sub['RMSECV'].min()
    imin = sub['RMSECV'].idxmin()
    m2m_at_min = sub.loc[imin, 'M2M']

    full_stats = dist_stats(sub['M2M'])
    region_rows.append(dict(model=name, region='full_grid', RMSECV_min=rmin, M2M_at_min=m2m_at_min, **full_stats))

    for pct in [1, 2, 5]:
        thr = rmin * (1 + pct/100.0)
        flat = sub[sub['RMSECV'] <= thr]
        st = dist_stats(flat['M2M'])
        region_rows.append(dict(model=name, region=f'dRMSECV_le_{pct}pct', RMSECV_min=rmin, M2M_at_min=m2m_at_min, **st))

region_df = pd.DataFrame(region_rows)
print("\n===== STEP 6/7: M2M distribution stats per model/region =====")
print(region_df[['model','region','n','RMSECV_min','M2M_at_min','min','median','P90','max','range','sd','skewness']].to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 9: summary table (min-RMSECV config, incl. large-test-set errors)
# ---------------------------------------------------------------------------
summary_rows = []
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    imin = sub['RMSECV'].idxmin()
    row = sub.loc[imin]
    med = dist_stats(sub['M2M'])
    summary_rows.append(dict(model=name, LV_X1=int(row['LV_X1']), LV_M2=int(row['LV_M2']), LV_X2=int(row['LV_X2']),
                              min_RMSECV=row['RMSECV'], M2M_at_min=row['M2M'], median_M2M=med['median'],
                              P90_M2M=med['P90'], range_M2M=med['range'], sd_M2M=med['sd'], skew_M2M=med['skewness'],
                              RMSE_within_large=row['RMSE_within_large'], RMSE_extrap_large=row['RMSE_extrap_large']))
summary_df = pd.DataFrame(summary_rows)
print("\n===== STEP 9: SUMMARY TABLE (large independent test set, min-RMSECV config) =====")
print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 8: M2M distribution structure vs extrapolation RMSE (across 7 models)
# ---------------------------------------------------------------------------
print("\n===== STEP 8: correlation (n=7 models) between M2M-distribution statistics and RMSE_extrap_large =====")
stat_cols = ['M2M_at_min', 'median_M2M', 'P90_M2M', 'range_M2M', 'sd_M2M', 'skew_M2M']
corr_rows = []
for col in stat_cols:
    r, p = pearsonr(summary_df[col], summary_df['RMSE_extrap_large'])
    corr_rows.append(dict(statistic=col, pearson_r_vs_RMSE_extrap_large=r, p_value=p))
corr_df = pd.DataFrame(corr_rows)
print(corr_df.to_string(index=False))
print("(n=7 models -- descriptive/exploratory only, not a hypothesis test with real power)")

# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
with pd.ExcelWriter('SO_PLS_results_large_test.xlsx') as writer:
    full_df.to_excel(writer, sheet_name='full_grid_extended', index=False)
    region_df.to_excel(writer, sheet_name='M2M_distribution_by_region', index=False)
    summary_df.to_excel(writer, sheet_name='summary_min_rmsecv', index=False)
    corr_df.to_excel(writer, sheet_name='M2M_vs_extrap_correlation', index=False)
print("\nSaved SO_PLS_results_large_test.xlsx (sheets: full_grid_extended, M2M_distribution_by_region, summary_min_rmsecv, M2M_vs_extrap_correlation)")

# ---------------------------------------------------------------------------
# STEP 10: plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1 & 2: M2M distributions within dRMSECV<=1% and <=2% regions, all 7 models
for pct, fname in [(1, 'M2M_dist_dRMSECV_1pct.png'), (2, 'M2M_dist_dRMSECV_2pct.png')]:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for idx, name in enumerate(MODEL_NAMES):
        sub = full_df[full_df['model'] == name]
        rmin = sub['RMSECV'].min()
        flat = sub[sub['RMSECV'] <= rmin*(1+pct/100.0)]
        ax = axes[idx]
        ax.hist(flat['M2M'], bins=20, color='#3E7C8C', alpha=0.8, edgecolor='black', linewidth=0.4)
        ax.set_title(f"{name}\nn={len(flat)}  median={flat['M2M'].median():.2f}", fontsize=9)
        ax.set_xlabel('M2M'); ax.grid(alpha=0.3)
    axes[7].axis('off')
    plt.suptitle(f'M2M distribution within flat region (dRMSECV <= {pct}%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=130, facecolor='white')
    print(f"Saved {fname}")

# 3: M2M summary statistic vs extrapolation RMSE (across models)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for idx, col in enumerate(stat_cols):
    ax = axes[idx]
    ax.scatter(summary_df['RMSE_extrap_large'], summary_df[col], s=50)
    for _, row in summary_df.iterrows():
        ax.annotate(row['model'].replace('_',' '), (row['RMSE_extrap_large'], row[col]), fontsize=7)
    ax.set_xlabel('RMSE extrapolation (large test set)'); ax.set_ylabel(col); ax.grid(alpha=0.3)
plt.suptitle('M2M distribution statistics vs extrapolation RMSE (n=7 models)', fontweight='bold')
plt.tight_layout()
plt.savefig('M2M_stat_vs_extrapolation_RMSE.png', dpi=130, facecolor='white')
print("Saved M2M_stat_vs_extrapolation_RMSE.png")

# 4: per-model M2M vs RMSECV scatter
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for idx, name in enumerate(MODEL_NAMES):
    sub = full_df[full_df['model'] == name]
    ax = axes[idx]
    ax.scatter(sub['RMSECV'], sub['M2M'], s=5, alpha=0.4, color='#16323F')
    ax.set_yscale('log'); ax.set_title(name, fontsize=9)
    ax.set_xlabel('RMSECV'); ax.set_ylabel('M2M'); ax.grid(alpha=0.3)
axes[7].axis('off')
plt.suptitle('M2M vs RMSECV, full LV grid, per model', fontweight='bold')
plt.tight_layout()
plt.savefig('M2M_vs_RMSECV_per_model.png', dpi=130, facecolor='white')
print("Saved M2M_vs_RMSECV_per_model.png")

# 5: M2M vs extrapolation RMSE across the full LV grid (large test set), all models
fig, ax = plt.subplots(figsize=(9, 6))
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    ax.scatter(sub['RMSE_extrap_large'], sub['M2M'], s=5, alpha=0.3, label=name)
ax.set_yscale('log'); ax.set_xlabel('RMSE extrapolation (large test set, per LV config)'); ax.set_ylabel('M2M')
ax.set_title('M2M vs extrapolation RMSE, full LV grid, all 7 models')
ax.legend(markerscale=3, fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('M2M_vs_extrapolationRMSE_fullgrid.png', dpi=130, facecolor='white')
print("Saved M2M_vs_extrapolationRMSE_fullgrid.png")

print("\nDONE.")

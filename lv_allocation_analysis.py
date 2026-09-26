"""
LV-allocation sensitivity analysis (Parts 1-8). Reuses the existing
calibration/large-test data and the same SO-PLS core (_fit/_scale/_resid,
same orthogonalization order X1->M2->X2, same A_MAX=6/B_MAX=15/C_MAX=15 grid)
as so_pls_experiment.py / m2m_distribution_analysis.py. No data regenerated,
no generator or SO-PLS algorithm changed. RMSECV/Q2 are reused verbatim from
SO_PLS_results.xlsx (not recomputed). New quantities computed here:
  - RMSE_calibration: computed directly from the in-sample fit's residuals
    (a straightforward read-out of the existing final-model fit, not a
    change to the algorithm)
  - M2M_extrapolation, RMSE_extrapolation, RMSE_within: same fitted model
    (frozen calibration scaling + loadings), transformed onto the large
    external test sets and decomposed the same way as the calibration SS
    decomposition. Caveat: the T1/T2/T3 orthogonal-projection basis is
    fit on calibration; on external data the cross-terms are not
    guaranteed to vanish, so SSF_extrapolation can be negative -- this is
    reported as a diagnostic, not smoothed over. RMSE_extrapolation is
    always computed directly from residuals (never derived from SSF).
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

# ---------------------------------------------------------------------------
# Load blocks (all already generated / saved in earlier turns)
# ---------------------------------------------------------------------------
X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
CAL_X2_COLS = list(X2_cal.columns); CAL_X1_COLS = list(X1_cal.columns)

X1_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='X1')[CAL_X1_COLS].values
Y_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_w = pd.read_excel('DG_test_within_domain_large.xlsx', sheet_name='X2')[CAL_X2_COLS].values

X1_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='X1')[CAL_X1_COLS].values
Y_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_e = pd.read_excel('DG_test_extrapolation_large.xlsx', sheet_name='X2')[CAL_X2_COLS].values

X1c = X1_cal[CAL_X1_COLS].values
X2c = X2_cal.values

def load_M2_cal(name): return pd.read_excel(f"M2_{name}.xlsx", sheet_name='calibration')[CAL_X2_COLS].values
def load_M2_large(name):
    xl = pd.ExcelFile(f"M2_{name}_large.xlsx")
    return (pd.read_excel(xl, sheet_name='within_large')[CAL_X2_COLS].values,
            pd.read_excel(xl, sheet_name='extrap_large')[CAL_X2_COLS].values)

prev = pd.read_excel('SO_PLS_results.xlsx', sheet_name='full_grid')  # RMSECV, Q2 -- reused, not recomputed

# ---------------------------------------------------------------------------
# Core helpers (identical to so_pls_experiment.py / m2m_distribution_analysis.py)
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


def full_grid_v2(X1, M2, X2, Ycal, X1e, M2e, X2e, Ye, X1w, M2w, X2w, Yw):
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2); m2s = _scale_apply(M2, muM2, sdM2, k2)
    muX2, sdX2, k3 = _scale_fit(X2); x2s = _scale_apply(X2, muX2, sdX2, k3)
    muY, sdy = Ycal.mean(0), Ycal.std(0, ddof=1); ys = (Ycal - muY) / sdy
    sst = float(np.sum(ys ** 2))

    x1se = _scale_apply(X1e, muX1, sdX1, k1); m2se = _scale_apply(M2e, muM2, sdM2, k2); x2se = _scale_apply(X2e, muX2, sdX2, k3)
    yse = (Ye - muY) / sdy; sste = float(np.sum(yse ** 2))
    x1sw = _scale_apply(X1w, muX1, sdX1, k1); m2sw = _scale_apply(M2w, muM2, sdM2, k2); x2sw = _scale_apply(X2w, muX2, sdX2, k3)

    shp = (A_MAX, B_MAX, C_MAX)
    out = {k: np.zeros(shp) for k in ['RMSE_cal', 'M2M_cal', 'SSX1_cal', 'SSM_cal', 'SSX_cal', 'SSF_cal',
                                       'RMSE_ext', 'M2M_ext', 'SSX1_ext', 'SSM_ext', 'SSX_ext', 'SSF_ext',
                                       'RMSE_within']}

    p1, T1, Q1, af = _fit(x1s, ys, A_MAX)
    T1e = p1.transform(x1se); T1w = p1.transform(x1sw)
    for a in range(1, A_MAX + 1):
        ae = min(a, af); Ta, Tae, Taw = T1[:, :ae], T1e[:, :ae], T1w[:, :ae]
        ssx1 = float(np.sum((Ta @ Q1[:, :ae].T) ** 2))
        ssx1e = float(np.sum((Tae @ Q1[:, :ae].T) ** 2))
        yh_a = Ta @ Q1[:, :ae].T; yh_ae = Tae @ Q1[:, :ae].T; yh_aw = Taw @ Q1[:, :ae].T

        m2r, gM = _resid(Ta, m2s); x2_a, gX = _resid(Ta, x2s); Ytr_a, gY = _resid(Ta, ys)
        m2re = m2se - Tae @ gM; x2_ae = x2se - Tae @ gX
        m2rw = m2sw - Taw @ gM; x2_aw = x2sw - Taw @ gX

        p2, T2m, Q2, bf = _fit(m2r, Ytr_a, B_MAX)
        T2e = p2.transform(m2re); T2w = p2.transform(m2rw)
        for b in range(1, B_MAX + 1):
            be = min(b, bf); Tb, Tbe, Tbw = T2m[:, :be], T2e[:, :be], T2w[:, :be]
            ssm = float(np.sum((Tb @ Q2[:, :be].T) ** 2))
            ssme = float(np.sum((Tbe @ Q2[:, :be].T) ** 2))
            yh_ab = yh_a + Tb @ Q2[:, :be].T; yh_abe = yh_ae + Tbe @ Q2[:, :be].T; yh_abw = yh_aw + Tbw @ Q2[:, :be].T

            x2_b, gX2 = _resid(Tb, x2_a); Ytr_b, gY2 = _resid(Tb, Ytr_a)
            x2_be = x2_ae - Tbe @ gX2; x2_bw = x2_aw - Tbw @ gX2

            p3, T3, Q3, cf = _fit(x2_b, Ytr_b, C_MAX)
            T3e = p3.transform(x2_be); T3w = p3.transform(x2_bw)
            for c in range(1, C_MAX + 1):
                ce = min(c, cf); Tc, Tce, Tcw = T3[:, :ce], T3e[:, :ce], T3w[:, :ce]
                ssx = float(np.sum((Tc @ Q3[:, :ce].T) ** 2))
                ssxe = float(np.sum((Tce @ Q3[:, :ce].T) ** 2))
                ssf = sst - ssx1 - ssm - ssx
                ssfe = sste - ssx1e - ssme - ssxe

                yhat_cal = (yh_ab + Tc @ Q3[:, :ce].T) * sdy + muY
                rmse_cal = float(np.sqrt(np.mean((yhat_cal.ravel() - Ycal.ravel()) ** 2)))
                yhat_ext = (yh_abe + Tce @ Q3[:, :ce].T) * sdy + muY
                rmse_ext = float(np.sqrt(np.mean((yhat_ext.ravel() - Ye.ravel()) ** 2)))
                yhat_w = (yh_abw + Tcw @ Q3[:, :ce].T) * sdy + muY
                rmse_w = float(np.sqrt(np.mean((yhat_w.ravel() - Yw.ravel()) ** 2)))

                i = (a-1, b-1, c-1)
                out['SSX1_cal'][i]=ssx1; out['SSM_cal'][i]=ssm; out['SSX_cal'][i]=ssx; out['SSF_cal'][i]=ssf
                out['M2M_cal'][i]=ssm/max(ssx,1e-9); out['RMSE_cal'][i]=rmse_cal
                out['SSX1_ext'][i]=ssx1e; out['SSM_ext'][i]=ssme; out['SSX_ext'][i]=ssxe; out['SSF_ext'][i]=ssfe
                out['M2M_ext'][i]=ssme/max(ssxe,1e-9); out['RMSE_ext'][i]=rmse_ext
                out['RMSE_within'][i]=rmse_w
    return out


print("Running extended grid (calibration + extrapolation SS decomposition) for all 7 models...")
all_rows = []
for name in MODEL_NAMES:
    M2c = load_M2_cal(name); M2w, M2e = load_M2_large(name)
    o = full_grid_v2(X1c, M2c, X2c, Y_cal, X1_e, M2e, X2_e, Y_e, X1_w, M2w, X2_w, Y_w)
    prev_m = prev[prev['model'] == name].set_index(['LV_X1', 'LV_M2', 'LV_X2'])
    for a in range(A_MAX):
        for b in range(B_MAX):
            for c in range(C_MAX):
                key = (a+1, b+1, c+1); i = (a, b, c)
                rmsecv = prev_m.loc[key, 'RMSECV'] if key in prev_m.index else np.nan
                q2 = prev_m.loc[key, 'Q2'] if key in prev_m.index else np.nan
                all_rows.append(dict(model=name, LV_X1=a+1, LV_M2=b+1, LV_X2=c+1, RMSECV=rmsecv, Q2=q2,
                                      RMSE_calibration=o['RMSE_cal'][i], M2M_calibration=o['M2M_cal'][i],
                                      SS_X1_cal=o['SSX1_cal'][i], SS_M_cal=o['SSM_cal'][i], SS_X_cal=o['SSX_cal'][i], SS_F_cal=o['SSF_cal'][i],
                                      RMSE_within=o['RMSE_within'][i],
                                      RMSE_extrapolation=o['RMSE_ext'][i], M2M_extrapolation=o['M2M_ext'][i],
                                      SS_X1_ext=o['SSX1_ext'][i], SS_M_ext=o['SSM_ext'][i], SS_X_ext=o['SSX_ext'][i], SS_F_ext=o['SSF_ext'][i]))
    print(f"  {name} done")

full_df = pd.DataFrame(all_rows)
print(f"\nfull_grid: {len(full_df)} rows. Negative SS_F_ext count (out-of-sample non-orthogonality diagnostic): "
      f"{(full_df['SS_F_ext'] < 0).sum()} / {len(full_df)}")

# =====================================================================
# PART 1: best configs per criterion (A-D)
# =====================================================================
best_rows = []
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    for crit_label, col in [('A_min_RMSECV', 'RMSECV'), ('B_min_calibration_RMSE', 'RMSE_calibration'),
                             ('C_min_within_RMSE', 'RMSE_within'), ('D_min_extrapolation_RMSE', 'RMSE_extrapolation')]:
        idx = sub[col].idxmin()
        row = sub.loc[idx]
        best_rows.append(dict(model=name, criterion=crit_label, LV_X1=int(row['LV_X1']), LV_M2=int(row['LV_M2']), LV_X2=int(row['LV_X2']),
                               RMSE_calibration=row['RMSE_calibration'], RMSECV=row['RMSECV'], Q2=row['Q2'],
                               RMSE_within=row['RMSE_within'], RMSE_extrapolation=row['RMSE_extrapolation'],
                               M2M_calibration=row['M2M_calibration']))
best_df = pd.DataFrame(best_rows)

# =====================================================================
# PART 2: LV sensitivity -- A (min RMSECV) vs D (min extrapolation RMSE)
# =====================================================================
sens_rows = []
for name in MODEL_NAMES:
    A = best_df[(best_df['model']==name) & (best_df['criterion']=='A_min_RMSECV')].iloc[0]
    D = best_df[(best_df['model']==name) & (best_df['criterion']=='D_min_extrapolation_RMSE')].iloc[0]
    d_rmsecv = D['RMSECV'] - A['RMSECV']; pct_rmsecv = 100*d_rmsecv/A['RMSECV']
    d_ext = D['RMSE_extrapolation'] - A['RMSE_extrapolation']; pct_ext = 100*d_ext/A['RMSE_extrapolation']
    d_m2m = D['M2M_calibration'] - A['M2M_calibration']
    sens_rows.append(dict(model=name, LV_A=(A['LV_X1'],A['LV_M2'],A['LV_X2']), LV_D=(D['LV_X1'],D['LV_M2'],D['LV_X2']),
                           RMSECV_A=A['RMSECV'], RMSECV_D=D['RMSECV'], delta_RMSECV=d_rmsecv, pct_delta_RMSECV=pct_rmsecv,
                           RMSE_extrap_A=A['RMSE_extrapolation'], RMSE_extrap_D=D['RMSE_extrapolation'],
                           delta_RMSE_extrap=d_ext, pct_delta_RMSE_extrap=pct_ext,
                           M2M_A=A['M2M_calibration'], M2M_D=D['M2M_calibration'], delta_M2M=d_m2m))
sens_df = pd.DataFrame(sens_rows)
print("\n===== PART 2: LV sensitivity (A=min RMSECV vs D=min extrapolation RMSE) =====")
print(sens_df.to_string(index=False))

# fold sensitivity deltas into best_config_comparison sheet as extra columns on the D rows
best_df = best_df.merge(sens_df[['model','delta_RMSECV','pct_delta_RMSECV','delta_RMSE_extrap','pct_delta_RMSE_extrap','delta_M2M']],
                         on='model', how='left')
best_df.loc[best_df['criterion'] != 'D_min_extrapolation_RMSE', ['delta_RMSECV','pct_delta_RMSECV','delta_RMSE_extrap','pct_delta_RMSE_extrap','delta_M2M']] = np.nan

# =====================================================================
# PART 3: flat regions (dRMSECV <= 1/2/5%), extrapolation RMSE + M2M dist
# =====================================================================
def dist_stats(v, prefix=''):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    if len(v) == 0:
        keys = ['min','median','max','sd','P10','P50','P90','range','skewness']
        return {f'{prefix}{k}': np.nan for k in keys}
    return {f'{prefix}min': v.min(), f'{prefix}median': np.median(v), f'{prefix}max': v.max(),
            f'{prefix}sd': v.std(ddof=1) if len(v)>1 else 0.0, f'{prefix}P10': np.percentile(v,10),
            f'{prefix}P50': np.percentile(v,50), f'{prefix}P90': np.percentile(v,90),
            f'{prefix}range': v.max()-v.min(), f'{prefix}skewness': float(skew(v)) if len(v)>2 else np.nan}

flat_region_dfs = {}
for pct in [1, 2, 5]:
    rows = []
    for name in MODEL_NAMES:
        sub = full_df[full_df['model'] == name]
        rmin = sub['RMSECV'].min()
        flat = sub[sub['RMSECV'] <= rmin*(1+pct/100.0)]
        row = dict(model=name, n_configs=len(flat), RMSECV_min=rmin)
        row.update(dist_stats(flat['RMSE_extrapolation'], prefix='RMSE_extrap_'))
        row.update(dist_stats(flat['M2M_calibration'], prefix='M2M_'))
        rows.append(row)
    flat_region_dfs[pct] = pd.DataFrame(rows)
    print(f"\n===== PART 3: flat region dRMSECV<= {pct}% =====")
    print(flat_region_dfs[pct].to_string(index=False))

# =====================================================================
# PART 4: calibration-only summary (full grid + dRMSECV<=5%)
# =====================================================================
cal_summary_rows = []
for region_name, get_sub in [('full_grid', lambda s: s), ('dRMSECV_le_5pct', lambda s: s[s['RMSECV'] <= s['RMSECV'].min()*1.05])]:
    for name in MODEL_NAMES:
        sub = full_df[full_df['model'] == name]
        rmin_idx = sub['RMSECV'].idxmin(); rmin_row = sub.loc[rmin_idx]
        region_sub = get_sub(sub)
        row = dict(model=name, region=region_name, LV_X1=int(rmin_row['LV_X1']), LV_M2=int(rmin_row['LV_M2']), LV_X2=int(rmin_row['LV_X2']),
                   min_RMSECV=rmin_row['RMSECV'], M2M_at_min=rmin_row['M2M_calibration'], n_configs=len(region_sub))
        row.update(dist_stats(region_sub['M2M_calibration'], prefix='M2M_'))
        row.update(dist_stats(region_sub['RMSE_calibration'], prefix='calRMSE_'))
        cal_summary_rows.append(row)
cal_summary_df = pd.DataFrame(cal_summary_rows)
print("\n===== PART 4: calibration-only summary =====")
print(cal_summary_df.to_string(index=False))

# =====================================================================
# PART 5: calibration vs extrapolation side-by-side (full-grid stats)
# =====================================================================
cve_rows = []
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    rmin_idx = sub['RMSECV'].idxmin(); r = sub.loc[rmin_idx]
    cve_rows.append(dict(
        model=name,
        M2M_at_min_RMSECV=r['M2M_calibration'],
        calibration_M2M_median=sub['M2M_calibration'].median(), calibration_M2M_P90=np.percentile(sub['M2M_calibration'],90),
        extrapolation_M2M_median=sub['M2M_extrapolation'].median(), extrapolation_M2M_P90=np.percentile(sub['M2M_extrapolation'],90),
        calibration_RMSE_at_min=r['RMSE_calibration'], calibration_RMSE_median=sub['RMSE_calibration'].median(), calibration_RMSE_P90=np.percentile(sub['RMSE_calibration'],90),
        extrapolation_RMSE_at_min=r['RMSE_extrapolation'], extrapolation_RMSE_median=sub['RMSE_extrapolation'].median(), extrapolation_RMSE_P90=np.percentile(sub['RMSE_extrapolation'],90),
        min_RMSECV=r['RMSECV'], Q2=r['Q2'],
        calibration_M2M_range=sub['M2M_calibration'].max()-sub['M2M_calibration'].min(),
        extrapolation_M2M_range=sub['M2M_extrapolation'].max()-sub['M2M_extrapolation'].min(),
        calibration_M2M_sd=sub['M2M_calibration'].std(ddof=1), extrapolation_M2M_sd=sub['M2M_extrapolation'].std(ddof=1),
        calibration_M2M_skew=float(skew(sub['M2M_calibration'])), extrapolation_M2M_skew=float(skew(sub['M2M_extrapolation'])),
    ))
cve_df = pd.DataFrame(cve_rows)

# =====================================================================
# PART 6: cal M2M vs ext M2M correlation (per model, across LV configs)
# =====================================================================
part6_rows = []
for name in MODEL_NAMES:
    sub = full_df[full_df['model'] == name]
    for region_label, region_sub in [('full_grid', sub), ('dRMSECV_le_5pct', sub[sub['RMSECV'] <= sub['RMSECV'].min()*1.05])]:
        pr, _ = pearsonr(region_sub['M2M_calibration'], region_sub['M2M_extrapolation'])
        sr, _ = spearmanr(region_sub['M2M_calibration'], region_sub['M2M_extrapolation'])
        absdiff = (region_sub['M2M_calibration'] - region_sub['M2M_extrapolation']).abs()
        reldiff = (absdiff / region_sub['M2M_calibration'].abs().replace(0, np.nan))
        part6_rows.append(dict(model=name, region=region_label, n=len(region_sub),
                                pearson_r=pr, spearman_r=sr, median_abs_diff=absdiff.median(),
                                median_rel_diff_pct=100*reldiff.median()))
part6_df = pd.DataFrame(part6_rows)
print("\n===== PART 6: M2M_calibration vs M2M_extrapolation correlation (same LV configs) =====")
print(part6_df.to_string(index=False))

cve_df = cve_df.merge(part6_df[part6_df['region']=='full_grid'][['model','pearson_r','spearman_r','median_abs_diff','median_rel_diff_pct']]
                       .rename(columns={'pearson_r':'M2M_cal_vs_ext_pearson_r_fullgrid','spearman_r':'M2M_cal_vs_ext_spearman_r_fullgrid',
                                        'median_abs_diff':'M2M_cal_vs_ext_median_absdiff_fullgrid','median_rel_diff_pct':'M2M_cal_vs_ext_median_reldiff_pct_fullgrid'}),
                       on='model', how='left')
cve_df = cve_df.merge(part6_df[part6_df['region']=='dRMSECV_le_5pct'][['model','pearson_r','spearman_r','median_abs_diff','median_rel_diff_pct']]
                       .rename(columns={'pearson_r':'M2M_cal_vs_ext_pearson_r_5pct','spearman_r':'M2M_cal_vs_ext_spearman_r_5pct',
                                        'median_abs_diff':'M2M_cal_vs_ext_median_absdiff_5pct','median_rel_diff_pct':'M2M_cal_vs_ext_median_reldiff_pct_5pct'}),
                       on='model', how='left')

print("\n===== PART 5: calibration vs extrapolation side-by-side =====")
print(cve_df.to_string(index=False))

# =====================================================================
# SAVE WORKBOOK
# =====================================================================
with pd.ExcelWriter('LV_allocation_analysis.xlsx') as w:
    best_df.to_excel(w, sheet_name='best_config_comparison', index=False)
    cal_summary_df.to_excel(w, sheet_name='calibration_summary', index=False)
    # extrapolation_summary mirrors calibration_summary structure but for extrapolation
    ext_summary_rows = []
    for region_name, get_sub in [('full_grid', lambda s: s), ('dRMSECV_le_5pct', lambda s: s[s['RMSECV'] <= s['RMSECV'].min()*1.05])]:
        for name in MODEL_NAMES:
            sub = full_df[full_df['model'] == name]
            rmin_idx = sub['RMSECV'].idxmin(); rmin_row = sub.loc[rmin_idx]
            region_sub = get_sub(sub)
            row = dict(model=name, region=region_name, LV_X1=int(rmin_row['LV_X1']), LV_M2=int(rmin_row['LV_M2']), LV_X2=int(rmin_row['LV_X2']),
                       min_RMSECV=rmin_row['RMSECV'], M2M_ext_at_min=rmin_row['M2M_extrapolation'], n_configs=len(region_sub))
            row.update(dist_stats(region_sub['M2M_extrapolation'], prefix='M2Mext_'))
            row.update(dist_stats(region_sub['RMSE_extrapolation'], prefix='extRMSE_'))
            ext_summary_rows.append(row)
    ext_summary_df = pd.DataFrame(ext_summary_rows)
    ext_summary_df.to_excel(w, sheet_name='extrapolation_summary', index=False)
    flat_region_dfs[1].to_excel(w, sheet_name='flat_region_1pct', index=False)
    flat_region_dfs[2].to_excel(w, sheet_name='flat_region_2pct', index=False)
    flat_region_dfs[5].to_excel(w, sheet_name='flat_region_5pct', index=False)
    cve_df.to_excel(w, sheet_name='calibration_vs_extrapolation', index=False)
    full_df.to_excel(w, sheet_name='full_grid', index=False)
print("\nSaved LV_allocation_analysis.xlsx (8 sheets)")

# =====================================================================
# PART 7 answers (printed)
# =====================================================================
print("\n===== PART 7: interpretation =====")
for _, row in sens_df.iterrows():
    print(f"{row['model']:<16} RMSECV: {row['RMSECV_A']:.4f}(A) vs {row['RMSECV_D']:.4f}(D)  "
          f"pct_delta={row['pct_delta_RMSECV']:.2f}%   |   extrapRMSE: {row['RMSE_extrap_A']:.3f}(A) vs {row['RMSE_extrap_D']:.3f}(D)  "
          f"pct_delta={row['pct_delta_RMSE_extrap']:.1f}%   |   M2M: {row['M2M_A']:.2f}(A) vs {row['M2M_D']:.2f}(D)")

# =====================================================================
# PLOTS
# =====================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. LV_allocation_vs_extrapolation.png
fig, axes = plt.subplots(2, 4, figsize=(19, 9))
axes = axes.ravel()
for idx, name in enumerate(MODEL_NAMES):
    sub = full_df[full_df['model'] == name]
    ax = axes[idx]
    ax.scatter(sub['RMSECV'], sub['RMSE_extrapolation'], s=5, alpha=0.25, color='gray')
    A = best_df[(best_df['model']==name) & (best_df['criterion']=='A_min_RMSECV')].iloc[0]
    D = best_df[(best_df['model']==name) & (best_df['criterion']=='D_min_extrapolation_RMSE')].iloc[0]
    Arow = sub[(sub['LV_X1']==A['LV_X1'])&(sub['LV_M2']==A['LV_M2'])&(sub['LV_X2']==A['LV_X2'])].iloc[0]
    Drow = sub[(sub['LV_X1']==D['LV_X1'])&(sub['LV_M2']==D['LV_M2'])&(sub['LV_X2']==D['LV_X2'])].iloc[0]
    ax.scatter([Arow['RMSECV']], [Arow['RMSE_extrapolation']], marker='*', s=220, color='red', edgecolor='black', label='min RMSECV', zorder=5)
    ax.scatter([Drow['RMSECV']], [Drow['RMSE_extrapolation']], marker='*', s=220, color='blue', edgecolor='black', label='min extrap RMSE', zorder=5)
    ax.set_title(name, fontsize=9); ax.set_xlabel('RMSECV'); ax.set_ylabel('RMSE extrapolation'); ax.grid(alpha=0.3)
    if idx == 0: ax.legend(fontsize=7)
axes[7].axis('off')
plt.suptitle('LV grid: RMSECV vs extrapolation RMSE, min-RMSECV and min-extrap-RMSE configs marked', fontweight='bold')
plt.tight_layout(); plt.savefig('LV_allocation_vs_extrapolation.png', dpi=130, facecolor='white')
print("Saved LV_allocation_vs_extrapolation.png")

# 2. calibration_M2M_distributions.png (full grid, all 7 models)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()
for idx, name in enumerate(MODEL_NAMES):
    sub = full_df[full_df['model'] == name]
    ax = axes[idx]
    ax.hist(sub['M2M_calibration'], bins=30, color='#3E7C8C', alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_title(f"{name}\nmedian={sub['M2M_calibration'].median():.2f}", fontsize=9)
    ax.set_xlabel('M2M (calibration)'); ax.grid(alpha=0.3)
axes[7].axis('off')
plt.suptitle('Calibration M2M distribution, full LV grid', fontweight='bold')
plt.tight_layout(); plt.savefig('calibration_M2M_distributions.png', dpi=130, facecolor='white')
print("Saved calibration_M2M_distributions.png")

# 3/4/5/6: two-panel (full grid | dRMSECV<=5%) scatter plots
def two_panel_scatter(xcol, ycol, xlabel, ylabel, fname, title, logy=False):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, get_sub) in zip(axs, [('full grid', lambda s: s), ('dRMSECV<=5%', lambda s: s[s['RMSECV']<=s['RMSECV'].min()*1.05])]):
        for name in MODEL_NAMES:
            sub = full_df[full_df['model'] == name]
            region_sub = get_sub(sub)
            ax.scatter(region_sub[xcol], region_sub[ycol], s=6, alpha=0.35, label=name)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(label); ax.grid(alpha=0.3)
        if logy: ax.set_yscale('log')
    axs[0].legend(fontsize=7, markerscale=3)
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout(); plt.savefig(fname, dpi=130, facecolor='white')
    print(f"Saved {fname}")

two_panel_scatter('M2M_calibration', 'M2M_extrapolation', 'M2M (calibration)', 'M2M (extrapolation)',
                   'calibration_vs_extrapolation_M2M.png', 'M2M: calibration vs extrapolation (same LV configs)', logy=False)
two_panel_scatter('RMSE_calibration', 'RMSE_extrapolation', 'RMSE calibration', 'RMSE extrapolation',
                   'calibration_vs_extrapolation_RMSE.png', 'RMSE: calibration vs extrapolation (same LV configs)')
two_panel_scatter('RMSECV', 'RMSE_extrapolation', 'RMSECV', 'RMSE extrapolation',
                   'RMSECV_vs_extrapolation_RMSE.png', 'RMSECV vs extrapolation RMSE')
two_panel_scatter('M2M_extrapolation', 'RMSE_extrapolation', 'M2M (extrapolation)', 'RMSE extrapolation',
                   'M2M_vs_extrapolation_RMSE.png', 'M2M (extrapolation-based) vs extrapolation RMSE')

print("\nDONE.")

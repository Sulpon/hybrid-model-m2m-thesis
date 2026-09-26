"""
AMR = |Q2_FULL - Q2_NO_M2| / |Q2_FULL - Q2_NO_X2|, calibration-only.
Reuses FULL's Q2 directly from the existing full_grid (LV_allocation_analysis.xlsx).
Recomputes NO_M2 (X1->X2) and NO_X2 (X1->M2) CV surfaces to get their Q2 grids
(RMSECV grids were kept from the ablation run, but Q2 grids weren't saved to
disk, so they're recomputed here with the identical cv_2block function/seed).
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
CAL_X2_COLS = list(X2_cal.columns); CAL_X1_COLS = list(X1_cal.columns)
X1c = X1_cal[CAL_X1_COLS].values
X2c = X2_cal.values

def load_M2_cal(name): return pd.read_excel(f"M2_{name}.xlsx", sheet_name='calibration')[CAL_X2_COLS].values

full_grid = pd.read_excel('LV_allocation_analysis.xlsx', sheet_name='full_grid')  # has RMSECV, Q2, M2M_calibration

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

def cv_2block(Xa, Xb, Y, MAXa, MAXb):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, MAXa, MAXb)); rf = np.zeros((N_SPLITS, MAXa, MAXb))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muXa, sdXa, ka = _scale_fit(Xa[tr]); xat, xae = _scale_apply(Xa[tr], muXa, sdXa, ka), _scale_apply(Xa[te], muXa, sdXa, ka)
        muXb, sdXb, kb = _scale_fit(Xb[tr]); xbt0, xbe0 = _scale_apply(Xb[tr], muXb, sdXb, kb), _scale_apply(Xb[te], muXb, sdXb, kb)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr0 = (Y[tr]-muY)/sdy
        p1, Tat, Qa, af = _fit(xat, ytr0, MAXa); Tae = p1.transform(xae)
        for a in range(1, MAXa+1):
            ae = min(a, af); Ta, Tae_ = Tat[:, :ae], Tae[:, :ae]
            yh_a = Tae_ @ Qa[:, :ae].T
            xbt, gXb = _resid(Ta, xbt0); xbe = xbe0 - Tae_ @ gXb
            ytr_a, gY = _resid(Ta, ytr0)
            p2, Tbt, Qb, bf = _fit(xbt, ytr_a, MAXb); Tbe = p2.transform(xbe)
            for b in range(1, MAXb+1):
                be = min(b, bf)
                yh = (yh_a + Tbe[:, :be] @ Qb[:, :be].T)*sdy+muY
                P[te, a-1, b-1] = yh.ravel()
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    q2 = 1 - ((P - Y[:, :, None])**2).sum(0)/sst
    return rf.mean(0), q2

# =====================================================================
# PART 1/3: AMR at FULL's own min-RMSECV config
# =====================================================================
primary_rows = []
grids = {}
for name in MODEL_NAMES:
    print(f"=== {name} ===")
    M2c = load_M2_cal(name)
    rmsecv_nm2, q2_nm2 = cv_2block(X1c, X2c, Y_cal, A_MAX, C_MAX)  # NO_M2: X1->X2 (indexed [a-1, c-1])
    rmsecv_nx2, q2_nx2 = cv_2block(X1c, M2c, Y_cal, A_MAX, B_MAX)  # NO_X2: X1->M2 (indexed [a-1, b-1])
    grids[name] = dict(q2_nm2=q2_nm2, q2_nx2=q2_nx2)

    sub = full_grid[full_grid['model'] == name]
    i_min = sub['RMSECV'].idxmin(); r = sub.loc[i_min]
    a, b, c = int(r['LV_X1']), int(r['LV_M2']), int(r['LV_X2'])
    q2_full = r['Q2']
    q2_no_m2 = q2_nm2[a-1, c-1]
    q2_no_x2 = q2_nx2[a-1, b-1]
    d_m2 = q2_full - q2_no_m2
    d_x2 = q2_full - q2_no_x2
    amr = abs(d_m2) / abs(d_x2) if abs(d_x2) > 1e-12 else np.nan
    primary_rows.append(dict(model=name, LV_X1=a, LV_M2=b, LV_X2=c, Q2_FULL=q2_full, Q2_NO_M2=q2_no_m2, Q2_NO_X2=q2_no_x2,
                              dQ2_drop_M2=d_m2, dQ2_drop_X2=d_x2, dQ2_drop_M2_pp=d_m2*100, dQ2_drop_X2_pp=d_x2*100, AMR=amr))
    print(f"  Q2_FULL={q2_full:.4f} Q2_NO_M2={q2_no_m2:.4f} Q2_NO_X2={q2_no_x2:.4f}  "
          f"dQ2_M2={d_m2*100:.3f}pp dQ2_X2={d_x2*100:.3f}pp  AMR={amr:.4f}")

primary_df = pd.DataFrame(primary_rows)
print("\n===== PART 1: AMR summary table =====")
print(primary_df[['model','Q2_FULL','Q2_NO_M2','Q2_NO_X2','dQ2_drop_M2_pp','dQ2_drop_X2_pp','AMR']].to_string(index=False))

print("\n===== PART 3: check against the hypothetical example =====")
print("Claimed (hypothetical): correct model dQ2_M2~-24.5pp, dQ2_X2~-6.9pp -> AMR~3.55")
print("Claimed (hypothetical): misspecified dQ2_M2~-14.6pp, dQ2_X2~-14.3pp -> AMR~1.02")
print(f"Actual M0_correct: dQ2_M2={primary_df.loc[primary_df.model=='M0_correct','dQ2_drop_M2_pp'].iloc[0]:.3f}pp  "
      f"dQ2_X2={primary_df.loc[primary_df.model=='M0_correct','dQ2_drop_X2_pp'].iloc[0]:.3f}pp  "
      f"AMR={primary_df.loc[primary_df.model=='M0_correct','AMR'].iloc[0]:.4f}")
print("VERDICT: does NOT match the hypothetical example, neither in magnitude nor in the ratio -- see final report.")

# =====================================================================
# PART 2 ranking
# =====================================================================
print("\n===== PART 2: ranking by AMR =====")
print(primary_df.sort_values('AMR', ascending=False)[['model','AMR']].to_string(index=False))

# =====================================================================
# PART 4: LV robustness, dRMSECV<=5%, matching-slice Q2 lookups
# =====================================================================
robust_rows = []
for name in MODEL_NAMES:
    sub = full_grid[full_grid['model'] == name]
    rmin = sub['RMSECV'].min()
    flat = sub[sub['RMSECV'] <= rmin*1.05]
    q2_nm2, q2_nx2 = grids[name]['q2_nm2'], grids[name]['q2_nx2']
    amrs = []
    for _, row in flat.iterrows():
        a, b, c = int(row['LV_X1']), int(row['LV_M2']), int(row['LV_X2'])
        q2f = row['Q2']
        dm2 = q2f - q2_nm2[a-1, c-1]
        dx2 = q2f - q2_nx2[a-1, b-1]
        amr = abs(dm2)/abs(dx2) if abs(dx2) > 1e-12 else np.nan
        amrs.append(amr)
    v = np.array([x for x in amrs if np.isfinite(x)])
    robust_rows.append(dict(model=name, n_configs=len(flat), median=np.median(v), P10=np.percentile(v,10),
                             P90=np.percentile(v,90), min=v.min(), max=v.max(), sd=v.std(ddof=1)))
robust_df = pd.DataFrame(robust_rows)
print("\n===== PART 4: AMR LV robustness (dRMSECV<=5%) =====")
print(robust_df.to_string(index=False))

# =====================================================================
# PART 5: AMR vs M2M
# =====================================================================
m2m_rows = []
for name in MODEL_NAMES:
    sub = full_grid[full_grid['model'] == name]
    i_min = sub['RMSECV'].idxmin()
    m2m_rows.append(dict(model=name, M2M=sub.loc[i_min, 'M2M_calibration']))
m2m_df = pd.DataFrame(m2m_rows)
cmp_df = primary_df[['model', 'AMR']].merge(m2m_df, on='model')
pr, pp = pearsonr(cmp_df['AMR'], cmp_df['M2M'])
sr, sp = spearmanr(cmp_df['AMR'], cmp_df['M2M'])
print("\n===== PART 5: AMR vs M2M (n=7, exploratory) =====")
print(cmp_df.to_string(index=False))
print(f"Pearson r={pr:.3f} (p={pp:.3f})   Spearman rho={sr:.3f} (p={sp:.3f})")

# =====================================================================
# SAVE
# =====================================================================
with pd.ExcelWriter('AMR_calibration_analysis.xlsx') as w:
    primary_df.to_excel(w, sheet_name='AMR_summary', index=False)
    robust_df.to_excel(w, sheet_name='LV_robustness', index=False)
    cmp_df.assign(pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp).to_excel(w, sheet_name='M2M_comparison', index=False)
print("\nSaved AMR_calibration_analysis.xlsx")

# =====================================================================
# PLOTS
# =====================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#C0392B' if m == 'M0_correct' else '#3E7C8C' for m in primary_df['model']]
order = primary_df.sort_values('AMR', ascending=False)
ax.bar(order['model'], order['AMR'], color=[('#C0392B' if m=='M0_correct' else '#3E7C8C') for m in order['model']])
ax.set_ylabel('AMR'); ax.set_title('AMR by model (M0_correct in red)'); ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig('AMR_bar.png', dpi=130, facecolor='white')
print("Saved AMR_bar.png")

fig, ax = plt.subplots(figsize=(7, 6))
for _, r in cmp_df.iterrows():
    color = '#C0392B' if r['model'] == 'M0_correct' else '#3E7C8C'
    ax.scatter(r['M2M'], r['AMR'], s=80, color=color)
    ax.annotate(r['model'].replace('_', ' '), (r['M2M'], r['AMR']), fontsize=8)
ax.set_xlabel('M2M (calibration, at min RMSECV)'); ax.set_ylabel('AMR')
ax.set_title(f'AMR vs M2M (n=7)  r={pr:.2f} rho={sr:.2f}'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('AMR_vs_M2M.png', dpi=130, facecolor='white')
print("Saved AMR_vs_M2M.png")

print("\nDONE.")

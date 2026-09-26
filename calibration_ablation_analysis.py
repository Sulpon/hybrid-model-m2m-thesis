"""
Calibration-only ablation study (Parts 1-9). Uses ONLY the 100 calibration
batches -- no extrapolation/within-domain data anywhere in this script.
Reuses the exact same SO-PLS core (_fit/_scale/_resid, same 10-fold CV
seed=42, same block-scaling by 1/sqrt(K)) as every prior script in this
analysis. FULL (X1->M2->X2) grid is NOT recomputed -- read directly from
the existing LV_allocation_analysis.xlsx full_grid sheet. Only the four
ablation structures (NO_M2, NO_X2, X1_ONLY, M2_ONLY) are newly computed,
using generic 1-block/2-block versions of the same CV + full-fit machinery.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

# ---------------------------------------------------------------------------
# Load calibration-only blocks
# ---------------------------------------------------------------------------
X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
CAL_X2_COLS = list(X2_cal.columns); CAL_X1_COLS = list(X1_cal.columns)
X1c = X1_cal[CAL_X1_COLS].values
X2c = X2_cal.values

def load_M2_cal(name): return pd.read_excel(f"M2_{name}.xlsx", sheet_name='calibration')[CAL_X2_COLS].values

full_grid = pd.read_excel('LV_allocation_analysis.xlsx', sheet_name='full_grid')  # FULL model, reused verbatim

# ---------------------------------------------------------------------------
# Core helpers (identical to all prior scripts)
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

# ---------------------------------------------------------------------------
# Generic 1-block and 2-block CV + full-fit surfaces (same conventions)
# ---------------------------------------------------------------------------
def cv_1block(X, Y, MAXn):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y); P = np.zeros((n, MAXn)); rf = np.zeros((N_SPLITS, MAXn))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muX, sdX, k = _scale_fit(X[tr]); xt, xe = _scale_apply(X[tr], muX, sdX, k), _scale_apply(X[te], muX, sdX, k)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ytr = (Y[tr]-muY)/sdy
        p, Tt, Q, af = _fit(xt, ytr, MAXn); Te = p.transform(xe)
        for a in range(1, MAXn+1):
            ae = min(a, af)
            yh = (Te[:, :ae] @ Q[:, :ae].T) * sdy + muY
            P[te, a-1] = yh.ravel()
            rf[f, a-1] = np.sqrt(np.mean((yh.ravel()-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    q2 = 1 - ((P - Y)**2).sum(0)/sst
    return rf.mean(0), q2

def full_fit_1block(X, Y, MAXn):
    muX, sdX, k = _scale_fit(X); xs = _scale_apply(X, muX, sdX, k)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    p, T, Q, af = _fit(xs, ys, MAXn)
    rmse = np.zeros(MAXn); r2 = np.zeros(MAXn)
    sst = float(np.sum(ys**2))
    for a in range(1, MAXn+1):
        ae = min(a, af)
        yhat = (T[:, :ae] @ Q[:, :ae].T)*sdy+muY
        rmse[a-1] = np.sqrt(np.mean((yhat.ravel()-Y.ravel())**2))
        ssf = sst - float(np.sum((T[:, :ae]@Q[:, :ae].T)**2))
        r2[a-1] = 1 - ssf/sst
    return rmse, r2

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

def full_fit_2block(Xa, Xb, Y, MAXa, MAXb):
    muXa, sdXa, ka = _scale_fit(Xa); xas = _scale_apply(Xa, muXa, sdXa, ka)
    muXb, sdXb, kb = _scale_fit(Xb); xbs = _scale_apply(Xb, muXb, sdXb, kb)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    rmse = np.zeros((MAXa, MAXb)); r2 = np.zeros((MAXa, MAXb))
    p1, Ta_, Qa, af = _fit(xas, ys, MAXa)
    for a in range(1, MAXa+1):
        ae = min(a, af); Ta = Ta_[:, :ae]
        ssxa = float(np.sum((Ta@Qa[:, :ae].T)**2))
        yh_a = Ta@Qa[:, :ae].T
        xbr, gXb = _resid(Ta, xbs); ytr_a, gY = _resid(Ta, ys)
        p2, Tb_, Qb, bf = _fit(xbr, ytr_a, MAXb)
        for b in range(1, MAXb+1):
            be = min(b, bf); Tb = Tb_[:, :be]
            yhat = (yh_a + Tb@Qb[:, :be].T)*sdy+muY
            rmse[a-1, b-1] = np.sqrt(np.mean((yhat.ravel()-Y.ravel())**2))
            ssxb = float(np.sum((Tb@Qb[:, :be].T)**2))
            ssf = sst - ssxa - ssxb
            r2[a-1, b-1] = 1 - ssf/sst
    return rmse, r2

# =====================================================================
# PART 1: fit ablation structures (NO_M2, NO_X2, X1_ONLY, M2_ONLY) for all 7 models
# X1_ONLY is identical across models (same X1 block) -- computed once
# =====================================================================
print("Computing X1_ONLY (shared across all models -- same X1 block)...")
rmsecv_X1o, q2_X1o = cv_1block(X1c, Y_cal, A_MAX)
rmse_X1o, r2_X1o = full_fit_1block(X1c, Y_cal, A_MAX)
best_a_X1o = int(np.argmin(rmsecv_X1o))
X1_ONLY = dict(RMSECV=rmsecv_X1o[best_a_X1o], Q2=q2_X1o[best_a_X1o], RMSE_cal=rmse_X1o[best_a_X1o], R2=r2_X1o[best_a_X1o], LV=best_a_X1o+1)
print(f"  X1_ONLY: LV={X1_ONLY['LV']} RMSECV={X1_ONLY['RMSECV']:.4f} Q2={X1_ONLY['Q2']:.4f}")

structures = {}  # name -> dict of per-model results + grids
for name in MODEL_NAMES:
    print(f"=== {name} ===")
    M2c = load_M2_cal(name)

    rmsecv_nm2, q2_nm2 = cv_2block(X1c, X2c, Y_cal, A_MAX, C_MAX)  # NO_M2: X1 -> X2
    rmse_nm2, r2_nm2 = full_fit_2block(X1c, X2c, Y_cal, A_MAX, C_MAX)
    i_nm2 = np.unravel_index(rmsecv_nm2.argmin(), rmsecv_nm2.shape)
    NO_M2 = dict(RMSECV=rmsecv_nm2[i_nm2], Q2=q2_nm2[i_nm2], RMSE_cal=rmse_nm2[i_nm2], R2=r2_nm2[i_nm2], LV=(i_nm2[0]+1, i_nm2[1]+1))

    rmsecv_nx2, q2_nx2 = cv_2block(X1c, M2c, Y_cal, A_MAX, B_MAX)  # NO_X2: X1 -> M2
    rmse_nx2, r2_nx2 = full_fit_2block(X1c, M2c, Y_cal, A_MAX, B_MAX)
    i_nx2 = np.unravel_index(rmsecv_nx2.argmin(), rmsecv_nx2.shape)
    NO_X2 = dict(RMSECV=rmsecv_nx2[i_nx2], Q2=q2_nx2[i_nx2], RMSE_cal=rmse_nx2[i_nx2], R2=r2_nx2[i_nx2], LV=(i_nx2[0]+1, i_nx2[1]+1))

    rmsecv_m2o, q2_m2o = cv_1block(M2c, Y_cal, B_MAX)
    rmse_m2o, r2_m2o = full_fit_1block(M2c, Y_cal, B_MAX)
    b_m2o = int(np.argmin(rmsecv_m2o))
    M2_ONLY = dict(RMSECV=rmsecv_m2o[b_m2o], Q2=q2_m2o[b_m2o], RMSE_cal=rmse_m2o[b_m2o], R2=r2_m2o[b_m2o], LV=b_m2o+1)

    sub = full_grid[full_grid['model'] == name]
    i_full = sub['RMSECV'].idxmin(); rfull = sub.loc[i_full]
    SST_FULL = float(rfull['SS_X1_cal'] + rfull['SS_M_cal'] + rfull['SS_X_cal'] + rfull['SS_F_cal'])
    r2_full = 1 - rfull['SS_F_cal'] / SST_FULL
    FULL = dict(RMSECV=rfull['RMSECV'], Q2=rfull['Q2'], RMSE_cal=rfull['RMSE_calibration'], R2=r2_full,
                LV=(int(rfull['LV_X1']), int(rfull['LV_M2']), int(rfull['LV_X2'])), M2M=rfull['M2M_calibration'])

    structures[name] = dict(FULL=FULL, NO_M2=NO_M2, NO_X2=NO_X2, M2_ONLY=M2_ONLY, X1_ONLY=X1_ONLY,
                             grid_nm2=rmsecv_nm2, grid_nx2=rmsecv_nx2, grid_m2o=rmsecv_m2o)
    print(f"  FULL RMSECV={FULL['RMSECV']:.4f} LV={FULL['LV']}   NO_M2={NO_M2['RMSECV']:.4f} LV={NO_M2['LV']}   "
          f"NO_X2={NO_X2['RMSECV']:.4f} LV={NO_X2['LV']}   M2_ONLY={M2_ONLY['RMSECV']:.4f} LV={M2_ONLY['LV']}")

ablation_rows = []
for name in MODEL_NAMES:
    for struct in ['FULL', 'NO_M2', 'NO_X2', 'X1_ONLY', 'M2_ONLY']:
        s = structures[name][struct]
        ablation_rows.append(dict(model=name, structure=struct, LV=str(s['LV']), RMSECV=s['RMSECV'], Q2=s['Q2'],
                                   RMSE_calibration=s['RMSE_cal'], R2=s['R2']))
ablation_df = pd.DataFrame(ablation_rows)
print("\n===== PART 1: ablation results =====")
print(ablation_df.to_string(index=False))

# =====================================================================
# PART 2/3/4/5: unique contributions, MIR, MAR, redundancy (own-best-config)
# =====================================================================
summary_rows = []
for name in MODEL_NAMES:
    s = structures[name]
    RF, RNM2, RNX2, RM2O, RX1O = s['FULL']['RMSECV'], s['NO_M2']['RMSECV'], s['NO_X2']['RMSECV'], s['M2_ONLY']['RMSECV'], s['X1_ONLY']['RMSECV']
    UMC = (RNM2 - RF) / RNM2
    UXC = (RNX2 - RF) / RNX2
    delta_M = RNM2 - RF
    delta_X = RNX2 - RF
    MIR = UMC / (UMC + UXC) if abs(UMC + UXC) > 1e-9 else np.nan
    MAR = RF / RM2O
    mech_penalty = (RM2O - RF) / RF
    G_M = RNM2 - RF; G_X = RNX2 - RF; G_BOTH = RX1O - RF
    denom = G_M + G_X
    redundancy = 1 - G_BOTH/denom if denom > 1e-9 else np.nan
    summary_rows.append(dict(model=name, RMSECV_FULL=RF, Q2_FULL=s['FULL']['Q2'], RMSECV_NO_M2=RNM2, RMSECV_NO_X2=RNX2,
                              RMSECV_M2_ONLY=RM2O, RMSECV_X1_ONLY=RX1O, delta_M=delta_M, delta_X=delta_X,
                              UMC=UMC, UXC=UXC, MIR=MIR, MAR=MAR, mechanistic_penalty=mech_penalty,
                              G_M=G_M, G_X=G_X, G_BOTH=G_BOTH, Redundancy_Index=redundancy,
                              M2M_calibration_at_min=s['FULL']['M2M']))
summary_df = pd.DataFrame(summary_rows)
print("\n===== PARTS 2-5: unique contributions, MIR, MAR, redundancy =====")
print(summary_df.to_string(index=False))

# =====================================================================
# PART 6: rankings
# =====================================================================
print("\n===== PART 6: rankings =====")
for col, asc in [('RMSECV_FULL', True), ('UMC', False), ('UXC', False), ('MIR', False), ('MAR', False)]:
    order = summary_df.sort_values(col, ascending=asc)['model'].tolist()
    print(f"  by {col}: {order}")

# =====================================================================
# PART 8: M2M vs ablation metrics correlation (n=7, exploratory)
# =====================================================================
print("\n===== PART 8: M2M vs ablation metrics correlation (n=7, exploratory only) =====")
corr8_rows = []
for col in ['UMC', 'UXC', 'MIR', 'MAR', 'RMSECV_FULL']:
    valid = summary_df[['M2M_calibration_at_min', col]].dropna()
    if len(valid) >= 3:
        r, p = pearsonr(valid['M2M_calibration_at_min'], valid[col])
    else:
        r, p = np.nan, np.nan
    corr8_rows.append(dict(metric=col, pearson_r_vs_M2M=r, p_value=p))
    print(f"  M2M vs {col}: r={r:.3f} p={p:.3f}")
corr8_df = pd.DataFrame(corr8_rows)

# =====================================================================
# PART 9: LV robustness within FULL's own dRMSECV<=5% region
# =====================================================================
print("\n===== PART 9: LV robustness (dRMSECV<=5% region of FULL) =====")
robust_rows = []
for name in MODEL_NAMES:
    s = structures[name]
    sub = full_grid[full_grid['model'] == name]
    rmin = sub['RMSECV'].min()
    flat = sub[sub['RMSECV'] <= rmin*1.05]
    umc_list, uxc_list, mir_list, mar_list = [], [], [], []
    for _, row in flat.iterrows():
        a, b, c = int(row['LV_X1']), int(row['LV_M2']), int(row['LV_X2'])
        rf = row['RMSECV']
        rnm2 = s['grid_nm2'][a-1, c-1]
        rnx2 = s['grid_nx2'][a-1, b-1]
        rm2o = rmsecv_m2o[b-1] if name == name else None  # placeholder, fixed below
        rm2o = structures[name]['grid_m2o'][b-1]
        umc = (rnm2-rf)/rnm2; uxc = (rnx2-rf)/rnx2
        mir = umc/(umc+uxc) if abs(umc+uxc) > 1e-9 else np.nan
        mar = rf/rm2o
        umc_list.append(umc); uxc_list.append(uxc); mir_list.append(mir); mar_list.append(mar)
    def stats(v):
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v) == 0:
            return dict(median=np.nan, P10=np.nan, P90=np.nan, range=np.nan, n=0)
        return dict(median=np.median(v), P10=np.percentile(v,10), P90=np.percentile(v,90), range=v.max()-v.min(), n=len(v))
    for metric_name, vals in [('UMC', umc_list), ('UXC', uxc_list), ('MIR', mir_list), ('MAR', mar_list)]:
        st = stats(vals)
        robust_rows.append(dict(model=name, metric=metric_name, n_configs=st['n'], median=st['median'], P10=st['P10'], P90=st['P90'], range=st['range']))
robust_df = pd.DataFrame(robust_rows)
print(robust_df.to_string(index=False))

# =====================================================================
# SAVE
# =====================================================================
with pd.ExcelWriter('calibration_ablation_analysis.xlsx') as w:
    summary_df.to_excel(w, sheet_name='model_summary', index=False)
    ablation_df.to_excel(w, sheet_name='ablation_results', index=False)
    summary_df[['model','delta_M','delta_X','UMC','UXC']].to_excel(w, sheet_name='unique_contributions', index=False)
    summary_df[['model','MIR','MAR','mechanistic_penalty']].to_excel(w, sheet_name='MIR_MAR', index=False)
    summary_df[['model','G_M','G_X','G_BOTH','Redundancy_Index']].to_excel(w, sheet_name='redundancy', index=False)
    robust_df.to_excel(w, sheet_name='LV_robustness', index=False)
    corr8_df.to_excel(w, sheet_name='M2M_comparison', index=False)
print("\nSaved calibration_ablation_analysis.xlsx (7 sheets)")

# =====================================================================
# PLOTS
# =====================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. calibration_model_comparison.png
fig, ax = plt.subplots(figsize=(11, 5))
structs = ['FULL', 'NO_M2', 'NO_X2', 'X1_ONLY', 'M2_ONLY']
xw = np.arange(len(MODEL_NAMES)); w = 0.15
for i, st in enumerate(structs):
    vals = [structures[m][st]['RMSECV'] for m in MODEL_NAMES]
    ax.bar(xw + (i-2)*w, vals, width=w, label=st)
ax.set_xticks(xw); ax.set_xticklabels(MODEL_NAMES, rotation=30, ha='right')
ax.set_ylabel('RMSECV'); ax.set_title('Calibration-only RMSECV by model structure'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('calibration_model_comparison.png', dpi=130, facecolor='white')
print("Saved calibration_model_comparison.png")

# 2. UMC_vs_UXC.png
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(summary_df['UMC'], summary_df['UXC'], s=60)
for _, r in summary_df.iterrows():
    ax.annotate(r['model'].replace('_', ' '), (r['UMC'], r['UXC']), fontsize=8)
ax.set_xlabel('UMC'); ax.set_ylabel('UXC'); ax.set_title('Unique mechanistic vs unique measured contribution'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('UMC_vs_UXC.png', dpi=130, facecolor='white')
print("Saved UMC_vs_UXC.png")

# 3. MIR_vs_MAR.png -- the most important plot: x=UMC, y=MAR (per instructions, labeled)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(summary_df['UMC'], summary_df['MAR'], s=60, color='#16323F')
for _, r in summary_df.iterrows():
    ax.annotate(r['model'].replace('_', ' '), (r['UMC'], r['MAR']), fontsize=8)
ax.set_xlabel('UMC (unique mechanistic contribution)'); ax.set_ylabel('MAR (mechanistic adequacy ratio)')
ax.set_title('UMC vs MAR -- mechanistic contribution vs mechanistic adequacy'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('MIR_vs_MAR.png', dpi=130, facecolor='white')
print("Saved MIR_vs_MAR.png (x=UMC, y=MAR as specified)")

# 4. M2M_vs_MIR.png
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(summary_df['M2M_calibration_at_min'], summary_df['MIR'], s=60, color='#3E7C8C')
for _, r in summary_df.iterrows():
    ax.annotate(r['model'].replace('_', ' '), (r['M2M_calibration_at_min'], r['MIR']), fontsize=8)
ax.set_xlabel('M2M (calibration, at min RMSECV)'); ax.set_ylabel('MIR'); ax.set_title('M2M vs MIR (n=7, exploratory)'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('M2M_vs_MIR.png', dpi=130, facecolor='white')
print("Saved M2M_vs_MIR.png")

# 5. LV_robustness.png
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ['UMC', 'UXC', 'MIR']):
    sub = robust_df[robust_df['metric'] == metric]
    yerr_low = sub['median'] - sub['P10']; yerr_high = sub['P90'] - sub['median']
    ax.errorbar(range(len(sub)), sub['median'], yerr=[yerr_low, yerr_high], fmt='o', capsize=4)
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub['model'], rotation=40, ha='right', fontsize=8)
    ax.set_title(f'{metric}: median (P10-P90), dRMSECV<=5%'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('LV_robustness.png', dpi=130, facecolor='white')
print("Saved LV_robustness.png")

print("\nDONE.")

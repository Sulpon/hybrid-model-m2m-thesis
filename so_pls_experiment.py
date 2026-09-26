"""
SO-PLS experiment: X1 -> M2 -> X2 -> Y, for each of the 7 candidate KD models.

Reuses the SO-PLS core pattern already established in `m2m distribtuion.py`
(_fit helper, block scaling by 1/sqrt(K), sequential orthogonalization order,
10-fold CV with the same "fit once at max LVs, slice for smaller LV counts"
efficiency trick) and implements the paper's Algorithm 1 SS decomposition
(SSM, SSX, SSF from a single full-calibration fit) generalized to 3 blocks.

Two surfaces are computed per model, over the full (a,b,c) LV grid:
  - CV surface (10-fold on the 100 calibration batches): RMSECV, Q2 (=qabc)
  - Full-fit surface (single fit on all 100 calibration batches, in-sample):
    R2, SS_X1, SS_M2, SS_X2, SS_F, M2M = SS_M2/SS_X2

At each model's minimum-RMSECV (a,b,c): fit the final model on all 100
calibration batches (scaling frozen from calibration only) and predict
calibration / within-domain / extrapolation.

Timepoint handling (documented, not silently patched): extrapolation's X2/M2
have 8 timepoints (0-105s) vs calibration/within's 7 (0-90s, same sample
interval). The SO-PLS model's PLS loadings are sized to calibration's column
count, so extrapolation's block columns are aligned to calibration's column
NAMES before scaling/prediction -- this drops the extra *_t8 (105s) columns.
That 105s point is therefore not used as a predictor in this first pass; it
is reported separately below, not silently discarded.

M5 (M5_no_D) is used RAW (negative concentrations included, not clipped).
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15   # LV_X1, LV_M2, LV_X2 grid caps (matches m2m distribtuion.py)
N_SPLITS, SEED = 10, 42

MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

# ---------------------------------------------------------------------------
# Load blocks
# ---------------------------------------------------------------------------
X1_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X1')
Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X1_within = pd.read_excel('DG_test_within_domain.xlsx', sheet_name='X1')
Y_within = pd.read_excel('DG_test_within_domain.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X1_extrap = pd.read_excel('DG_test_extrapolation.xlsx', sheet_name='X1')
Y_extrap = pd.read_excel('DG_test_extrapolation.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)

X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')
X2_within = pd.read_excel('DG_test_within_domain.xlsx', sheet_name='X2')
X2_extrap_raw = pd.read_excel('DG_test_extrapolation.xlsx', sheet_name='X2')

CAL_X2_COLS = list(X2_cal.columns)          # 45 cols, t1..t7 -- the reference column set
X2_extrap = X2_extrap_raw[CAL_X2_COLS]      # drop *_t8 (105s) -- not used as a predictor here
dropped_105s_cols = [c for c in X2_extrap_raw.columns if c not in CAL_X2_COLS]
print(f"Timepoint alignment: dropping {len(dropped_105s_cols)} extrapolation-only columns "
      f"(the 105s point) to match calibration's {len(CAL_X2_COLS)}-column X2/M2 structure:")
print(f"  {dropped_105s_cols}")

CAL_X1_COLS = list(X1_cal.columns)


def load_M2(model_name):
    xl = pd.ExcelFile(f"M2_{model_name}.xlsx")
    m2_cal = pd.read_excel(xl, sheet_name='calibration')[CAL_X2_COLS]
    m2_within = pd.read_excel(xl, sheet_name='within_domain')[CAL_X2_COLS]
    m2_extrap_raw = pd.read_excel(xl, sheet_name='extrapolation')
    m2_extrap = m2_extrap_raw[CAL_X2_COLS]
    return m2_cal.values, m2_within.values, m2_extrap.values


X1c, X1w, X1e = X1_cal[CAL_X1_COLS].values, X1_within[CAL_X1_COLS].values, X1_extrap[CAL_X1_COLS].values
X2c, X2w, X2e = X2_cal.values, X2_within.values, X2_extrap.values

# ---------------------------------------------------------------------------
# Core helpers (same pattern as m2m distribtuion.py's _fit / _scale)
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
    """Regress target on scores T (least squares) and return residual + coefs."""
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g


# ---------------------------------------------------------------------------
# CV surface: RMSECV, Q2 across the (a,b,c) grid, 10-fold on calibration only
# ---------------------------------------------------------------------------
def cv_surface(X1, M2, X2, Y):
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    n = len(Y)
    P_abc = np.zeros((n, A_MAX, B_MAX, C_MAX))
    rf = np.zeros((N_SPLITS, A_MAX, B_MAX, C_MAX))
    for f, (tr, te) in enumerate(cv.split(Y)):
        muX1, sdX1, k1 = _scale_fit(X1[tr]); x1t, x1e = _scale_apply(X1[tr], muX1, sdX1, k1), _scale_apply(X1[te], muX1, sdX1, k1)
        muM2, sdM2, k2 = _scale_fit(M2[tr]); m2t0, m2e0 = _scale_apply(M2[tr], muM2, sdM2, k2), _scale_apply(M2[te], muM2, sdM2, k2)
        muX2, sdX2, k3 = _scale_fit(X2[tr]); x2t0, x2e0 = _scale_apply(X2[tr], muX2, sdX2, k3), _scale_apply(X2[te], muX2, sdX2, k3)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy

        p1, T1t, Q1, af = _fit(x1t, Ytr0, A_MAX); T1e = p1.transform(x1e)
        for a in range(1, A_MAX + 1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            m2t, gM = _resid(Tat, m2t0); m2e = m2e0 - Tae @ gM
            x2t_a, gX = _resid(Tat, x2t0); x2e_a = x2e0 - Tae @ gX
            Ytr_a, gY = _resid(Tat, Ytr0)
            p2, T2t, Q2, bf = _fit(m2t, Ytr_a, B_MAX); T2e = p2.transform(m2e)
            for b in range(1, B_MAX + 1):
                be = min(b, bf); Tbt, Tbe = T2t[:, :be], T2e[:, :be]
                yh_ab = yh_a + Tbe @ Q2[:, :be].T
                x2t_b, gX2 = _resid(Tbt, x2t_a); x2e_b = x2e_a - Tbe @ gX2
                Ytr_b, gY2 = _resid(Tbt, Ytr_a)
                p3, T3t, Q3, cf = _fit(x2t_b, Ytr_b, C_MAX); T3e = p3.transform(x2e_b)
                for c in range(1, C_MAX + 1):
                    ce = min(c, cf)
                    yh = (yh_ab + T3e[:, :ce] @ Q3[:, :ce].T) * sdy + muY
                    P_abc[te, a - 1, b - 1, c - 1] = yh.ravel()
                    rf[f, a - 1, b - 1, c - 1] = np.sqrt(np.mean((yh.ravel() - Y[te].ravel()) ** 2))
    sst = np.sum((Y - Y.mean(0)) ** 2)
    qabc = 1 - ((P_abc - Y[:, :, None, None]) ** 2).sum(0) / sst
    rmsecv = rf.mean(0)
    return rmsecv, qabc


# ---------------------------------------------------------------------------
# Full-fit surface (in-sample, all 100 calibration batches): SSM/SSX/SSF/R2/M2M
# ---------------------------------------------------------------------------
def full_fit_surface(X1, M2, X2, Y):
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2); m2s = _scale_apply(M2, muM2, sdM2, k2)
    muX2, sdX2, k3 = _scale_fit(X2); x2s = _scale_apply(X2, muX2, sdX2, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY) / sdy
    sst = float(np.sum(ys ** 2))

    R2 = np.zeros((A_MAX, B_MAX, C_MAX)); SSX1 = np.zeros_like(R2); SSM = np.zeros_like(R2)
    SSX = np.zeros_like(R2); SSF = np.zeros_like(R2)

    p1, T1, Q1, af = _fit(x1s, ys, A_MAX)
    for a in range(1, A_MAX + 1):
        ae = min(a, af); Ta = T1[:, :ae]
        ssx1 = float(np.sum((Ta @ Q1[:, :ae].T) ** 2))
        m2r, gM = _resid(Ta, m2s)
        x2_a, gX = _resid(Ta, x2s)
        Ytr_a, gY = _resid(Ta, ys)
        p2, T2, Q2, bf = _fit(m2r, Ytr_a, B_MAX)
        for b in range(1, B_MAX + 1):
            be = min(b, bf); Tb = T2[:, :be]
            ssm = float(np.sum((Tb @ Q2[:, :be].T) ** 2))
            x2_b, gX2 = _resid(Tb, x2_a)
            Ytr_b, gY2 = _resid(Tb, Ytr_a)
            p3, T3, Q3, cf = _fit(x2_b, Ytr_b, C_MAX)
            for c in range(1, C_MAX + 1):
                ce = min(c, cf); Tc = T3[:, :ce]
                ssx = float(np.sum((Tc @ Q3[:, :ce].T) ** 2))
                ssf = sst - ssx1 - ssm - ssx
                R2[a-1, b-1, c-1] = 1 - ssf / sst
                SSX1[a-1, b-1, c-1] = ssx1; SSM[a-1, b-1, c-1] = ssm
                SSX[a-1, b-1, c-1] = ssx; SSF[a-1, b-1, c-1] = ssf
    M2M = SSM / np.maximum(SSX, 1e-9)
    return R2, SSX1, SSM, SSX, SSF, M2M


# ---------------------------------------------------------------------------
# Final model at a chosen (a,b,c): fit on all calibration, predict new data
# ---------------------------------------------------------------------------
def fit_final(X1, M2, X2, Y, a, b, c):
    muX1, sdX1, k1 = _scale_fit(X1); x1s = _scale_apply(X1, muX1, sdX1, k1)
    muM2, sdM2, k2 = _scale_fit(M2); m2s = _scale_apply(M2, muM2, sdM2, k2)
    muX2, sdX2, k3 = _scale_fit(X2); x2s = _scale_apply(X2, muX2, sdX2, k3)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY) / sdy

    p1, T1, Q1, af = _fit(x1s, ys, a); ae = min(a, af); T1 = T1[:, :ae]; Q1 = Q1[:, :ae]
    m2r, gM = _resid(T1, m2s)
    x2_a, gX = _resid(T1, x2s)
    Ytr_a, gY = _resid(T1, ys)
    p2, T2, Q2, bf = _fit(m2r, Ytr_a, b); be = min(b, bf); T2 = T2[:, :be]; Q2 = Q2[:, :be]
    x2_b, gX2 = _resid(T2, x2_a)
    Ytr_b, gY2 = _resid(T2, Ytr_a)
    p3, T3, Q3, cf = _fit(x2_b, Ytr_b, c); ce = min(c, cf); T3 = T3[:, :ce]; Q3 = Q3[:, :ce]

    Yhat_cal = (T1 @ Q1.T + T2 @ Q2.T + T3 @ Q3.T) * sdy + muY

    model = dict(muX1=muX1, sdX1=sdX1, k1=k1, muM2=muM2, sdM2=sdM2, k2=k2, muX2=muX2, sdX2=sdX2, k3=k3,
                 muY=muY, sdy=sdy, p1=p1, ae=ae, Q1=Q1, gM=gM, gX=gX, p2=p2, be=be, Q2=Q2, gX2=gX2, p3=p3, ce=ce, Q3=Q3)
    return model, Yhat_cal


def predict_final(model, X1n, M2n, X2n):
    x1s = _scale_apply(X1n, model['muX1'], model['sdX1'], model['k1'])
    m2s = _scale_apply(M2n, model['muM2'], model['sdM2'], model['k2'])
    x2s = _scale_apply(X2n, model['muX2'], model['sdX2'], model['k3'])
    T1n = model['p1'].transform(x1s)[:, :model['ae']]
    m2n_r = m2s - T1n @ model['gM']
    x2n_a = x2s - T1n @ model['gX']
    T2n = model['p2'].transform(m2n_r)[:, :model['be']]
    x2n_b = x2n_a - T2n @ model['gX2']
    T3n = model['p3'].transform(x2n_b)[:, :model['ce']]
    Yhat = (T1n @ model['Q1'].T + T2n @ model['Q2'].T + T3n @ model['Q3'].T) * model['sdy'] + model['muY']
    return Yhat


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2)))


def r2(y_true, y_pred):
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - np.sum((y_true.ravel() - y_pred.ravel()) ** 2) / sst)


# ---------------------------------------------------------------------------
# Run for all 7 models
# ---------------------------------------------------------------------------
full_rows = []
summary_rows = []
failures = []

for name in MODEL_NAMES:
    print(f"\n=== {name} ===")
    try:
        M2c, M2w, M2e = load_M2(name)
        if np.isnan(M2c).any() or np.isnan(M2w).any() or np.isnan(M2e).any():
            raise ValueError("NaN detected in M2 block before fitting")

        rmsecv, qabc = cv_surface(X1c, M2c, X2c, Y_cal)
        R2s, SSX1s, SSMs, SSXs, SSFs, M2Ms = full_fit_surface(X1c, M2c, X2c, Y_cal)

        for a in range(A_MAX):
            for b in range(B_MAX):
                for c in range(C_MAX):
                    full_rows.append(dict(model=name, LV_X1=a+1, LV_M2=b+1, LV_X2=c+1,
                                           RMSECV=rmsecv[a,b,c], Q2=qabc[a,b,c], R2=R2s[a,b,c],
                                           SS_X1=SSX1s[a,b,c], SS_M=SSMs[a,b,c], SS_X=SSXs[a,b,c],
                                           SS_F=SSFs[a,b,c], M2M=M2Ms[a,b,c]))

        i = np.unravel_index(rmsecv.argmin(), rmsecv.shape)
        best = dict(LV_X1=i[0]+1, LV_M2=i[1]+1, LV_X2=i[2]+1)
        print(f"  best LV config (min RMSECV): {best}  RMSECV={rmsecv[i]:.3f}  Q2={qabc[i]:.3f}  "
              f"R2={R2s[i]:.3f}  M2M={M2Ms[i]:.3f}")

        model, Yhat_cal = fit_final(X1c, M2c, X2c, Y_cal, best['LV_X1'], best['LV_M2'], best['LV_X2'])
        Yhat_within = predict_final(model, X1w, M2w, X2w)
        Yhat_extrap = predict_final(model, X1e, M2e, X2e)

        rmse_cal, r2_cal = rmse(Y_cal, Yhat_cal), r2(Y_cal, Yhat_cal)
        rmse_within, r2_within = rmse(Y_within, Yhat_within), r2(Y_within, Yhat_within)
        rmse_extrap, r2_extrap = rmse(Y_extrap, Yhat_extrap), r2(Y_extrap, Yhat_extrap)

        print(f"  RMSE  calibration={rmse_cal:.3f} (R2={r2_cal:.3f})  "
              f"within={rmse_within:.3f} (R2={r2_within:.3f})  extrapolation={rmse_extrap:.3f} (R2={r2_extrap:.3f})")

        summary_rows.append(dict(model=name, LV_X1=best['LV_X1'], LV_M2=best['LV_M2'], LV_X2=best['LV_X2'],
                                  RMSECV=rmsecv[i], Q2=qabc[i], R2_calibration=r2_cal, M2M=M2Ms[i],
                                  RMSE_calibration=rmse_cal, RMSE_within_domain=rmse_within,
                                  RMSE_extrapolation=rmse_extrap, R2_within_domain=r2_within,
                                  R2_extrapolation=r2_extrap))
    except Exception as e:
        print(f"  *** FAILURE for {name}: {type(e).__name__}: {e}")
        failures.append(dict(model=name, error=f"{type(e).__name__}: {e}"))

full_df = pd.DataFrame(full_rows)
summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter('SO_PLS_results.xlsx') as writer:
    full_df.to_excel(writer, sheet_name='full_grid', index=False)
    summary_df.to_excel(writer, sheet_name='summary_best_config', index=False)
    if failures:
        pd.DataFrame(failures).to_excel(writer, sheet_name='failures', index=False)

print(f"\nSaved SO_PLS_results.xlsx  (full_grid: {len(full_df)} rows, summary_best_config: {len(summary_df)} rows)")
if failures:
    print(f"FAILURES: {failures}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(summary_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    xw = np.arange(len(summary_df))
    w = 0.25
    ax.bar(xw - w, summary_df['RMSE_calibration'], width=w, label='calibration')
    ax.bar(xw, summary_df['RMSE_within_domain'], width=w, label='within-domain (T2<=370)')
    ax.bar(xw + w, summary_df['RMSE_extrapolation'], width=w, label='extrapolation (T2>370)')
    ax.set_xticks(xw); ax.set_xticklabels(summary_df['model'], rotation=30, ha='right')
    ax.set_ylabel('RMSE (E purity, %)'); ax.set_title('SO-PLS prediction error by model and domain')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('SO_PLS_extrapolation_performance.png', dpi=140, facecolor='white')
    print("Saved SO_PLS_extrapolation_performance.png")

if len(full_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in MODEL_NAMES:
        sub = full_df[full_df['model'] == name]
        if len(sub):
            ax.scatter(sub['RMSECV'], sub['M2M'], s=6, alpha=0.35, label=name)
    ax.set_xlabel('RMSECV'); ax.set_ylabel('M2M = SS_M / SS_X')
    ax.set_yscale('log')
    ax.set_title('M2M vs RMSECV across the full LV grid (all 7 models)')
    ax.legend(markerscale=3, fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('SO_PLS_M2M_vs_RMSECV.png', dpi=140, facecolor='white')
    print("Saved SO_PLS_M2M_vs_RMSECV.png")

print("\n===== COMPACT SUMMARY TABLE =====")
if len(summary_df):
    print(summary_df[['model','LV_X1','LV_M2','LV_X2','RMSECV','Q2','RMSE_within_domain','RMSE_extrapolation','M2M']].to_string(index=False))

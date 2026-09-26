"""
Resolve the MAR LV-selection ambiguity: evaluate M2-only at the SAME LV_M2
as each FULL-model configuration (never letting M2-only pick its own
optimum). Calibration-only, reuses existing full_grid (FULL model RMSECV,
already computed) and recomputes only the M2-only 1-block CV surface
(same _fit/_scale/10-fold/seed=42 convention as every prior script).
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

B_MAX = 15
N_SPLITS, SEED = 10, 42
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

Y_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='Y')['E_purity'].values.reshape(-1, 1)
X2_cal_cols = list(pd.read_excel('DG_calibration.xlsx', sheet_name='X2').columns)

def load_M2_cal(name):
    return pd.read_excel(f"M2_{name}.xlsx", sheet_name='calibration')[X2_cal_cols].values

full_grid = pd.read_excel('LV_allocation_analysis.xlsx', sheet_name='full_grid')

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
    return rf.mean(0)

primary_rows = []
flat_rows = []
all_configs = []

for name in MODEL_NAMES:
    M2c = load_M2_cal(name)
    rmsecv_m2o = cv_1block(M2c, Y_cal, B_MAX)  # index b-1 -> LV_M2=b

    sub = full_grid[full_grid['model'] == name].copy()
    sub['RMSECV_M2_ONLY_sameLV'] = sub['LV_M2'].apply(lambda b: rmsecv_m2o[int(b)-1])
    sub['MAR_sameLV'] = sub['RMSECV'] / sub['RMSECV_M2_ONLY_sameLV']
    all_configs.append(sub[['model','LV_X1','LV_M2','LV_X2','RMSECV','RMSECV_M2_ONLY_sameLV','MAR_sameLV']])

    i_min = sub['RMSECV'].idxmin(); r = sub.loc[i_min]
    primary_rows.append(dict(model=name, LV_X1=int(r['LV_X1']), LV_M2=int(r['LV_M2']), LV_X2=int(r['LV_X2']),
                              RMSECV_FULL=r['RMSECV'], RMSECV_M2_ONLY_sameLV=r['RMSECV_M2_ONLY_sameLV'],
                              MAR_sameLV=r['MAR_sameLV']))

    rmin = sub['RMSECV'].min()
    flat = sub[sub['RMSECV'] <= rmin*1.05]
    v = flat['MAR_sameLV'].values
    flat_rows.append(dict(model=name, n_configs=len(flat), median=np.median(v), P10=np.percentile(v,10),
                           P90=np.percentile(v,90), min=v.min(), max=v.max(), sd=v.std(ddof=1)))

primary_df = pd.DataFrame(primary_rows)
flat_df = pd.DataFrame(flat_rows)
full_configs_df = pd.concat(all_configs, ignore_index=True)

print("===== PRIMARY (FULL-model min-RMSECV configuration) =====")
print(primary_df.to_string(index=False))

print("\n===== MAR_sameLV distribution within dRMSECV<=5% region =====")
print(flat_df.to_string(index=False))

with pd.ExcelWriter('MAR_sameLV_analysis.xlsx') as w:
    primary_df.to_excel(w, sheet_name='primary_config', index=False)
    flat_df.to_excel(w, sheet_name='flat_region_5pct', index=False)
    full_configs_df.to_excel(w, sheet_name='full_grid_MAR_sameLV', index=False)
print("\nSaved MAR_sameLV_analysis.xlsx")

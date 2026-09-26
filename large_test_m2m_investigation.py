"""
Step 2-10 of the M2M robustness investigation.

STEP 2: generate a large, independent test set (100 within-domain T2~U(362,370)
+ 100 extrapolation T2~U(370,382)) using the SAME SDE generator core and
corrected sigma_A=0.25 as before. New seed (2024). Calibration data and
Data Generation.py are NOT touched.

STEP 3: batches are discretized directly onto the SAME fixed 7-point grid
used by calibration (t=0,15,...,90s) -- no truncation logic needed, no t=105
point is ever produced for these large test sets (t2_seconds ~ U(100,400)s
guarantees every batch covers at least 90s).

STEP 4: M2 for the large test sets, all 7 models, exact same rhs + parameter
conventions as build_M2_blocks.py (M0-M5 fixed nominal params, M6 refit via
curve_fit on the 100 calibration batches ONLY, then frozen).

STEP 5+: extend the EXISTING SO-PLS full LV grid (same A_MAX/B_MAX/C_MAX,
same _fit/_scale/orthogonalization code from so_pls_experiment.py) to also
predict the two new large test sets at every (a,b,c), then run the
distribution / flat-region / M2M-vs-quality analysis requested.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import skew
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']
A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED_CV = 10, 42
MODEL_NAMES = ['M0_correct', 'M1_no_side', 'M2_order1_D', 'M3_wrong_Ea4',
               'M4_lumped_EF', 'M5_no_D', 'M6_author_mis']

# =====================================================================
# STEP 2: large independent test set (SAME generator core, corrected sigma_A)
# =====================================================================
V1, V2 = 0.1, 0.2
CA0, CD0 = 2000.0, 1900.0
A1, A2 = 28, 40
A3_true, A4_true = 10, 20
Ea1_mean, Ea2_mean, Ea3_mean, Ea4_mean = 20000, 30000, 50000, 55000
CV = 0.01
opt_sigma1 = [0.025, 0.9, 0.08]
opt_sigma2 = [0.25, 0.03, 0.022, 0.07, 0.54, 0.04]  # corrected sigma_A

import sdeint

def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    return np.array([-k1 * CA, k1 * CA - k2 * CB, k2 * CB])

def stage1_noise(y, t):
    return np.diag(opt_sigma1)

def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    return np.array([0, -k3*CB*CD**2, -k4*CC*CD**2, -2*k3*CB*CD**2-2*k4*CC*CD**2, k3*CB*CD**2, k4*CC*CD**2])

def stage2_noise(y, t):
    return np.diag(opt_sigma2)

def one_sde_simulation_custom(T2_lo, T2_hi):
    Ea1 = np.random.normal(Ea1_mean, CV*Ea1_mean); Ea2 = np.random.normal(Ea2_mean, CV*Ea2_mean)
    Ea3 = np.random.normal(Ea3_mean, CV*Ea3_mean); Ea4 = np.random.normal(Ea4_mean, CV*Ea4_mean)
    T1 = np.random.uniform(290, 310); T2 = np.random.uniform(T2_lo, T2_hi)
    t1_seconds = np.random.uniform(350, 650); t2_seconds = np.random.uniform(100, 400)
    k1 = A1*np.exp(-Ea1/(R*T1)); k2 = A2*np.exp(-Ea2/(R*T1))
    k3 = A3_true*np.exp(-Ea3/(R*T2)); k4 = A4_true*np.exp(-Ea4/(R*T2))
    CA0_sample = np.random.normal(CA0, 0.05*CA0); CD0_sample = np.random.normal(CD0, 0.05*CD0)
    t1v = np.linspace(0, t1_seconds, 1000)
    sol1 = sdeint.itoint(lambda y,t: stage1_drift(y,t,k1,k2), lambda y,t: stage1_noise(y,t), np.array([CA0_sample,0.,0.]), t1v)
    t2v = np.linspace(0, t2_seconds, 1000)
    CA1f, CB1f, CC1f = sol1[-1]; dilution = V1/V2
    y0_2 = np.array([CA1f*dilution, CB1f*dilution, CC1f*dilution, CD0_sample, 0., 0.])
    sol2 = sdeint.itoint(lambda y,t: stage2_drift(y,t,k3,k4), lambda y,t: stage2_noise(y,t), y0_2, t2v)
    return t1v/60, sol1, t2v/60, sol2, dict(CA0_sample=CA0_sample, CD0_sample=CD0_sample, T1=T1, T2=T2,
                                             t1_seconds=t1_seconds, t2_seconds=t2_seconds)

TARGET_T_GRID_MIN = np.arange(7) * 0.25  # 0,15,...,90 s in minutes -- SAME grid as calibration

def gen_large_split(n, T2_lo, T2_hi, label=''):
    rows1, rows2_raw, params_list = [], [], []
    resampled = []
    for i in range(n):
        t1v, sol1, t2v, sol2, p = one_sde_simulation_custom(T2_lo, T2_hi)
        retries = 0
        while not (np.isfinite(sol1).all() and np.isfinite(sol2).all()):
            # sdeint.itoint's fixed-step Euler-Maruyama occasionally diverges
            # (rare, ~1%, seen at long stage-2 durations). Redraw this batch's
            # random conditions/trajectory only -- documented, not silent.
            retries += 1
            t1v, sol1, t2v, sol2, p = one_sde_simulation_custom(T2_lo, T2_hi)
            if retries > 20:
                raise RuntimeError(f"{label} batch {i}: failed to obtain a finite SDE trajectory after 20 resamples")
        if retries:
            resampled.append((i, retries, p['T2'], p['t2_seconds']))
        rows1.append((t1v, sol1)); rows2_raw.append((t2v, sol2)); params_list.append(p)
    if resampled:
        print(f"  [{label}] resampled {len(resampled)} batch(es) that diverged on first draw:")
        for i, retries, T2v, t2s in resampled:
            print(f"    batch {i}: {retries} retries, first-draw T2={T2v:.2f}K t2_seconds={t2s:.1f}s")
    # measurement noise (same convention: pct of mean final concentration)
    final_c2 = np.array([sol2[-1] for _, sol2 in rows2_raw])
    mean2 = final_c2.mean(axis=0); noise2 = mean2 * 0.0035
    final_c1 = np.array([sol1[-1] for _, sol1 in rows1])
    mean1 = final_c1.mean(axis=0); noise1 = mean1 * 0.0035

    X1_rows, X2_rows, Y_rows = [], [], []
    for i in range(n):
        t1v, sol1 = rows1[i]; t2v, sol2 = rows2_raw[i]; p = params_list[i]
        CAf1 = np.maximum(0, np.interp(t1v[-1], t1v, sol1[:,0]) + np.random.normal(0, noise1[0]))
        CBf1 = np.maximum(0, np.interp(t1v[-1], t1v, sol1[:,1]) + np.random.normal(0, noise1[1]))
        CCf1 = np.maximum(0, np.interp(t1v[-1], t1v, sol1[:,2]) + np.random.normal(0, noise1[2]))
        X1_rows.append([p['CA0_sample'], p['t1_seconds']/60, p['T1'], CAf1, CBf1, CCf1])

        row = [p['CD0_sample'], p['t2_seconds']/60, p['T2']]
        block = np.zeros((7, 6))
        for j in range(6):
            interp = np.interp(TARGET_T_GRID_MIN, t2v, sol2[:, j])
            block[:, j] = np.maximum(0, interp + np.random.normal(0, noise2[j], 7))
        for j in range(6):
            row += block[:, j].tolist()
        X2_rows.append(row)

        CAf, CBf, CCf, CDf, CEf, CFf = block[-1]
        Y_rows.append(CEf/(CAf+CBf+CCf+CDf+CEf+CFf)*100)

    X1_df = pd.DataFrame(X1_rows, columns=["CA0","t1","T1","CA1_final","CB1_final","CC1_final"])
    cols = ['CD0','t2_final','T2']
    for c in species:
        for t in range(7):
            cols.append(f"{c}_t{t+1}")
    X2_df = pd.DataFrame(X2_rows, columns=cols)
    Y_df = pd.DataFrame(Y_rows, columns=["E_purity"])
    return X1_df, X2_df, Y_df, [p['T2'] for p in params_list]


print("STEP 2: generating large independent test set (100 within T2~U(362,370), 100 extrap T2~U(370,382))...")
np.random.seed(2024)
X1_w_df, X2_w_df, Y_w_df, T2_w = gen_large_split(100, 362, 370, label='within')
X1_e_df, X2_e_df, Y_e_df, T2_e = gen_large_split(100, 370, 382, label='extrap')

with pd.ExcelWriter('DG_test_within_domain_large.xlsx') as w:
    X1_w_df.to_excel(w, sheet_name='X1', index=False); X2_w_df.to_excel(w, sheet_name='X2', index=False); Y_w_df.to_excel(w, sheet_name='Y', index=False)
with pd.ExcelWriter('DG_test_extrapolation_large.xlsx') as w:
    X1_e_df.to_excel(w, sheet_name='X1', index=False); X2_e_df.to_excel(w, sheet_name='X2', index=False); Y_e_df.to_excel(w, sheet_name='Y', index=False)

print(f"  within  T2 range=[{min(T2_w):.2f},{max(T2_w):.2f}]  n={len(T2_w)}  any NaN X2={X2_w_df.isna().any().any()}")
print(f"  extrap  T2 range=[{min(T2_e):.2f},{max(T2_e):.2f}]  n={len(T2_e)}  any NaN X2={X2_e_df.isna().any().any()}")
print("  Saved DG_test_within_domain_large.xlsx, DG_test_extrapolation_large.xlsx")

# =====================================================================
# STEP 4: M2 for the large test sets, all 7 models (same rhs as build_M2_blocks.py)
# =====================================================================
def m_correct(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea4/(R*T2v))
    return [0,-k3*CB*CD**2,-k4*CC*CD**2,-2*k3*CB*CD**2-2*k4*CC*CD**2,k3*CB*CD**2,k4*CC*CD**2]
def m_no_side(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v))
    return [0,-k3*CB*CD**2,0,-2*k3*CB*CD**2,k3*CB*CD**2,0]
def m_order1_D(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea4/(R*T2v))
    return [0,-k3*CB*CD,-k4*CC*CD,-2*k3*CB*CD-2*k4*CC*CD,k3*CB*CD,k4*CC*CD]
def m_wrong_Ea4(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea3/(R*T2v))
    return [0,-k3*CB*CD**2,-k4*CC*CD**2,-2*k3*CB*CD**2-2*k4*CC*CD**2,k3*CB*CD**2,k4*CC*CD**2]
def m_lumped(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v)); rate=k3*(CB+CC)*CD**2
    return [0,-k3*CB*CD**2,-k3*CC*CD**2,-2*rate,k3*CB*CD**2,k3*CC*CD**2]
def m_no_D(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v))*1e6; k4=A4*np.exp(-Ea4/(R*T2v))*1e6
    return [0,-k3*CB,-k4*CC,-2*k3*CB-2*k4*CC,k3*CB,k4*CC]
def m_author_mis(y,t,T2v,A3,A4,Ea3,Ea4):
    CA,CB,CC,CD,CE,CF=y; k3=A3*np.exp(-Ea3/(R*T2v))
    return [0,-k3*CB*CD,0,-2*k3*CB*CD,k3*CB*CD,0]

MODELS = {
    'M0_correct':(m_correct,False), 'M1_no_side':(m_no_side,False), 'M2_order1_D':(m_order1_D,False),
    'M3_wrong_Ea4':(m_wrong_Ea4,False), 'M4_lumped_EF':(m_lumped,False), 'M5_no_D':(m_no_D,False),
    'M6_author_mis':(m_author_mis,True),
}
TRUE_PARAMS = (A3_true, A4_true, Ea3_mean, Ea4_mean)
TGRID_SEC = np.arange(7) * 15.0

X2_cal = pd.read_excel('DG_calibration.xlsx', sheet_name='X2')

def get_IC_T2(df, row):
    ic = np.array([df.loc[row, f'{s}_t1'] for s in species], dtype=float)
    return ic, float(df.loc[row, 'T2'])

def fit_M6_params():
    df = X2_cal; n = len(df)
    ics = np.array([get_IC_T2(df, i)[0] for i in range(n)]); T2s = np.array([get_IC_T2(df, i)[1] for i in range(n)])
    y_obs = np.array([[df.loc[i, f'{s}_t{k+1}'] for s in species for k in range(7)] for i in range(n)]).ravel()
    def model_flat(_, A3, A4, Ea3, Ea4):
        preds = np.zeros((n, 6, 7))
        for i in range(n):
            preds[i] = odeint(m_author_mis, ics[i], TGRID_SEC, args=(T2s[i], A3, A4, Ea3, Ea4)).T
        return preds.reshape(n, -1).ravel()
    fitted, _ = curve_fit(model_flat, np.zeros_like(y_obs), y_obs, p0=list(TRUE_PARAMS), maxfev=10000,
                           bounds=([0,0,0,0],[100,100,np.inf,np.inf]))
    return tuple(fitted)

print("\nSTEP 4: fitting M6 params on calibration only (reused, not refit on test)...")
M6_params = fit_M6_params()
print(f"  M6 [A3,A4,Ea3,Ea4] = {np.round(M6_params,4)}")

def simulate_M2_for_df(df, rhs, params):
    n = len(df)
    out = np.zeros((n, 6, 7))
    for i in range(n):
        ic, T2v = get_IC_T2(df, i)
        out[i] = odeint(rhs, ic, TGRID_SEC, args=(T2v,) + params).T
    return out

def to_df(arr, df_src):
    cols = {'CD0': df_src['CD0'].values, 't2_final': df_src['t2_final'].values, 'T2': df_src['T2'].values}
    for j, s in enumerate(species):
        for k in range(7):
            cols[f'{s}_t{k+1}'] = arr[:, j, k]
    return pd.DataFrame(cols)

M2_large = {}  # (split,model) -> df
for split_name, df in [('within_large', X2_w_df), ('extrap_large', X2_e_df)]:
    for model_name, (rhs, uses_fit) in MODELS.items():
        params = M6_params if uses_fit else TRUE_PARAMS
        arr = simulate_M2_for_df(df, rhs, params)
        M2_large[(split_name, model_name)] = to_df(arr, df)

for model_name in MODELS:
    with pd.ExcelWriter(f"M2_{model_name}_large.xlsx") as w:
        M2_large[('within_large', model_name)].to_excel(w, sheet_name='within_large', index=False)
        M2_large[('extrap_large', model_name)].to_excel(w, sheet_name='extrap_large', index=False)
print("  Saved M2_<model>_large.xlsx for all 7 models")

any_nan = any(M2_large[k].isna().any().any() for k in M2_large)
any_huge = any((M2_large[k].abs() > 1e6).any().any() for k in M2_large)
print(f"  Sanity: any NaN in large M2 blocks={any_nan}  any huge (>1e6)={any_huge}")

print("\nSTEP 2-4 DONE.\n")

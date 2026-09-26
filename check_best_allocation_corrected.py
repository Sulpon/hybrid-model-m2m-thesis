"""
With the corrected generator now validated against the real Excel files,
regenerate case (i) and case (ii) data again, build the M2 blocks (case i:
true-parameter re-integration, no fitting, matching the paper's own
"well-specified" convention; case ii: curve_fit misspecified fit, matching
the notebook), and run the SAME validated blind-argmin SO-PLS CV search
(cv_surface, unmodified from verify_paper_LV.py) to find each case's own
best LV allocation. Compare against the paper's {4,6,2}/{2,3,6} and the
user's own found {4,6,4}/{3,3,6}.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from Data_Generation_corrected import (stage1_drift, stage1_noise, stage2_drift, stage2_noise,
                                        one_sde_simulation, discrete_trajectories, R,
                                        Ea3_mean, Ea4_mean, A3, A4)

N = 100
A_MAX, B_MAX, C_MAX = 6, 15, 15
N_SPLITS, SEED = 10, 42
species = ['A', 'B', 'C', 'D', 'E', 'F']

# =====================================================================
# Regenerate case (i) and case (ii), same as compare_corrected_vs_excel.py
# =====================================================================
def gen_case(seed):
    np.random.seed(seed)
    all_stage1, all_stage2, all_params = [], [], []
    for i in range(N):
        (t1v, sol1), (t2v, sol2), params = one_sde_simulation(data_type="training")
        all_stage1.append((t1v, sol1)); all_stage2.append((t2v, sol2)); all_params.append(params)
    disc1, disc2 = discrete_trajectories(all_stage1, all_stage2)
    min_pts = min(len(d[0]) for d in disc2)
    return disc1, disc2, all_params, min_pts

print("Regenerating case (i) [seed=10] and case (ii) [seed=11]...")
disc1_i, disc2_i, params_i, min_pts_i = gen_case(10)
disc1_ii, disc2_ii, params_ii, min_pts_ii = gen_case(11)
print(f"  min_pts: case i={min_pts_i}, case ii={min_pts_ii}")

def build_X1(disc1, params):
    rows = [[params[i]['CA0_sample'], params[i]['T1'], params[i]['t1_seconds']/60,
             *disc1[i][1][-1]] for i in range(N)]
    return np.array(rows)

def build_X2(disc2, min_pts):
    return np.array([disc2[i][1][:min_pts].T.flatten() for i in range(N)]).reshape(N, 6, min_pts)

def build_Y(disc2, min_pts):
    # BUG FIX: Y must be each batch's own TRUE final point (its own t2_seconds,
    # up to 400s), NOT the shared-truncated min_pts grid point (~90s common to
    # all batches). Verified against real Y.xlsx: recomputing Y from X2's own
    # truncated last column gives mean abs diff 4.79pp vs the real Y.xlsx,
    # corr=0.85 -- Y is measured/computed at each batch's own actual endpoint.
    final = np.array([disc2[i][1][-1] for i in range(N)])
    return (final[:, 4] / final.sum(axis=1) * 100).reshape(-1, 1)

X1_i = build_X1(disc1_i, params_i)
X1_ii = build_X1(disc1_ii, params_ii)
Y_i = build_Y(disc2_i, min_pts_i)
Y_ii = build_Y(disc2_ii, min_pts_ii)

# X2 as (N, 6*min_pts) flattened, species-major (matches CAL_X2_COLS convention elsewhere)
def flatten_X2(disc2, min_pts):
    return np.array([disc2[i][1][:min_pts, :].T.flatten() for i in range(N)])

X2_i = flatten_X2(disc2_i, min_pts_i)
X2_ii = flatten_X2(disc2_ii, min_pts_ii)

# =====================================================================
# M2 for case (i): TRUE parameters, no fitting (paper's well-specified convention)
# =====================================================================
def stage2_ode_correct(y, t, T2, Ea3v, Ea4v, A3v, A4v):
    CA, CB, CC, CD, CE, CF = y
    k3 = A3v * np.exp(-Ea3v / (R * T2)); k4 = A4v * np.exp(-Ea4v / (R * T2))
    return [0, -k3*CB*CD**2, -k4*CC*CD**2, -2*k3*CB*CD**2-2*k4*CC*CD**2, k3*CB*CD**2, k4*CC*CD**2]

tgrid_i = np.arange(min_pts_i) * 15.0
M2_i = np.zeros((N, 6, min_pts_i))
for i in range(N):
    ic = disc2_i[i][1][0]
    sol = odeint(stage2_ode_correct, ic, tgrid_i, args=(params_i[i]['T2'], Ea3_mean, Ea4_mean, A3, A4))
    M2_i[i] = sol.T
M2_i = M2_i.reshape(N, -1)
# append t2 (minutes), T2, D0 -- matching M2.xlsx's real column structure exactly
# (paper: "any measured inputs to the KD model present in X2 are moved into M2")
meta_i = np.array([[params_i[i]['t2_seconds']/60, params_i[i]['T2'], params_i[i]['CD0_sample']] for i in range(N)])
M2_i = np.hstack([M2_i, meta_i])

# =====================================================================
# M2 for case (ii): curve_fit misspecified (same as compare_corrected_vs_excel.py)
# =====================================================================
def stage2_ode_mis(y, t, T2, Ea3v, Ea4v, A3f, A4f):
    CA, CB, CC, CD, CE, CF = y
    k3 = A3f * np.exp(-Ea3v / (R * T2))
    return [0, -k3*CB*CD, 0, -2*k3*CB*CD, k3*CB*CD, 0]

tgrid_ii = np.arange(min_pts_ii) * 15.0
data_dict_list = [dict(T2=params_ii[i]['T2'], conc=disc2_ii[i][1][:min_pts_ii]) for i in range(N)]

def model_flat(_, Ea3v, Ea4v, A3f, A4f):
    preds = []
    for d in data_dict_list:
        sol = odeint(stage2_ode_mis, d['conc'][0], tgrid_ii, args=(d['T2'], Ea3v, Ea4v, A3f, A4f))
        preds.append(sol.ravel())
    return np.concatenate(preds)

y_obs = np.concatenate([d['conc'].ravel() for d in data_dict_list])
fitted, _ = curve_fit(model_flat, np.zeros_like(y_obs), y_obs, p0=[50000, 55000, 10, 20],
                       maxfev=10000, bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, 100]))
print(f"  case (ii) fitted [Ea3,Ea4,A3,A4] = {np.round(fitted, 3)}")

M2_ii = np.zeros((N, 6, min_pts_ii))
for i in range(N):
    ic = disc2_ii[i][1][0]
    sol = odeint(stage2_ode_mis, ic, tgrid_ii, args=(params_ii[i]['T2'], *fitted))
    M2_ii[i] = sol.T
M2_ii = M2_ii.reshape(N, -1)
meta_ii = np.array([[params_ii[i]['t2_seconds']/60, params_ii[i]['T2'], params_ii[i]['CD0_sample']] for i in range(N)])
M2_ii = np.hstack([M2_ii, meta_ii])

# ---- sanity check: does the regenerated M2 actually match the real M2.xlsx/M2_mis.xlsx? ----
M2a = pd.read_excel('M2.xlsx'); M2m = pd.read_excel('M2_mis.xlsx')
n_pts_a = (M2a.shape[1] - 3) // 6
print(f"\n===== M2 sanity check: regenerated vs real M2.xlsx (case i), final timepoint =====")
for j, sp in enumerate(species):
    a_final = M2a[f'{sp}_fit_{n_pts_a}'].values
    g_final = M2_i[:, j*min_pts_i + (min_pts_i-1)]
    print(f"{sp}: author {a_final.mean():9.2f}+/-{a_final.std(ddof=1):6.2f}   gen {g_final.mean():9.2f}+/-{g_final.std(ddof=1):6.2f}")
n_pts_m = (M2m.shape[1] - 3) // 6
print(f"\n===== M2 sanity check: regenerated vs real M2_mis.xlsx (case ii), final timepoint =====")
for j, sp in enumerate(species):
    a_final = M2m[f'{sp}_fit_{n_pts_m}'].values
    g_final = M2_ii[:, j*min_pts_ii + (min_pts_ii-1)]
    print(f"{sp}: author {a_final.mean():9.2f}+/-{a_final.std(ddof=1):6.2f}   gen {g_final.mean():9.2f}+/-{g_final.std(ddof=1):6.2f}")

# =====================================================================
# Validated blind-argmin SO-PLS CV search (identical to verify_paper_LV.py)
# =====================================================================
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
    return rf.mean(0), qabc

print("\nRunning blind-argmin CV search, case (i)...")
rmsecv_i, qabc_i = cv_surface(X1_i, M2_i, X2_i, Y_i)
i_i = np.unravel_index(rmsecv_i.argmin(), rmsecv_i.shape)
best_i = (i_i[0]+1, i_i[1]+1, i_i[2]+1)
print(f"  case (i) best LV = {best_i}  RMSECV={rmsecv_i[i_i]:.4f}  Q2={qabc_i[i_i]:.4f}")

print("\nRunning blind-argmin CV search, case (ii)...")
rmsecv_ii, qabc_ii = cv_surface(X1_ii, M2_ii, X2_ii, Y_ii)
i_ii = np.unravel_index(rmsecv_ii.argmin(), rmsecv_ii.shape)
best_ii = (i_ii[0]+1, i_ii[1]+1, i_ii[2]+1)
print(f"  case (ii) best LV = {best_ii}  RMSECV={rmsecv_ii[i_ii]:.4f}  Q2={qabc_ii[i_ii]:.4f}")

print("\n===== COMPARISON =====")
for label, rmsecv, best, user_lv, paper_lv in [
    ('case (i)', rmsecv_i, best_i, (4,6,4), (4,6,2)),
    ('case (ii)', rmsecv_ii, best_ii, (3,3,6), (2,3,6)),
]:
    r_my = rmsecv[best[0]-1, best[1]-1, best[2]-1]
    r_user = rmsecv[user_lv[0]-1, user_lv[1]-1, user_lv[2]-1]
    r_paper = rmsecv[paper_lv[0]-1, paper_lv[1]-1, paper_lv[2]-1]
    print(f"{label}: corrected-data optimum={best} RMSECV={r_my:.4f}   "
          f"at user's LV{user_lv} RMSECV={r_user:.4f}   at paper's LV{paper_lv} RMSECV={r_paper:.4f}")

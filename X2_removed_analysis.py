"""
Quantifies exactly what the existing SMB-PLS/SO-PLS M2->X2 orthogonalization
removes from X2, at every LV of M2's stage, using the EXACT NIPALS loop from
sopls_r.py (not a batch shortcut). Calibration data only, all 7 models.

All SS here are RAW X-SPACE quantities (sum of squared entries of the scaled/
block-normalized X2 matrix) -- NOT Y-explained-variance (unlike SS_M/SS_X
from the SO-PLS decomposition elsewhere in this project). This is
deliberately described as "the portion of X2 removed by the M2-based
orthogonalization," not "shared information."

Algebraic fact used (proven last turn, re-stated here): because there is
only one block (X2) after M2, Xhat_k = ts @ pinv(ts'ts) @ ts' @ Xk is rank 1
and tT (the fused super-score) is exactly proportional to ts. This means the
ACTUAL deflation applied to X2 (tT @ pk', pk = Xk'@tT/(tT'tT)) is
algebraically IDENTICAL to Xhat_k -- verified below, not assumed -- so
tracking Xhat_k directly is exactly tracking what the algorithm removes.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']

def ode_correct(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_no_side(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD**2
    return [0, -r3, 0, -2*r3, r3, 0]
def ode_order1_D(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB*CD; r4 = k4*CC*CD
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_wrong_Ea4(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea3/(R*T2))
    r3 = k3*CB*CD**2; r4 = k4*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_lumped(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD**2; r4 = k3*CC*CD**2
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_no_D(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, CC, _, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))
    r3 = k3*CB; r4 = k4*CC
    return [0, -r3, -r4, -2*r3-2*r4, r3, r4]
def ode_author_mis(y, t, T2, Ea3, Ea4, A3, A4):
    _, CB, _, CD, _, _ = y
    k3 = A3*np.exp(-Ea3/(R*T2)); r3 = k3*CB*CD
    return [0, -r3, 0, -2*r3, r3, 0]

MODELS = {
    'M0_correct': ode_correct, 'M1_no_side': ode_no_side, 'M2_order1_D': ode_order1_D,
    'M3_wrong_Ea4': ode_wrong_Ea4, 'M4_lumped_EF': ode_lumped, 'M5_no_D': ode_no_D,
    'M6_author_mis': ode_author_mis,
}
LV = {
    'M0_correct': (4, 6, 2), 'M1_no_side': (1, 13, 4), 'M2_order1_D': (4, 6, 4),
    'M3_wrong_Ea4': (4, 6, 2), 'M4_lumped_EF': (4, 8, 2), 'M5_no_D': (3, 7, 4),
    'M6_author_mis': (3, 3, 6),
}
FITTED = {
    'M0_correct': [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side': [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D': [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4': [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF': [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D': [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}
# SS_M_forward / SS_X_forward / M2M from the established forward SO-PLS decomposition (Y-explained-variance units)
SOPLS_FWD = {
    'M0_correct': dict(SS_M=71.723, SS_X=5.706, M2M=12.569),
    'M1_no_side': dict(SS_M=73.721, SS_X=11.267, M2M=6.543),
    'M2_order1_D': dict(SS_M=71.965, SS_X=6.194, M2M=11.619),
    'M3_wrong_Ea4': dict(SS_M=71.665, SS_X=5.729, M2M=12.509),
    'M4_lumped_EF': dict(SS_M=72.051, SS_X=5.731, M2M=12.571),
    'M5_no_D': dict(SS_M=54.412, SS_X=22.965, M2M=2.369),
    'M6_author_mis': dict(SS_M=65.979, SS_X=12.366, M2M=5.335),
}

X1a = pd.read_excel('X1.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
X2a = pd.read_excel('X2.xlsx')
Ya = pd.read_excel('Y.xlsx')['E_pur'].values.reshape(-1, 1)
M2a_meta = pd.read_excel('M2.xlsx')[['t2', 'T2', 'D0']]
n_pts_a = X2a.shape[1] // 6

X1m = pd.read_excel('X1_mis.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
X2m = pd.read_excel('X2_mis.xlsx')
Ym = pd.read_excel('Y_mis.xlsx')['E_pur'].values.reshape(-1, 1)
M2m_meta = pd.read_excel('M2_mis.xlsx')[['t2', 'T2', 'D0']]
n_pts_m = X2m.shape[1] // 6

N = len(X1a)
tgrid_a = np.arange(n_pts_a) * 15.0
tgrid_m = np.arange(n_pts_m) * 15.0

def build_ic_and_X2(X2df, n_pts):
    IC = np.array([[X2df[f'{s}_1'].iloc[i] for s in species] for i in range(len(X2df))])
    X2flat = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in species for k in range(n_pts)] for i in range(len(X2df))])
    return IC, X2flat

IC_a, X2flat_a = build_ic_and_X2(X2a, n_pts_a)
IC_m, X2flat_m = build_ic_and_X2(X2m, n_pts_m)

def build_M2(rhs, IC, T2v, n_pts, tgrid, meta, params):
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        sol = odeint(rhs, IC[i], tgrid, args=(T2v[i], *params))
        M2[i] = sol.T
    return np.hstack([M2.reshape(N, -1), meta.values])

def fit_scaler(X):
    mean = X.mean(0); std = X.std(0, ddof=1); std[std == 0] = 1
    return mean, std
def apply_scaler(X, mean, std): return (X - mean) / std
def scale_full(X):
    m, s = fit_scaler(X); return apply_scaler(X, m, s)


def run_and_track_X2_removal(X1n, M2n, X2n, Y_s, a, b):
    """EXACT NIPALS loop (X1 stage then M2 stage), tracking Xhat_k and its SS
    at every LV of M2's stage, plus verifying tT@pk' == Xhat_k exactly."""
    X_current = [X1n.copy(), M2n.copy(), X2n.copy()]
    Y_current = Y_s.copy()

    # X1 stage (establishes X2's state entering M2's stage)
    for _a in range(a):
        u = Y_current.copy(); t_old = None
        for _ in range(500):
            Xs = X_current[0]
            ws = Xs.T @ u / (u.T @ u); ws = ws / np.linalg.norm(ws); ts = Xs @ ws
            block_scores = [ts]
            for k in range(1, 3):
                Xk = X_current[k]
                Xhat_k = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk
                wk = Xhat_k.T @ u / (u.T @ u); wk = wk / np.linalg.norm(wk); tk = Xhat_k @ wk
                block_scores.append(tk)
            T = np.hstack(block_scores)
            wT = T.T @ u / (u.T @ u); wT = wT / np.linalg.norm(wT); tT = T @ wT
            q = Y_current.T @ tT / (tT.T @ tT); u_new = Y_current @ q / (q.T @ q)
            if t_old is not None and np.linalg.norm(tT - t_old) < 1e-10: break
            t_old = tT.copy(); u = u_new.copy()
        Yhat_a = tT @ q.T
        for k in range(3):
            Xk = X_current[k]; pk = Xk.T @ tT / (tT.T @ tT); X_current[k] = Xk - tT @ pk.T
        Y_current = Y_current - Yhat_a

    SS_X2_at_M2_start = float(np.sum(X_current[2] ** 2))

    # M2 stage: track Xhat_k (removed-from-X2 increment) at every LV
    per_lv = []
    cumulative_removed = 0.0
    max_deflation_mismatch = 0.0
    for lv in range(b):
        u = Y_current.copy(); t_old = None
        for _ in range(500):
            Xs = X_current[1]
            ws = Xs.T @ u / (u.T @ u); ws = ws / np.linalg.norm(ws); ts = Xs @ ws
            Xk = X_current[2]
            Xhat_k = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk
            wk = Xhat_k.T @ u / (u.T @ u); wk = wk / np.linalg.norm(wk); tk = Xhat_k @ wk
            T = np.hstack([ts, tk])
            wT = T.T @ u / (u.T @ u); wT = wT / np.linalg.norm(wT); tT = T @ wT
            q = Y_current.T @ tT / (tT.T @ tT); u_new = Y_current @ q / (q.T @ q)
            if t_old is not None and np.linalg.norm(tT - t_old) < 1e-10: break
            t_old = tT.copy(); u = u_new.copy()

        # recompute the FINAL Xhat_k (using the converged ts) for the SS accounting
        Xk_before = X_current[2]
        Xhat_k_final = ts @ np.linalg.pinv(ts.T @ ts) @ ts.T @ Xk_before
        ss_before = float(np.sum(Xk_before ** 2))
        ss_removed_this_lv = float(np.sum(Xhat_k_final ** 2))

        Yhat_a = tT @ q.T
        for k in range(3):
            Xkk = X_current[k]; pk = Xkk.T @ tT / (tT.T @ tT)
            deflation = tT @ pk.T
            if k == 2:
                mismatch = float(np.max(np.abs(deflation - Xhat_k_final)))
                max_deflation_mismatch = max(max_deflation_mismatch, mismatch)
            X_current[k] = Xkk - deflation
        Y_current = Y_current - Yhat_a

        ss_after = float(np.sum(X_current[2] ** 2))
        cumulative_removed += ss_removed_this_lv
        per_lv.append(dict(LV=lv + 1, SS_before_this_LV=ss_before, SS_Xhat_k_this_LV=ss_removed_this_lv,
                            pct_of_current_X2_removed=ss_removed_this_lv / ss_before * 100,
                            SS_after_this_LV=ss_after, cumulative_removed=cumulative_removed,
                            cumulative_pct_of_original=cumulative_removed / SS_X2_at_M2_start * 100))

    SS_X2_remaining = float(np.sum(X_current[2] ** 2))
    return SS_X2_at_M2_start, cumulative_removed, SS_X2_remaining, per_lv, max_deflation_mismatch


summary_rows, per_lv_rows = [], []
for name, rhs in MODELS.items():
    print(f"=== {name} ===")
    a, b, c = LV[name]
    params = FITTED[name]
    if name == 'M6_author_mis':
        X1_, T2v_, n_pts_, tgrid_, meta_, Y_ = X1m, M2m_meta['T2'].values, n_pts_m, tgrid_m, M2m_meta, Ym
        X2flat_ = X2flat_m; IC_ = IC_m
    else:
        X1_, T2v_, n_pts_, tgrid_, meta_, Y_ = X1a, M2a_meta['T2'].values, n_pts_a, tgrid_a, M2a_meta, Ya
        X2flat_ = X2flat_a; IC_ = IC_a

    M2raw = build_M2(rhs, IC_, T2v_, n_pts_, tgrid_, meta_, params)
    X1s = scale_full(X1_); M2s = scale_full(M2raw); X2s = scale_full(X2flat_); Ys = scale_full(Y_)
    X1n = X1s / np.sqrt(X1s.shape[1]); M2n = M2s / np.sqrt(M2s.shape[1]); X2n = X2s / np.sqrt(X2s.shape[1])

    ss_orig, ss_removed, ss_remaining, per_lv, mismatch = run_and_track_X2_removal(X1n, M2n, X2n, Ys, a, b)
    recon_err = abs(ss_orig - (ss_removed + ss_remaining))

    print(f"  SS_X2_at_M2_start={ss_orig:.4f}  SS_X2_removed={ss_removed:.4f}  SS_X2_remaining={ss_remaining:.4f}")
    print(f"  reconstruction check |orig-(removed+remaining)|={recon_err:.2e}   "
          f"max |tT@pk' - Xhat_k| across all LVs={mismatch:.2e}")
    print(f"  Removed_X2_% = {ss_removed/ss_orig*100:.2f}%")

    for r in per_lv:
        r2 = dict(model=name, **r)
        per_lv_rows.append(r2)

    fwd = SOPLS_FWD[name]
    summary_rows.append(dict(model=name, LV_M2=b, SS_X2_original=ss_orig, SS_X2_removed=ss_removed,
                              SS_X2_remaining=ss_remaining, Removed_X2_pct=ss_removed/ss_orig*100,
                              reconstruction_check_abs_err=recon_err, max_deflation_vs_Xhatk_mismatch=mismatch,
                              SS_M_forward=fwd['SS_M'], SS_X_forward=fwd['SS_X'], M2M=fwd['M2M']))

summary_df = pd.DataFrame(summary_rows)
per_lv_df = pd.DataFrame(per_lv_rows)

print("\n===== SUMMARY =====")
print(summary_df.to_string(index=False))

print("\n===== Comparison: SS_X2_removed (X-space) vs SS_M_forward (Y-explained-variance) -- DIFFERENT UNITS =====")
print(summary_df[['model', 'Removed_X2_pct', 'SS_M_forward', 'SS_X_forward', 'M2M']].to_string(index=False))

with pd.ExcelWriter('X2_removed_analysis.xlsx') as w:
    summary_df.to_excel(w, sheet_name='summary', index=False)
    per_lv_df.to_excel(w, sheet_name='per_LV', index=False)
    summary_df[['model', 'SS_X2_original', 'SS_X2_removed', 'SS_X2_remaining', 'reconstruction_check_abs_err']].to_excel(w, sheet_name='reconstruction_check', index=False)
    summary_df[['model', 'Removed_X2_pct', 'SS_M_forward', 'SS_X_forward', 'M2M']].to_excel(w, sheet_name='comparison_M2M', index=False)
print("\nSaved X2_removed_analysis.xlsx")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
for name in MODELS:
    sub = per_lv_df[per_lv_df['model'] == name]
    ax.plot(sub['LV'], sub['cumulative_pct_of_original'], marker='o', label=name)
ax.set_xlabel('LV (M2 stage)'); ax.set_ylabel('Cumulative % of X2 (at M2-stage start) removed')
ax.set_title('Portion of X2 removed by M2-based orthogonalization, by LV')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('X2_removed_by_LV.png', dpi=140, facecolor='white')
print("Saved X2_removed_by_LV.png")

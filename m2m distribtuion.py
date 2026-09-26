"""
Full M2M analysis pipeline: flat-region distributions + shared-information
metric, for all 7 candidate models (M0 correct, M1-M5 synthetic
misspecifications, M6 the author's own published misspecification).

Data sources (important, established after debugging earlier tonight):
  - M0, M1-M5: built on the REAL X1.xlsx/X2.xlsx/Y.xlsx via gen_all.py,
    which re-integrates each candidate KD model's mechanism against the
    same real measured data. M0 uses the REAL M2.xlsx directly (gen_all.py's
    own re-integration of the correct model has solver-precision drift
    vs. the real file).
  - M6: uses its OWN native files (X1_mis.xlsx/X2_mis.xlsx/Y_mis.xlsx/
    M2_mis.xlsx) -- this is NOT the same measured data as M0-M5 (confirmed:
    same CA0/T1/t1, but different downstream reacted-species values), so it
    must not be evaluated against gen_all.py's X1/X2/Y.

Two orthogonalization orders are used:
  - STANDARD  (X1 -> M2 -> X2): gives the usual M2M ratio and its flat-region
    distribution (median, P10, P90, mean), plus X2's UNIQUE contribution
    (what's left for X2 after M2 already took first claim).
  - REVERSED  (X1 -> X2 -> M2): gives X2's STANDALONE ceiling (what X2 could
    explain on its own, with no competition from M2 at all).

Shared information = X2's standalone ceiling minus X2's unique contribution
= the portion of X2's explanatory power that M2 absorbs/reproduces.
    shared_frac = shared_information / X2_standalone_ceiling

Requires: X1.xlsx, X2.xlsx, M2.xlsx, Y.xlsx (correct case) and
          X1_mis.xlsx, X2_mis.xlsx, M2_mis.xlsx, Y_mis.xlsx (misspecified
          case, used only for M6) in the working directory, plus gen_all.py
          available to regenerate the M1-M5 mechanistic blocks.
"""
import pickle, numpy as np, pandas as pd, json, subprocess, sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

A_MAX, B_MAX, C_MAX = 6, 15, 15   # matches m2m_at_min.py's grid exactly


# ---------------------------------------------------------------------------
# Core PLS fitting helper (shared by both orthogonalization orders)
# ---------------------------------------------------------------------------
def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n


def _scale(B, tr, te):
    mu = B[tr].mean(0); sd = B[tr].std(0, ddof=1); sd[sd < 1e-12] = 1.0
    k = np.sqrt(B.shape[1])
    return (B[tr] - mu) / sd / k, (B[te] - mu) / sd / k


# ---------------------------------------------------------------------------
# STANDARD order: X1 -> M2 -> X2. Gives q_a, q_ab, q_abc and per-fold RMSE
# (used for the flat region / 1-SE rule) across the full (a,b,c) grid.
# ---------------------------------------------------------------------------
def surface_standard(X1, M2, X2, Y, n_splits=10, seed=42):
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(Y)
    P_a = np.zeros((n, A_MAX)); P_ab = np.zeros((n, A_MAX, B_MAX)); P_abc = np.zeros((n, A_MAX, B_MAX, C_MAX))
    rf = np.zeros((n_splits, A_MAX, B_MAX, C_MAX))
    for f, (tr, te) in enumerate(cv.split(Y)):
        x1t, x1e = _scale(X1, tr, te); m2t0, m2e0 = _scale(M2, tr, te); x2t0, x2e0 = _scale(X2, tr, te)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy
        p1, T1t, Q1, af = _fit(x1t, Ytr0, A_MAX); T1e = p1.transform(x1e)
        for a in range(1, A_MAX + 1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            P_a[te, a - 1] = (yh_a * sdy + muY).ravel()
            Tp = np.linalg.pinv(Tat.T @ Tat) @ Tat.T
            gM = Tp @ m2t0; gX = Tp @ x2t0; gY = Tp @ Ytr0
            m2t, m2e = m2t0 - Tat @ gM, m2e0 - Tae @ gM
            x2t_a, x2e_a = x2t0 - Tat @ gX, x2e0 - Tae @ gX
            Ytr_a = Ytr0 - Tat @ gY
            p2, T2t, Q2, bf = _fit(m2t, Ytr_a, B_MAX); T2e = p2.transform(m2e)
            for b in range(1, B_MAX + 1):
                be = min(b, bf); Tbt, Tbe = T2t[:, :be], T2e[:, :be]
                yh_ab = yh_a + Tbe @ Q2[:, :be].T
                P_ab[te, a - 1, b - 1] = (yh_ab * sdy + muY).ravel()
                Tp2 = np.linalg.pinv(Tbt.T @ Tbt) @ Tbt.T
                gX2 = Tp2 @ x2t_a; gY2 = Tp2 @ Ytr_a
                x2t_b, x2e_b = x2t_a - Tbt @ gX2, x2e_a - Tbe @ gX2
                Ytr_b = Ytr_a - Tbt @ gY2
                p3, T3t, Q3, cf = _fit(x2t_b, Ytr_b, C_MAX); T3e = p3.transform(x2e_b)
                for c in range(1, C_MAX + 1):
                    ce = min(c, cf)
                    yh = (yh_ab + T3e[:, :ce] @ Q3[:, :ce].T) * sdy + muY
                    P_abc[te, a - 1, b - 1, c - 1] = yh.ravel()
                    rf[f, a - 1, b - 1, c - 1] = np.sqrt(np.mean((yh.ravel() - Y[te].ravel()) ** 2))
    sst = np.sum((Y - Y.mean(0)) ** 2)
    qa = 1 - ((P_a - Y) ** 2).sum(0) / sst
    qab = 1 - ((P_ab - Y[:, :, None]) ** 2).sum(0) / sst
    qabc = 1 - ((P_abc - Y[:, :, None, None]) ** 2).sum(0) / sst
    return qa, qab, qabc, rf


# ---------------------------------------------------------------------------
# REVERSED order: X1 -> X2 -> M2. Only need q_ab here (= X1+X2's ceiling,
# with X2 given first claim, uncontaminated by M2) to compute X2's
# standalone contribution as qab_rev - qa.
# ---------------------------------------------------------------------------
def surface_reversed(X1, X2, M2, Y, n_splits=10, seed=42):
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(Y)
    P_ab = np.zeros((n, A_MAX, B_MAX))
    for f, (tr, te) in enumerate(cv.split(Y)):
        x1t, x1e = _scale(X1, tr, te); x2t0, x2e0 = _scale(X2, tr, te)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy
        p1, T1t, Q1, af = _fit(x1t, Ytr0, A_MAX); T1e = p1.transform(x1e)
        for a in range(1, A_MAX + 1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            Tp = np.linalg.pinv(Tat.T @ Tat) @ Tat.T
            gX = Tp @ x2t0; gY = Tp @ Ytr0
            x2t, x2e = x2t0 - Tat @ gX, x2e0 - Tae @ gX
            Ytr_a = Ytr0 - Tat @ gY
            p2, T2t, Q2, bf = _fit(x2t, Ytr_a, B_MAX); T2e = p2.transform(x2e)
            for b in range(1, B_MAX + 1):
                be = min(b, bf); Tbt, Tbe = T2t[:, :be], T2e[:, :be]
                yh_ab = yh_a + Tbe @ Q2[:, :be].T
                P_ab[te, a - 1, b - 1] = (yh_ab * sdy + muY).ravel()
    sst = np.sum((Y - Y.mean(0)) ** 2)
    qab = 1 - ((P_ab - Y[:, :, None]) ** 2).sum(0) / sst
    return qab


# ---------------------------------------------------------------------------
# Per-model analysis: flat-region M2M distribution + shared-information metric
# ---------------------------------------------------------------------------
def analyze_model(name, X1, X2, Y, M2):
    qa, qab, qabc, rf = surface_standard(X1, M2, X2, Y)
    rmse = rf.mean(0); se = rf.std(0, ddof=1) / np.sqrt(rf.shape[0])
    i = np.unravel_index(rmse.argmin(), rmse.shape)
    best_abc = tuple(int(x) + 1 for x in i)
    thr = rmse[i] + se[i]
    flat = rmse <= thr

    dM2 = qab - qa[:, None]
    dX2 = qabc - qab[:, :, None]
    m2m_grid = dM2[:, :, None] / np.maximum(dX2, 1e-9)
    valid = np.isfinite(m2m_grid) & (m2m_grid > 0) & (m2m_grid < 1e4)
    v = m2m_grid[flat & valid]

    m2m_point = float(m2m_grid[i])
    x2_unique_at_best = float(dX2[i])

    qab_rev = surface_reversed(X1, X2, M2, Y)
    x2_alone_at_best = float((qab_rev - qa[:, None])[i[0], i[1]])
    shared_frac = 1.0 - (x2_unique_at_best / x2_alone_at_best) if x2_alone_at_best > 0 else float('nan')

    return {
        'name': name, 'best_abc': best_abc, 'rmsecv': float(rmse[i]), 'n_flat': int(flat.sum()),
        'm2m_point': m2m_point,
        'm2m_median': float(np.median(v)), 'm2m_p10': float(np.percentile(v, 10)),
        'm2m_p90': float(np.percentile(v, 90)), 'm2m_mean': float(v.mean()),
        'x2_unique_pct': x2_unique_at_best * 100, 'x2_alone_pct': x2_alone_at_best * 100,
        'shared_frac_pct': shared_frac * 100,
    }, v


# ---------------------------------------------------------------------------
# Distribution plot: all 7 models, one figure
# ---------------------------------------------------------------------------
def plot_all_distributions(results, arrays, outpath='m2m_all_models_distributions.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NAVY = '#16323F'; TEAL = '#3E7C8C'

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.ravel()
    for idx, r in enumerate(results):
        v = arrays[r['name']]
        ax = axes[idx]
        ax.hist(v, bins=20, color=TEAL, alpha=0.75, edgecolor='black', linewidth=0.4)
        ax.axvline(r['m2m_median'], color=NAVY, linestyle='--', linewidth=1.6)
        ax.set_title(f"{r['name']}\nmedian={r['m2m_median']:.2f}  shared={r['shared_frac_pct']:.1f}%", fontsize=10.5)
        ax.set_xlabel('M2M'); ax.set_ylabel('count'); ax.grid(alpha=0.2)
    axes[7].axis('off')
    plt.suptitle('M2M Flat-Region Distributions, All 7 Models', fontsize=14, y=1.01, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'Saved {outpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # regenerate the 6-model dataset (M0-M5) on the REAL X1/X2/Y via gen_all.py
    subprocess.run([sys.executable, 'gen_all.py'], check=True)
    d = pickle.load(open('all_models.pkl', 'rb'))
    X1 = d['X1'][['CA0', 'T1', 't1', 'A', 'B', 'C']].values
    X2 = d['X2'].values
    Y = d['Y'].values
    d['blocks']['M0_correct'] = pd.read_excel('M2.xlsx')  # use the REAL M2.xlsx, not gen_all.py's re-integration

    results, arrays = [], {}
    for name, df in d['blocks'].items():
        r, v = analyze_model(name, X1, X2, Y, df.values)
        results.append(r); arrays[name] = v
        print(f"{name:<16} best={str(r['best_abc']):<12} n_flat={r['n_flat']:>4}  "
              f"median={r['m2m_median']:6.2f}  P90={r['m2m_p90']:6.2f}  "
              f"shared={r['shared_frac_pct']:5.1f}%")

    # M6: its OWN native files, NOT gen_all.py's shared X1/X2/Y
    X1m = pd.read_excel('X1_mis.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
    X2m = pd.read_excel('X2_mis.xlsx').values
    Ym = pd.read_excel('Y_mis.xlsx').values
    M2m = pd.read_excel('M2_mis.xlsx').values
    r6, v6 = analyze_model('M6_author_mis', X1m, X2m, Ym, M2m)
    results.append(r6); arrays['M6_author_mis'] = v6
    print(f"{'M6_author_mis':<16} best={str(r6['best_abc']):<12} n_flat={r6['n_flat']:>4}  "
          f"median={r6['m2m_median']:6.2f}  P90={r6['m2m_p90']:6.2f}  "
          f"shared={r6['shared_frac_pct']:5.1f}%")

    json.dump(results, open('full_results.json', 'w'), indent=2)
    plot_all_distributions(results, arrays)
    print('\nSaved full_results.json and m2m_all_models_distributions.png')

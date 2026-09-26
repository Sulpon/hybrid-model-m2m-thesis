"""
TASK 1-7 implementation: mechanistic-only (M->Y), measured-only (X->Y), and
hybrid (M+X->Y) reference quantities for all 7 candidate models, plus the
commonality-style unique/shared decomposition.

DESIGN DECISIONS MADE EXPLICIT (see accompanying chat message for justification):

1. X1 handling: X1 is treated as a common baseline block, present in EVERY
   model (M-only, X-only, hybrid) at a FIXED number of components `a`, taken
   from that candidate's own existing RMSECV-optimal best_abc[0] in
   full_results.json. This keeps "batch/initial-condition" information
   constant across the comparison and makes R_M^2, R_X^2, R_MX^2 land on the
   same total-Y-variance scale as the existing SSM/SSX (both of which are
   ALSO defined as increments over X1 in m2m distribtuion.py). Concretely:
       R_M^2  = CV R^2 of (X1 + M  -> Y)
       R_X^2  = CV R^2 of (X1 + X2 -> Y)
       R_MX^2 = CV R^2 of (X1 + [M,X2 jointly] -> Y)

2. LV selection: M and X2 are EACH allowed their own CV-optimal component
   count (swept 1..15 independently), not forced to share the (b,c) from the
   model's original hybrid best_abc. This answers Task 2's fairness question:
   forcing M and X to use the same, or the hybrid-optimal, component count
   would unfairly cap whichever block "needs" more components to show what
   it alone can do. X1's LV count IS held fixed across the three models for
   a given candidate, because X1 is not part of the M-vs-X question.

3. R_MX^2 (joint, non-order-biased): implemented as CONCATENATED-block PLS
   regression (M and X2 residual-after-X1 columns stacked into one predictor
   matrix, standard sklearn PLSRegression) rather than a hand-extended
   MB-PLS. Reasoning: sklearn's PLSRegression already has a correct, tested
   .transform() for held-out CV prediction; the existing my_mbpls() in
   mbpls_compare_final.py only fits a single dataset and has no CV/held-out
   prediction capability, so extending it correctly would be substantial new,
   unvalidated code. Concatenated PLS still satisfies the requirement that no
   block gets first claim (components are chosen to jointly covary with Y
   across the combined variable space) -- it just doesn't produce block-level
   weights, which are not needed here since Unique_M/Unique_X/Shared are
   obtained by combining this joint R^2 with the independently-computed
   single-block R_M^2/R_X^2, not from MB-PLS block weights.

4. X2 is identical across M0-M5 (same X1.xlsx/X2.xlsx/Y.xlsx, only the M
   block differs -- confirmed in gen_all.py). R_X^2 therefore only needs
   recomputing when `a` changes. M6 uses its own native X1_mis/X2_mis/Y_mis
   files and always needs its own R_X^2.

Does NOT touch or import m2m distribtuion.py -- this is a new, additive
script. The _fit/_scale helpers below are a verbatim copy of the ones in
m2m distribtuion.py, kept in lockstep for consistency, not reimplemented
with different behaviour.
"""
import json, subprocess, sys, pickle
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

M_MAX, X_MAX, MX_MAX = 15, 15, 20
N_SPLITS, SEED = 10, 42


def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0] - 1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n


def _scale(B, tr, te):
    mu = B[tr].mean(0); sd = B[tr].std(0, ddof=1); sd[sd < 1e-12] = 1.0
    k = np.sqrt(B.shape[1])
    return (B[tr] - mu) / sd / k, (B[te] - mu) / sd / k


def cv_r2_single(X1, Xb, Y, a_fixed, block_max, n_splits=N_SPLITS, seed=SEED):
    """CV R^2 curve for (X1[a_fixed] + Xb[1..block_max]) -> Y."""
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(Y)
    P = np.zeros((n, block_max))
    for tr, te in cv.split(Y):
        x1t, x1e = _scale(X1, tr, te); bt0, be0 = _scale(Xb, tr, te)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy
        p1, T1t, Q1, af = _fit(x1t, Ytr0, a_fixed); T1e = p1.transform(x1e)
        ae = min(a_fixed, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
        yh_a = Tae @ Q1[:, :ae].T
        Tp = np.linalg.pinv(Tat.T @ Tat) @ Tat.T
        gB = Tp @ bt0; gY = Tp @ Ytr0
        bt, be = bt0 - Tat @ gB, be0 - Tae @ gB
        Ytr_a = Ytr0 - Tat @ gY
        p2, T2t, Q2, bf = _fit(bt, Ytr_a, block_max); T2e = p2.transform(be)
        for k in range(1, block_max + 1):
            ke = min(k, bf); Tkt, Tke = T2t[:, :ke], T2e[:, :ke]
            yh = (yh_a + Tke @ Q2[:, :ke].T) * sdy + muY
            P[te, k - 1] = yh.ravel()
    sst = np.sum((Y - Y.mean(0)) ** 2)
    r2 = 1 - ((P - Y) ** 2).sum(0) / sst
    best_k = int(np.argmax(r2)) + 1
    return r2, best_k, float(r2[best_k - 1])


def cv_r2_joint_concat(X1, Xm, Xx, Y, a_fixed, joint_max, n_splits=N_SPLITS, seed=SEED):
    """CV R^2 curve for (X1[a_fixed] + concat[M,X2][1..joint_max]) -> Y, a
    genuinely joint (non order-biased) fit of M and X2 together."""
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(Y)
    joint_max_eff = min(joint_max, Xm.shape[1] + Xx.shape[1])
    P = np.zeros((n, joint_max_eff))
    for tr, te in cv.split(Y):
        x1t, x1e = _scale(X1, tr, te); mt0, me0 = _scale(Xm, tr, te); xt0, xe0 = _scale(Xx, tr, te)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Ytr0 = (Y[tr] - muY) / sdy
        p1, T1t, Q1, af = _fit(x1t, Ytr0, a_fixed); T1e = p1.transform(x1e)
        ae = min(a_fixed, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
        yh_a = Tae @ Q1[:, :ae].T
        Tp = np.linalg.pinv(Tat.T @ Tat) @ Tat.T
        gM = Tp @ mt0; gX = Tp @ xt0; gY = Tp @ Ytr0
        mt, me = mt0 - Tat @ gM, me0 - Tae @ gM
        xt, xe = xt0 - Tat @ gX, xe0 - Tae @ gX
        Ytr_a = Ytr0 - Tat @ gY
        Ct, Ce = np.hstack([mt, xt]), np.hstack([me, xe])
        p2, T2t, Q2, jf = _fit(Ct, Ytr_a, joint_max_eff); T2e = p2.transform(Ce)
        for k in range(1, joint_max_eff + 1):
            ke = min(k, jf); Tkt, Tke = T2t[:, :ke], T2e[:, :ke]
            yh = (yh_a + Tke @ Q2[:, :ke].T) * sdy + muY
            P[te, k - 1] = yh.ravel()
    sst = np.sum((Y - Y.mean(0)) ** 2)
    r2 = 1 - ((P - Y) ** 2).sum(0) / sst
    best_k = int(np.argmax(r2)) + 1
    return r2, best_k, float(r2[best_k - 1])


if __name__ == '__main__':
    # regenerate M1-M5 candidate mechanistic blocks (deterministic, matches m2m distribtuion.py)
    subprocess.run([sys.executable, 'gen_all.py'], check=True)
    d = pickle.load(open('all_models.pkl', 'rb'))
    X1 = d['X1'][['CA0', 'T1', 't1', 'A', 'B', 'C']].values
    X2 = d['X2'].values
    Y = d['Y'].values
    d['blocks']['M0_correct'] = pd.read_excel('M2.xlsx')  # real M2.xlsx, not gen_all.py's re-integration

    full_results = {r['name']: r for r in json.load(open('full_results.json'))}

    rows = []
    rx2_cache = {}  # keyed by a_fixed, for the shared X1/X2/Y family (M0-M5)

    for name, Mdf in d['blocks'].items():
        M = Mdf.values
        a_fixed = full_results[name]['best_abc'][0]

        r2_m_curve, best_m, R_M2 = cv_r2_single(X1, M, Y, a_fixed, M_MAX)

        if a_fixed in rx2_cache:
            best_x, R_X2 = rx2_cache[a_fixed]
        else:
            r2_x_curve, best_x, R_X2 = cv_r2_single(X1, X2, Y, a_fixed, X_MAX)
            rx2_cache[a_fixed] = (best_x, R_X2)

        r2_mx_curve, best_mx, R_MX2 = cv_r2_joint_concat(X1, M, X2, Y, a_fixed, MX_MAX)

        Unique_M = R_MX2 - R_X2
        Unique_X = R_MX2 - R_M2
        Shared = R_M2 + R_X2 - R_MX2
        Residual = 1 - R_MX2

        rows.append({
            'model': name, 'a_X1': a_fixed, 'best_m': best_m, 'best_x': best_x, 'best_mx': best_mx,
            'R_M2': R_M2, 'R_X2': R_X2, 'R_MX2': R_MX2,
            'R_M2_over_R_X2': R_M2 / R_X2 if R_X2 > 0 else float('nan'),
            'R_X2_minus_R_M2': R_X2 - R_M2,
            'Unique_M': Unique_M, 'Unique_X': Unique_X, 'Shared': Shared, 'Residual': Residual,
            'closure_check': Unique_M + Shared + Unique_X + Residual,
            'm2m_point_existing': full_results[name]['m2m_point'],
        })
        print(f"{name:<16} a={a_fixed} best(m,x,mx)=({best_m:>2},{best_x:>2},{best_mx:>2})  "
              f"R_M2={R_M2:6.3f} R_X2={R_X2:6.3f} R_MX2={R_MX2:6.3f}  "
              f"Uniq_M={Unique_M:6.3f} Shared={Shared:6.3f} Uniq_X={Unique_X:6.3f}  "
              f"closure={Unique_M+Shared+Unique_X+Residual:.4f}")

    # M6: native mis files, its own X1/X2/Y (not shared with M0-M5)
    X1m = pd.read_excel('X1_mis.xlsx')[['CA0', 'T1', 't1', 'A', 'B', 'C']].values
    X2m = pd.read_excel('X2_mis.xlsx').values
    Ym = pd.read_excel('Y_mis.xlsx').values
    M2m = pd.read_excel('M2_mis.xlsx').values
    name = 'M6_author_mis'
    a_fixed = full_results[name]['best_abc'][0]
    r2_m_curve, best_m, R_M2 = cv_r2_single(X1m, M2m, Ym, a_fixed, M_MAX)
    r2_x_curve, best_x, R_X2 = cv_r2_single(X1m, X2m, Ym, a_fixed, X_MAX)
    r2_mx_curve, best_mx, R_MX2 = cv_r2_joint_concat(X1m, M2m, X2m, Ym, a_fixed, MX_MAX)
    Unique_M = R_MX2 - R_X2; Unique_X = R_MX2 - R_M2; Shared = R_M2 + R_X2 - R_MX2; Residual = 1 - R_MX2
    rows.append({
        'model': name, 'a_X1': a_fixed, 'best_m': best_m, 'best_x': best_x, 'best_mx': best_mx,
        'R_M2': R_M2, 'R_X2': R_X2, 'R_MX2': R_MX2,
        'R_M2_over_R_X2': R_M2 / R_X2 if R_X2 > 0 else float('nan'),
        'R_X2_minus_R_M2': R_X2 - R_M2,
        'Unique_M': Unique_M, 'Unique_X': Unique_X, 'Shared': Shared, 'Residual': Residual,
        'closure_check': Unique_M + Shared + Unique_X + Residual,
        'm2m_point_existing': full_results[name]['m2m_point'],
    })
    print(f"{name:<16} a={a_fixed} best(m,x,mx)=({best_m:>2},{best_x:>2},{best_mx:>2})  "
          f"R_M2={R_M2:6.3f} R_X2={R_X2:6.3f} R_MX2={R_MX2:6.3f}  "
          f"Uniq_M={Unique_M:6.3f} Shared={Shared:6.3f} Uniq_X={Unique_X:6.3f}  "
          f"closure={Unique_M+Shared+Unique_X+Residual:.4f}")

    out = pd.DataFrame(rows)
    out.to_json('reference_models_results.json', orient='records', indent=2)
    pd.set_option('display.width', 200)
    print("\n" + out.to_string(index=False))
    print("\nSaved reference_models_results.json")

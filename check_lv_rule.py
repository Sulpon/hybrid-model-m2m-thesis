"""
Does the case study 1 headline table survive the LV-selection rule we advocate?

The reported table (metric_unique_m2m.xlsx) fixed the full / KD / DD
allocations at their ARGMIN-RMSECV values. The methodology everywhere else --
and the source paper's own Sec. 5.2.1 -- selects the most parsimonious
allocation inside the 1-SE flat region instead.

This recomputes the same 20-seed CV estimator under the 1-SE rule and puts the
two side by side. Machinery is imported from case_study_1 unchanged.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
import case_study_1 as CS

SEEDS = [1, 3, 7, 11, 13, 17, 23, 42, 55, 77, 99, 123, 256, 404, 777,
         1000, 2024, 4096, 31337, 65535]
ARGMIN_LV = {'M0_correct': ((4, 6, 2), (3, 10), (1, 8)),
             'M1_no_side': ((1, 13, 4), (1, 10), (1, 8)),
             'M2_order1_D': ((4, 6, 4), (1, 8), (1, 8)),
             'M3_wrong_Ea4': ((4, 6, 2), (3, 10), (1, 8)),
             'M4_lumped_EF': ((4, 8, 2), (1, 13), (1, 8)),
             'M5_no_D': ((3, 7, 4), (1, 6), (1, 8)),
             'M6_author_mis': ((3, 3, 6), (3, 15), (6, 6))}


def cv2_rf(B1, B2, Y, m1, m2):
    """2-block SO-PLS returning the per-fold RMSE cube, so the 1-SE rule can
    be applied to the sub-models too. Same operations as case_study_1.cv_2."""
    cv = KFold(CS.N_SPLITS, shuffle=True, random_state=CS.SEED)
    rf = np.zeros((CS.N_SPLITS, m1, m2))
    for f, (tr, te) in enumerate(cv.split(Y)):
        mu1, sd1, k1 = CS._sf(B1[tr]); b1t, b1e = CS._sa(B1[tr], mu1, sd1, k1), CS._sa(B1[te], mu1, sd1, k1)
        mu2, sd2, k2 = CS._sf(B2[tr]); b2t0, b2e0 = CS._sa(B2[tr], mu2, sd2, k2), CS._sa(B2[te], mu2, sd2, k2)
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        p1, T1t, Q1, af = CS._fit(b1t, Y0, m1); T1e = p1.transform(b1e)
        for a in range(1, m1+1):
            ae = min(a, af); Tat, Tae = T1t[:, :ae], T1e[:, :ae]
            yh_a = Tae @ Q1[:, :ae].T
            b2t, g2 = CS._rs(Tat, b2t0); b2e = b2e0 - Tae @ g2
            Ya, _ = CS._rs(Tat, Y0)
            p2, T2t, Q2, bf = CS._fit(b2t, Ya, m2); T2e = p2.transform(b2e)
            for b in range(1, m2+1):
                be = min(b, bf)
                yh = ((yh_a + T2e[:, :be] @ Q2[:, :be].T)*sdy + muY).ravel()
                rf[f, a-1, b-1] = np.sqrt(np.mean((yh - Y[te].ravel())**2))
    return rf


def pick2(B1, B2, Y, m1, m2):
    rf = cv2_rf(B1, B2, Y, m1, m2)
    pick, _, _, _, _ = CS.pick_1se(rf.mean(0), rf)
    return pick


def q2_at(blocks, Y, lvs, seed):
    """Q2 at ONE fixed allocation, for a given CV seed. Same scaling,
    orthogonalisation and pooling as case_study_1; only the seed varies."""
    cv = KFold(CS.N_SPLITS, shuffle=True, random_state=seed)
    P = np.zeros(len(Y))
    for tr, te in cv.split(Y):
        sc = [CS._sf(B[tr]) for B in blocks]
        tr_b = [CS._sa(B[tr], *s) for B, s in zip(blocks, sc)]
        te_b = [CS._sa(B[te], *s) for B, s in zip(blocks, sc)]
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1)
        resid = (Y[tr]-muY)/sdy
        yh = np.zeros((len(te), 1))
        for k, n in enumerate(lvs):
            p, T, Q, nf = CS._fit(tr_b[k], resid, n)
            ne = min(n, nf); T, Q = T[:, :ne], Q[:, :ne]
            Te = p.transform(te_b[k])[:, :ne]
            yh = yh + Te @ Q.T
            resid, _ = CS._rs(T, resid)
            for j in range(k+1, len(blocks)):
                tr_b[j], g = CS._rs(T, tr_b[j])
                te_b[j] = te_b[j] - Te @ g
        P[te] = (yh*sdy + muY).ravel()
    return 1 - ((P - Y.ravel())**2).sum()/np.sum((Y - Y.mean())**2)


if __name__ == '__main__':
    A = CS.load_A()
    rows = []
    for name, rhs in CS.MODELS.items():
        d = A['mis'] if name == 'M6_author_mis' else A['std']
        X1, X2, Y = d['X1'], d['X2'], d['Y']
        M2 = CS.build_M2_A(rhs, d, CS.FITTED_A[name])

        # --- 1-SE parsimonious allocations, each sub-model on its own grid ---
        _, rf3 = CS.cv_3(X1, M2, X2, Y, CS.MAX_X1, CS.MAX_M2, CS.MAX_X2)
        p3, _, _, _, i3 = CS.pick_1se(rf3.mean(0), rf3)
        full = tuple(v+1 for v in p3)
        kd = tuple(v+1 for v in pick2(X1, M2, Y, CS.MAX_X1, CS.MAX_M2))
        dd = tuple(v+1 for v in pick2(X1, X2, Y, CS.MAX_X1, CS.MAX_X2))

        for tag, (f_, k_, d_) in [('1-SE', (full, kd, dd)),
                                  ('argmin', ARGMIN_LV[name])]:
            rs = []
            for s in SEEDS:
                qf = q2_at([X1, M2, X2], Y, f_, s)
                qk = q2_at([X1, M2], Y, k_, s)
                qd = q2_at([X1, X2], Y, d_, s)
                uM, uX = qf-qd, qf-qk
                rs.append(uM/uX if uX > 1e-9 else np.nan)
            rs = np.array(rs, float)
            rows.append(dict(model=name, rule=tag, LV_full=str(f_), LV_KD=str(k_),
                             LV_DD=str(d_), ratio_mean=np.nanmean(rs),
                             ratio_sd=np.nanstd(rs, ddof=1),
                             frac_gt1=float(np.nanmean(rs > 1))))
            print(f'{name:14s} {tag:6s} LV {str(f_):10s} KD {str(k_):8s} DD {str(d_):8s} '
                  f'M2M* {np.nanmean(rs):7.3f} +/- {np.nanstd(rs, ddof=1):.3f}  '
                  f'P(>1) {np.nanmean(rs>1):.2f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_excel('check_lv_rule.xlsx', index=False)
    p = df.pivot(index='model', columns='rule', values='ratio_mean')
    print('\n=== M2M* under each selection rule ===')
    print(p[['argmin', '1-SE']].round(3).to_string())
    BAD = {'M1_no_side', 'M5_no_D', 'M6_author_mis'}
    for rule in ['argmin', '1-SE']:
        v = p[rule]
        lo = min(v[m] for m in v.index if m not in BAD)
        hi = max(v[m] for m in v.index if m in BAD)
        print(f'  {rule:6s}: worst sound = {lo:.3f}   best broken = {hi:.3f}   '
              f'{"SEPARATES" if lo > hi else "OVERLAPS"}')
    print('\nSaved check_lv_rule.xlsx')

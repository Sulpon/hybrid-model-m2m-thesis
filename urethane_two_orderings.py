"""
Urethane case study: both block orderings, for the two models the papers define.

  Run 1 (forward)  M -> X      the mechanistic block enters first
  Run 2 (reverse)  X -> M      the measured block enters first

Each ordering gets its OWN blind argmin-RMSECV allocation, exactly as in the
two-ordering analysis of case study 1. Sequential (Type-I) sums of squares are
then taken at that allocation, on standardised Y, so SS_Y = n - 1 = 49.

Commonality identity for two blocks:
    unique_M = SS_M from the REVERSE run   (M entering last)
    unique_X = SS_X from the FORWARD run   (X entering last)
    shared   = SS_M(forward) - SS_M(reverse)  =  SS_X(reverse) - SS_X(forward)
The two expressions for the shared term agree exactly only when both orderings
use the same per-block latent-variable counts. With independently optimised
allocations they differ slightly; both are reported so the gap is visible.

Blocks as established for this case study:
    M  FP-predicted nC, nD, nE over the FULL batch + the KD inputs fv1, fv2, T
    X  measured nC, nD, nE up to a decision point
    Y  final nD  (final nC is unusable: its noise exceeds its batch-to-batch spread)
"""
import numpy as np
import pandas as pd
import warnings, json, os, time
warnings.filterwarnings('ignore')
import urethane_case_study as U
import urethane_m2m as M

HORIZONS = [30.0, 60.0]
MAXA, MAXB = 15, 15

def ss_surface(B1, B2, Y, m1, m2):
    """Sequential SS of BOTH blocks over the (a,b) grid for the ordering B1 -> B2."""
    mu1, sd1, k1 = M._sf(B1); b1 = M._sa(B1, mu1, sd1, k1)
    mu2, sd2, k2 = M._sf(B2); b20 = M._sa(B2, mu2, sd2, k2)
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y - muY)/sdy
    sst = float(np.sum(ys**2))
    S1 = np.zeros((m1, m2)); S2 = np.zeros((m1, m2))
    p1, T1, Q1, af = M._fit(b1, ys, m1)
    for a in range(1, m1+1):
        ae = min(a, af); Ta, Qa = T1[:, :ae], Q1[:, :ae]
        s1 = float(np.sum((Ta @ Qa.T)**2))
        b2r, _ = M._rs(Ta, b20); ya, _ = M._rs(Ta, ys)
        p2, T2v, Q2, bf = M._fit(b2r, ya, m2)
        for b in range(1, m2+1):
            be = min(b, bf)
            S1[a-1, b-1] = s1
            S2[a-1, b-1] = float(np.sum((T2v[:, :be] @ Q2[:, :be].T)**2))
    return S1, S2, sst

def run(B1, B2, Y, m1, m2):
    """Blind argmin RMSECV, then the sequential SS at that allocation."""
    q, rf = M.cv2(B1, B2, Y, m1, m2)
    rmse = rf.mean(0)
    i = np.unravel_index(rmse.argmin(), rmse.shape)
    S1, S2, sst = ss_surface(B1, B2, Y, m1, m2)
    return dict(lv1=i[0]+1, lv2=i[1]+1, RMSECV=float(rmse[i]), Q2=float(q[i]),
                ss1=float(S1[i]), ss2=float(S2[i]),
                ssF=sst - float(S1[i]) - float(S2[i]), SSY=sst)

if __name__ == '__main__':
    t0 = time.time()
    cache = json.load(open(M.PARAM_CACHE)) if os.path.exists(M.PARAM_CACHE) else {}
    t, inp, meas, clean, _, ic = U.generate(U.N_CALIBRATION, 101)
    Y = meas[:, -1, M.RESP_IDX].reshape(-1, 1)
    nb = len(inp)
    inputs_unf = np.array([[f(x) for x in t] for f in inp]).transpose(0, 2, 1).reshape(nb, -1)
    print(f"{nb} batches; Y = final {M.RESPONSE}; SS_Y = {nb-1}\n", flush=True)

    MODELS = [('U0_correct', M.u0_reversible, 'U0_reversible'),
              ('U1_FPmodel', M.u1_irrev, 'U1_irrev')]
    rows = []
    for H in HORIZONS:
        for tag, rhs, key in MODELS:
            if key not in cache:
                th_, c_ = M.fit(rhs, inp, meas, ic, t)
                cache[key] = dict(th=[float(v) for v in th_], cost=c_)
                json.dump(cache, open(M.PARAM_CACHE, 'w'), indent=2)
            th = np.array(cache[key]['th'])
            pred = np.array([M.sim(rhs, th, f, ic, t) for f in inp])
            Mb = np.hstack([pred.transpose(0, 2, 1).reshape(nb, -1), inputs_unf])
            Xb = meas[:, t <= H, :].transpose(0, 2, 1).reshape(nb, -1)

            f_ = run(Mb, Xb, Y, MAXA, MAXB)      # forward  M -> X
            r_ = run(Xb, Mb, Y, MAXB, MAXA)      # reverse  X -> M

            uM, uX = r_['ss2'], f_['ss2']        # each block entering last
            sh_f = f_['ss1'] - r_['ss2']         # SS_M(fwd) - SS_M(rev)
            sh_r = r_['ss1'] - f_['ss2']         # SS_X(rev) - SS_X(fwd)
            rows.append(dict(
                horizon=H, model=tag,
                fwd_LV=f"({f_['lv1']},{f_['lv2']})", fwd_RMSECV=f_['RMSECV'], fwd_Q2=f_['Q2'],
                fwd_SS_M=f_['ss1'], fwd_SS_X=f_['ss2'], fwd_SS_F=f_['ssF'],
                fwd_M2M=f_['ss1']/f_['ss2'] if f_['ss2'] > 1e-12 else np.nan,
                rev_LV=f"({r_['lv2']},{r_['lv1']})", rev_RMSECV=r_['RMSECV'], rev_Q2=r_['Q2'],
                rev_SS_M=r_['ss2'], rev_SS_X=r_['ss1'], rev_SS_F=r_['ssF'],
                rev_M2M=r_['ss2']/r_['ss1'] if r_['ss1'] > 1e-12 else np.nan,
                unique_M=uM, unique_X=uX, shared_fwd=sh_f, shared_rev=sh_r,
                shared_mean=0.5*(sh_f + sh_r), SSY=f_['SSY']))
            r = rows[-1]
            print(f"  H={H:.0f}h {tag:11s} FWD {r['fwd_LV']:7s} SS_M={r['fwd_SS_M']:6.3f} "
                  f"SS_X={r['fwd_SS_X']:6.3f} | REV {r['rev_LV']:7s} SS_M={r['rev_SS_M']:6.3f} "
                  f"SS_X={r['rev_SS_X']:6.3f}   ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 260, 'display.max_columns', 40)
    for H in HORIZONS:
        s = df[df.horizon == H]
        print(f"\n=== decision point t <= {H:.0f} h ===")
        print("\n-- Run 1: M -> X (mechanistic block first) --")
        print(s[['model', 'fwd_LV', 'fwd_RMSECV', 'fwd_Q2', 'fwd_SS_M', 'fwd_SS_X',
                 'fwd_SS_F', 'fwd_M2M']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print("\n-- Run 2: X -> M (measured block first) --")
        print(s[['model', 'rev_LV', 'rev_RMSECV', 'rev_Q2', 'rev_SS_M', 'rev_SS_X',
                 'rev_SS_F', 'rev_M2M']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        print("\n-- commonality split (SS units, SS_Y = 49) --")
        print(s[['model', 'unique_M', 'unique_X', 'shared_fwd', 'shared_rev',
                 'shared_mean']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    df.to_excel('urethane_two_orderings.xlsx', index=False)
    print(f"\nSaved urethane_two_orderings.xlsx   ({time.time()-t0:.0f}s)")

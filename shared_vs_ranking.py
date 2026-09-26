"""
Does the SHARED contribution carry ranking information that the ratio discards?

The modified ratio uses only the two unique terms. This script computes the full
commonality split -- unique_M, unique_X, shared -- for every candidate on the
Regime C data (paper-style initial conditions), and asks whether each component
predicts (a) mechanism quality, measured independently by the kinetic fit cost,
and (b) out-of-domain performance from the rolling-prediction test.

Blocks and conventions are unchanged: X1 -> M2 -> X2 -> Y, KFold(10, seed 42),
block scaling 1/sqrt(K), parsimonious allocation inside the 1-SE flat region.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import warnings, json
warnings.filterwarnings('ignore')

R = 8.314
SPECIES = ['A', 'B', 'C', 'D', 'E', 'F']
MAX_X1, MAX_M2, MAX_X2 = 6, 15, 15
N_SPLITS, SEED = 10, 42
DIL = 0.1/0.2

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
MODELS = {'M0_correct': ode_correct, 'M1_no_side': ode_no_side, 'M2_order1_D': ode_order1_D,
          'M3_wrong_Ea4': ode_wrong_Ea4, 'M4_lumped_EF': ode_lumped, 'M5_no_D': ode_no_D,
          'M6_author_mis': ode_author_mis}

def read(fn):
    x1 = pd.read_excel(fn, sheet_name='X1')[['CA0','T1','t1','CA1_final','CB1_final','CC1_final']].values
    x2 = pd.read_excel(fn, sheet_name='X2')
    y = pd.read_excel(fn, sheet_name='Y')['E_purity'].values.reshape(-1, 1)
    return x1, x2, y
X1c, D2c, Yc = read('RG_calibration.xlsx')
NPTS = (D2c.shape[1]-3)//6
TG = np.arange(NPTS)*15.0
n = len(Yc)
CD0 = D2c['CD0'].values
ICc = np.column_stack([X1c[:,3]*DIL, X1c[:,4]*DIL, X1c[:,5]*DIL, CD0, np.zeros(n), np.zeros(n)])
X2c = np.array([[D2c[f'{s}_t{k+1}'].iloc[i] for s in SPECIES for k in range(NPTS)] for i in range(n)])
METAc = D2c[['t2_final','T2','CD0']].values
T2c = D2c['T2'].values

def _fit(X, Yv, m):
    m = max(1, min(m, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(m, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, m
def _sf(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])
def _sa(B, mu, sd, k): return (B-mu)/sd/k
def _rs(T, tgt):
    g = np.linalg.pinv(T.T@T)@T.T@tgt
    return tgt - T@g, g

def cvN(blocks, Y, ms):
    """Q2 surface over the LV grid for an ordered list of blocks, plus fold RMSE."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=SEED)
    shape = tuple(ms)
    P = np.zeros((len(Y),)+shape); rf = np.zeros((N_SPLITS,)+shape)
    for f, (tr, te) in enumerate(cv.split(Y)):
        sc = []
        for B in blocks:
            mu, sd, k = _sf(B[tr]); sc.append([_sa(B[tr],mu,sd,k), _sa(B[te],mu,sd,k)])
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); Y0 = (Y[tr]-muY)/sdy
        if len(blocks) == 1:
            p, T, Q, nf = _fit(sc[0][0], Y0, ms[0]); Te = p.transform(sc[0][1])
            for a in range(1, ms[0]+1):
                ae = min(a, nf)
                yh = ((Te[:,:ae]@Q[:,:ae].T)*sdy+muY).ravel()
                P[te, a-1] = yh; rf[f, a-1] = np.sqrt(np.mean((yh-Y[te].ravel())**2))
        else:
            b1t, b1e = sc[0]; b2t0, b2e0 = sc[1]
            b3t0, b3e0 = sc[2] if len(blocks) > 2 else (None, None)
            p1, T1t, Q1, af = _fit(b1t, Y0, ms[0]); T1e = p1.transform(b1e)
            for a in range(1, ms[0]+1):
                ae = min(a, af); Tat, Tae = T1t[:,:ae], T1e[:,:ae]
                yh_a = Tae@Q1[:,:ae].T
                b2t, g2 = _rs(Tat, b2t0); b2e = b2e0 - Tae@g2
                Ya, _ = _rs(Tat, Y0)
                if len(blocks) > 2:
                    b3a, g3 = _rs(Tat, b3t0); b3ea = b3e0 - Tae@g3
                p2, T2t, Q2, bf = _fit(b2t, Ya, ms[1]); T2e = p2.transform(b2e)
                for b in range(1, ms[1]+1):
                    be = min(b, bf); Tbt, Tbe = T2t[:,:be], T2e[:,:be]
                    yh_ab = yh_a + Tbe@Q2[:,:be].T
                    if len(blocks) == 2:
                        yh = (yh_ab*sdy+muY).ravel()
                        P[te, a-1, b-1] = yh; rf[f, a-1, b-1] = np.sqrt(np.mean((yh-Y[te].ravel())**2))
                    else:
                        b3b, g4 = _rs(Tbt, b3a); b3eb = b3ea - Tbe@g4
                        Yb, _ = _rs(Tbt, Ya)
                        p3, T3t, Q3, cf = _fit(b3b, Yb, ms[2]); T3e = p3.transform(b3eb)
                        for c in range(1, ms[2]+1):
                            ce = min(c, cf)
                            yh = ((yh_ab + T3e[:,:ce]@Q3[:,:ce].T)*sdy+muY).ravel()
                            P[te, a-1, b-1, c-1] = yh
                            rf[f, a-1, b-1, c-1] = np.sqrt(np.mean((yh-Y[te].ravel())**2))
    sst = np.sum((Y-Y.mean(0))**2)
    q = 1 - ((P - Y.reshape((-1,)+(1,)*len(shape)))**2).sum(0)/sst
    return q, rf

if __name__ == '__main__':
    cache = json.load(open('regen_fitted_params_paperIC.json'))
    rows = []
    qX1, _ = cvN([X1c], Yc, [MAX_X1])
    for name, rhs in MODELS.items():
        p = cache[name]
        M2 = np.zeros((n, 6, NPTS))
        for i in range(n):
            M2[i] = odeint(rhs, ICc[i], TG, args=(T2c[i], *p)).T
        M2c = np.hstack([M2.reshape(n, -1), METAc])

        qF, rf = cvN([X1c, M2c, X2c], Yc, [MAX_X1, MAX_M2, MAX_X2])
        qKD, _ = cvN([X1c, M2c], Yc, [MAX_X1, MAX_M2])       # X1 + M2
        qDD, _ = cvN([X1c, X2c], Yc, [MAX_X1, MAX_X2])       # X1 + X2
        rmse = rf.mean(0)
        i0 = np.unravel_index(rmse.argmin(), rmse.shape)
        se = float(rf[:, i0[0], i0[1], i0[2]].std(ddof=1)/np.sqrt(N_SPLITS))
        mask = rmse <= rmse[i0] + se
        idx = np.argwhere(mask); tot = idx.sum(1)+3
        cand = idx[tot == tot.min()]
        pick = tuple(cand[np.array([rmse[tuple(u)] for u in cand]).argmin()])
        a, b, c = pick
        uM = float(qF[pick] - qDD[a, c])
        uX = float(qF[pick] - qKD[a, b])
        joint = float(qF[pick] - qX1[a])
        rows.append(dict(model=name, fit_cost=float('nan'),
                         Q2_full=float(qF[pick]), joint=joint,
                         uM=uM, uX=uX, shared=joint-uM-uX,
                         ratio=uM/uX if uX > 1e-9 else np.nan,
                         shared_frac=(joint-uM-uX)/joint))
        print(f'{name:14s} Q2 {qF[pick]:.4f}  uM {uM:+.4f}  uX {uX:+.4f}  '
              f'shared {joint-uM-uX:+.4f}  ratio {rows[-1]["ratio"]:6.3f}', flush=True)

    df = pd.DataFrame(rows).set_index('model')
    cost = pd.read_excel('regen_fit_and_test_paperIC.xlsx').set_index('model')['fit_cost']
    df['fit_cost'] = cost
    cu = pd.read_excel('iterative_prediction.xlsx', sheet_name='rmsep_by_k')
    Ye = pd.read_excel('RG_extrap.xlsx', sheet_name='Y').E_purity.values
    SSTe = ((Ye-Ye.mean())**2).sum()
    roll = cu[cu.method == 'SO-PLS-offlineCal']
    roll = roll[roll.regime == 'extrap'].groupby('model').R2p.mean()
    df['roll_R2e'] = roll

    pd.set_option('display.width', 220)
    print('\n=== commonality split, sorted by shared ===')
    print(df[['Q2_full','uM','uX','shared','shared_frac','ratio','fit_cost','roll_R2e']]
          .sort_values('shared', ascending=False).round(4).to_string())
    print('\n=== Spearman against mechanism quality and extrapolation (n=7) ===')
    for col in ['uM','uX','shared','shared_frac','ratio','Q2_full']:
        s = df[[col,'fit_cost']].dropna(); a = spearmanr(s[col], s.fit_cost)
        s2 = df[[col,'roll_R2e']].dropna(); b = spearmanr(s2[col], s2.roll_R2e)
        print(f'  {col:12s} vs fit_cost {a.correlation:+.3f} (p={a.pvalue:.3f}, n={len(s)})'
              f'   vs rolling R2 {b.correlation:+.3f} (p={b.pvalue:.3f}, n={len(s2)})')
    df.to_excel('shared_vs_ranking.xlsx')
    print('\nSaved shared_vs_ranking.xlsx')

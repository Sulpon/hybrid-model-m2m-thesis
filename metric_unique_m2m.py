"""
Assessment of the proposed metric

        M2M_unique = unique(M2) / unique(X2)

against the current  M2M = SS_M2_forward / SS_X2_forward, which credits the
shared M2/X2 variance entirely to M2.

The proposal has a natural reference at 1 (M2 adds more than X2 / less than X2).
Whether that reference is meaningful depends on how "unique" is estimated, so
FOUR estimators are computed and compared. They are NOT interchangeable.

  (SS-own)   in-sample sequential SS, each ordering at its OWN argmin-RMSECV
             allocation.  unique_M = SS_M2 from X1->X2->M2
                          unique_X = SS_X2 from X1->M2->X2
             WARNING: the two numerators use different LV counts.

  (SS-match) same, but both orderings forced to the SAME per-block LV counts
             (the forward full-model optimum a,b,c). unique_M then uses b LVs
             for M2 and unique_X uses c LVs for X2 -- still not equal to each
             other, but now each block is held at the allocation the full model
             actually chose for it, so the comparison is not contaminated by
             the reverse run re-optimising M2 upward.

  (SS-equal) both uniques computed with the SAME number of LVs for the entering
             block (L = min(b,c) and L = max(b,c) both reported), which is the
             only fully LV-symmetric in-sample version.

  (CV)       cross-validated predictive uniques:
                 unique_M = Q2_full - Q2_DD(X1+X2)
                 unique_X = Q2_full - Q2_KD(X1+M2)
             In-sample SS cannot decrease when LVs are added; CV uniques can,
             so this version is self-limiting.

Stability of the CV version is checked over 20 KFold seeds at FIXED allocations
(the seed-42 optima), so the spread reflects partition noise only, not re-search.

NOTE ON GROUND TRUTH: M0 is the only correctly specified mechanism. M1..M5 are
misspecified in known ways; M6 is the author's misspecification AND runs on the
*_mis dataset, so M6 is never compared numerically against the others.
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import warnings, time
warnings.filterwarnings('ignore')

R = 8.314
species = ['A', 'B', 'C', 'D', 'E', 'F']
N_SPLITS = 10
SEED = 42
SEEDS = [1, 3, 7, 11, 13, 17, 23, 42, 55, 77, 99, 123, 256, 404, 777, 1000, 2024, 4096, 31337, 65535]

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
FITTED = {
    'M0_correct':    [4.9380789e+04, 5.6481086e+04, 8.023, 33.615],
    'M1_no_side':    [5.0088694e+04, 5.5000000e+04, 11.377, 20.0],
    'M2_order1_D':   [3.1123942e+04, 4.3435688e+04, 6.397, 145.759],
    'M3_wrong_Ea4':  [4.99379e+04, 5.50000e+04, 9.75, 3.506],
    'M4_lumped_EF':  [5.0039241e+04, 5.5000000e+04, 9.883, 20.0],
    'M5_no_D':       [2577.581, 658929.499, 9735.492, 9822.344],
    'M6_author_mis': [3.5882909e+04, 5.5000000e+04, 38.713, 20.0],
}

X1a = pd.read_excel('X1.xlsx')[['CA0','T1','t1','A','B','C']].values
X2a = pd.read_excel('X2.xlsx'); Ya = pd.read_excel('Y.xlsx')['E_pur'].values.reshape(-1,1)
M2a_meta = pd.read_excel('M2.xlsx')[['t2','T2','D0']]
n_pts_a = X2a.shape[1]//6
X1m = pd.read_excel('X1_mis.xlsx')[['CA0','T1','t1','A','B','C']].values
X2m = pd.read_excel('X2_mis.xlsx'); Ym = pd.read_excel('Y_mis.xlsx')['E_pur'].values.reshape(-1,1)
M2m_meta = pd.read_excel('M2_mis.xlsx')[['t2','T2','D0']]
n_pts_m = X2m.shape[1]//6
N = len(X1a)

def build_ic_and_X2(X2df, n_pts):
    IC = np.array([[X2df[f'{s}_1'].iloc[i] for s in species] for i in range(len(X2df))])
    X2flat = np.array([[X2df[f'{s}_{k+1}'].iloc[i] for s in species for k in range(n_pts)]
                        for i in range(len(X2df))])
    return IC, X2flat
IC_a, X2flat_a = build_ic_and_X2(X2a, n_pts_a)
IC_m, X2flat_m = build_ic_and_X2(X2m, n_pts_m)

def build_M2(rhs, IC, T2v, n_pts, tgrid, meta, params):
    M2 = np.zeros((N, 6, n_pts))
    for i in range(N):
        M2[i] = odeint(rhs, IC[i], tgrid, args=(T2v[i], *params)).T
    return np.hstack([M2.reshape(N, -1), meta.values])

def _fit(X, Yv, n):
    n = max(1, min(n, X.shape[1], X.shape[0]-1, np.linalg.matrix_rank(X)))
    p = PLSRegression(n, scale=False).fit(X, Yv)
    return p, p.transform(X), p.y_loadings_, n
def _scale_fit(B):
    mu = B.mean(0); sd = B.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    return mu, sd, np.sqrt(B.shape[1])
def _scale_apply(B, mu, sd, k): return (B - mu)/sd/k
def _resid(T, target):
    g = np.linalg.pinv(T.T @ T) @ T.T @ target
    return target - T @ g, g

def seq_SS(blocks, Y, ns):
    """In-sample sequential SS for an arbitrary ordered list of blocks."""
    muY, sdy = Y.mean(0), Y.std(0, ddof=1); ys = (Y-muY)/sdy
    sst = float(np.sum(ys**2))
    scaled = []
    for B in blocks:
        mu, sd, k = _scale_fit(B); scaled.append(_scale_apply(B, mu, sd, k))
    out, ycur = [], ys
    for j, (B, n) in enumerate(zip(scaled, ns)):
        p, T, Q, nf = _fit(B, ycur, n); ne = min(n, nf); T, Q = T[:,:ne], Q[:,:ne]
        out.append(float(np.sum((T @ Q.T)**2)))
        for jj in range(j+1, len(scaled)):
            scaled[jj], _ = _resid(T, scaled[jj])
        ycur, _ = _resid(T, ycur)
    return out, sst

def cv_fixed(blocks, Y, ns, seed):
    """Q2 and RMSECV for one ordered block sequence at FIXED LV counts."""
    cv = KFold(N_SPLITS, shuffle=True, random_state=seed)
    P = np.zeros(len(Y)); rf = np.zeros(N_SPLITS)
    for f, (tr, te) in enumerate(cv.split(Y)):
        sc = []
        for B in blocks:
            mu, sd, k = _scale_fit(B[tr])
            sc.append([_scale_apply(B[tr],mu,sd,k), _scale_apply(B[te],mu,sd,k)])
        muY, sdy = Y[tr].mean(0), Y[tr].std(0, ddof=1); ycur = (Y[tr]-muY)/sdy
        yh = np.zeros((len(te), 1))
        for j, (pair, n) in enumerate(zip(sc, ns)):
            Btr, Bte = pair
            p, T, Q, nf = _fit(Btr, ycur, n); ne = min(n, nf)
            Te = p.transform(Bte)[:, :ne]; T = T[:, :ne]; Q = Q[:, :ne]
            yh += Te @ Q.T
            for jj in range(j+1, len(sc)):
                res, g = _resid(T, sc[jj][0]); sc[jj][0] = res; sc[jj][1] = sc[jj][1] - Te @ g
            ycur, _ = _resid(T, ycur)
        pred = (yh*sdy + muY).ravel()
        P[te] = pred
        rf[f] = np.sqrt(np.mean((pred - Y[te].ravel())**2))
    sst = float(np.sum((Y - Y.mean(0))**2))
    return 1 - float(((P - Y.ravel())**2).sum())/sst, float(rf.mean())

# seed-42 optima established earlier under the identical protocol
fwd = pd.read_excel('sopls_two_orderings.xlsx').set_index('model')
tb_m2 = pd.read_excel('sopls_two_block.xlsx', sheet_name='X1_M2').set_index('model')
tb_x2 = pd.read_excel('sopls_two_block.xlsx', sheet_name='X1_X2').set_index('dataset')

rows = []
perseed = []
t0 = time.time()
for name, rhs in MODELS.items():
    if name == 'M6_author_mis':
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1m, M2m_meta, Ym, n_pts_m, IC_m, X2flat_m
        dd_key = 'mis dataset (M6)'
    else:
        X1_, meta_, Y_, n_pts_, IC_, X2_ = X1a, M2a_meta, Ya, n_pts_a, IC_a, X2flat_a
        dd_key = 'main dataset (M0-M5)'
    tgrid = np.arange(n_pts_)*15.0
    M2_ = build_M2(rhs, IC_, meta_['T2'].values, n_pts_, tgrid, meta_, FITTED[name])

    fr = fwd.loc[name]
    a, b, c = int(fr.fwd_LV_X1), int(fr.fwd_LV_M2), int(fr.fwd_LV_X2)
    ar, cr, br = int(fr.rev_LV_X1), int(fr.rev_LV_X2), int(fr.rev_LV_M2)

    # ---------- current M2M (forward, shared credited to M2) ----------
    M2M_current = float(fr.fwd_SS_M2)/float(fr.fwd_SS_X2)

    # ---------- (SS-own) each ordering at its own optimum ----------
    uM_own, uX_own = float(fr.rev_SS_M2), float(fr.fwd_SS_X2)
    lv_uM_own, lv_uX_own = br, c

    # ---------- (SS-match) both orderings at the forward full optimum ----------
    ssf, _ = seq_SS([X1_, M2_, X2_], Y_, [a, b, c])      # X1, M2, X2
    ssr, _ = seq_SS([X1_, X2_, M2_], Y_, [a, c, b])      # X1, X2, M2 -- same per-block LVs
    uX_match, uM_match = ssf[2], ssr[2]

    # ---------- (SS-equal) identical LV count L for the entering block ----------
    eq = {}
    for L in sorted({min(b, c), max(b, c)}):
        s_f, _ = seq_SS([X1_, M2_, X2_], Y_, [a, b, L])
        s_r, _ = seq_SS([X1_, X2_, M2_], Y_, [a, c, L])
        eq[L] = (s_r[2], s_f[2])                          # (unique_M, unique_X) at L LVs each

    # ---------- (CV) predictive uniques, fixed allocations, many seeds ----------
    kd_a, kd_b = int(tb_m2.loc[name].LV_X1), int(tb_m2.loc[name].LV_M2)
    dd_a, dd_c = int(tb_x2.loc[dd_key].LV_X1), int(tb_x2.loc[dd_key].LV_X2)
    ratios, uMs, uXs = [], [], []
    for s in SEEDS:
        q_full, r_full = cv_fixed([X1_, M2_, X2_], Y_, [a, b, c], s)
        q_kd,  r_kd    = cv_fixed([X1_, M2_],      Y_, [kd_a, kd_b], s)
        q_dd,  r_dd    = cv_fixed([X1_, X2_],      Y_, [dd_a, dd_c], s)
        uM, uX = q_full - q_dd, q_full - q_kd
        uMs.append(uM); uXs.append(uX); ratios.append(uM/uX if uX > 1e-9 else np.nan)
        perseed.append(dict(model=name, seed=s,
            LV_full=f"({a},{b},{c})", LV_KD=f"({kd_a},{kd_b})", LV_DD=f"({dd_a},{dd_c})",
            Q2_full=q_full, Q2_KD=q_kd, Q2_DD=q_dd,
            RMSECV_full=r_full, RMSECV_KD=r_kd, RMSECV_DD=r_dd,
            unique_M=uM, unique_X=uX, ratio=uM/uX if uX > 1e-9 else np.nan))
    ratios, uMs, uXs = np.array(ratios), np.array(uMs), np.array(uXs)
    i42 = SEEDS.index(42)

    rows.append(dict(model=name,
        LV_full=f"({a},{b},{c})", LV_rev=f"({ar},{cr},{br})",
        M2M_current=M2M_current,
        SSown_uM=uM_own, SSown_uX=uX_own, SSown_ratio=uM_own/uX_own,
        SSown_LV_uM=lv_uM_own, SSown_LV_uX=lv_uX_own,
        SSmatch_uM=uM_match, SSmatch_uX=uX_match, SSmatch_ratio=uM_match/uX_match,
        SSeq_lo_L=min(eq), SSeq_lo_ratio=eq[min(eq)][0]/eq[min(eq)][1],
        SSeq_hi_L=max(eq), SSeq_hi_ratio=eq[max(eq)][0]/eq[max(eq)][1],
        CV_uM_42=uMs[i42], CV_uX_42=uXs[i42], CV_ratio_42=ratios[i42],
        CV_ratio_mean=ratios.mean(), CV_ratio_sd=ratios.std(ddof=1),
        CV_ratio_min=ratios.min(), CV_ratio_max=ratios.max(),
        CV_frac_gt1=float((ratios > 1).mean())))
    r = rows[-1]
    print(f"{name:14s} cur={M2M_current:6.3f} | SSown={r['SSown_ratio']:6.3f} "
          f"(LV {lv_uM_own} vs {c}) | SSmatch={r['SSmatch_ratio']:6.3f} | "
          f"SSeq@{min(eq)}={r['SSeq_lo_ratio']:6.3f} SSeq@{max(eq)}={r['SSeq_hi_ratio']:6.3f} | "
          f"CV={r['CV_ratio_42']:6.3f} (mean {ratios.mean():.3f}+/-{ratios.std(ddof=1):.3f}, "
          f"{ratios.min():.3f}-{ratios.max():.3f}, P(>1)={r['CV_frac_gt1']:.2f})", flush=True)

df = pd.DataFrame(rows)
pd.set_option('display.width', 320, 'display.max_columns', 80)
print("\n=== ratio under each estimator (M6 on a DIFFERENT dataset -- not comparable) ===")
print(df[['model','M2M_current','SSown_ratio','SSmatch_ratio','SSeq_lo_ratio','SSeq_hi_ratio',
          'CV_ratio_42','CV_ratio_mean','CV_ratio_sd','CV_frac_gt1']].round(4).to_string(index=False))
print("\n=== LV asymmetry in the SS-own estimator ===")
print(df[['model','LV_full','LV_rev','SSown_LV_uM','SSown_LV_uX','SSown_uM','SSown_uX',
          'SSmatch_uM','SSmatch_uX']].round(3).to_string(index=False))

sub = df[df.model != 'M6_author_mis']
print("\n=== rank order on the comparable six (best first) ===")
for col in ['M2M_current','SSown_ratio','SSmatch_ratio','SSeq_lo_ratio','CV_ratio_mean']:
    order = sub.sort_values(col, ascending=False).model.str.split('_').str[0].tolist()
    print(f"  {col:16s}: {' > '.join(order)}")
print("\n=== sign test vs the reference value 1 (CV estimator, 20 seeds) ===")
print(df[['model','CV_ratio_mean','CV_ratio_min','CV_ratio_max','CV_frac_gt1']].round(4).to_string(index=False))

ps = pd.DataFrame(perseed)
print("\n=== FULL per-seed CV components (20 seeds x 7 models) ===")
print(ps.round(5).to_string(index=False))
print("\n=== per-model mean of the CV components over the 20 seeds ===")
print(ps.groupby('model')[['Q2_full','Q2_KD','Q2_DD','RMSECV_full','RMSECV_KD','RMSECV_DD',
                           'unique_M','unique_X','ratio']].mean().round(5).to_string())
print("\n=== SS components used by the in-sample estimators ===")
print(df[['model','LV_full','LV_rev','SSown_LV_uM','SSown_LV_uX','SSown_uM','SSown_uX','SSown_ratio',
          'SSmatch_uM','SSmatch_uX','SSmatch_ratio','SSeq_lo_L','SSeq_lo_ratio',
          'SSeq_hi_L','SSeq_hi_ratio','M2M_current']].round(4).to_string(index=False))
with pd.ExcelWriter('metric_unique_m2m.xlsx') as w:
    df.to_excel(w, sheet_name='summary', index=False)
    ps.to_excel(w, sheet_name='per_seed', index=False)
print(f"\nSaved metric_unique_m2m.xlsx (sheets: summary, per_seed)   ({time.time()-t0:.0f}s)")

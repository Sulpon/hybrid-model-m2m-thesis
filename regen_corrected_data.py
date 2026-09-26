"""
Regenerate a calibration + within-domain + extrapolation triple using the
CORRECTED diffusion coefficients (Data_Generation_corrected.py, i.e. the
author's notebook values sigma1=[0.75,0.75,0.25],
sigma2=[0.25,0.75,0.25,0.75,0.75,0.25]).

Data_Generation_corrected.py is imported unchanged. The only thing this script
does to it is set the module attribute T2_range_testing at runtime to carve a
strictly-outside-domain extrapolation range; the file itself is not edited.

Three independent sets, 100 batches each:
    calibration   T2 ~ U(330, 370)   <- model is trained here
    within        T2 ~ U(330, 370)   <- proper in-domain control (independent draw)
    extrapolation T2 ~ U(370, 382)   <- strictly outside the calibration range

The existing DG_* within-domain set spanned only T2 362-370 (the top 20% of the
calibration range), which is why extrapolation appeared to beat it. This draws
the control from the same distribution as calibration, which is what a control
has to be.

Two deliberate deviations from gen_calibration_extrapolation.py, both to avoid
contaminating calibration with test data:
  1. the 0.35%-of-mean measurement-noise scale is computed from the CALIBRATION
     batches only and then applied to all three sets (it is an instrument
     property, not a per-set quantity);
  2. X2 is truncated to a COMMON number of timepoints across all three sets, so
     the blocks are column-aligned between train and test. Truncating per split
     (as the original did) can silently produce different widths.

Y is built from each batch's OWN final discretized point, per the established
convention -- not from the truncated last column of X2.
"""
import numpy as np
import pandas as pd
import time
import Data_Generation_corrected as dgc

SEED = 20260916
N_CAL = N_WIT = N_EXT = 100
NOISE_PCT = 0.0035
SAMPLE_INTERVAL = 0.25          # minutes
SPECIES = ['A', 'B', 'C', 'D', 'E', 'F']

np.random.seed(SEED)
t0 = time.time()

def simulate(n, data_type, label, t2_range=None):
    """Draw n batches, resampling any that produce a non-finite trajectory."""
    old = dgc.T2_range_testing
    if t2_range is not None:
        dgc.T2_range_testing = t2_range
    s1, s2, pars, retries = [], [], [], 0
    try:
        while len(pars) < n:
            (t1v, sol1), (t2v, sol2), p = dgc.one_sde_simulation(data_type)
            if not (np.isfinite(sol1).all() and np.isfinite(sol2).all()):
                retries += 1
                continue
            s1.append((t1v, sol1)); s2.append((t2v, sol2)); pars.append(p)
    finally:
        dgc.T2_range_testing = old
    print(f"  {label:13s} n={n}  resampled {retries} divergent batch(es)", flush=True)
    return s1, s2, pars

print("Simulating (corrected sigma values)...", flush=True)
c1, c2, cp = simulate(N_CAL, "training", "calibration")
w1, w2, wp = simulate(N_WIT, "training", "within")
e1, e2, ep = simulate(N_EXT, "testing",  "extrapolation", t2_range=(370.0, 382.0))
print(f"Simulation done in {time.time()-t0:.1f}s\n", flush=True)

# ---- measurement-noise scale, from CALIBRATION batches only ----
noise1 = np.mean([s[-1] for _, s in c1], axis=0) * NOISE_PCT
noise2 = np.mean([s[-1] for _, s in c2], axis=0) * NOISE_PCT
print("noise sd per species (0.35% of calibration mean final conc.)")
print("  stage 1 (A,B,C):", np.round(noise1, 4))
print("  stage 2 (A-F)  :", np.round(noise2, 4), "\n")

def discretize(stage, noise, ndim):
    out = []
    for tv, sol in stage:
        tk = np.arange(0, tv[-1], SAMPLE_INTERVAL)
        arr = np.zeros((len(tk), ndim))
        for j in range(ndim):
            interp = np.interp(tk, tv, sol[:, j])
            arr[:, j] = np.maximum(0, interp + np.random.normal(0, noise[j], len(tk)))
        out.append((tk, arr))
    return out

d1 = {k: discretize(v, noise1, 3) for k, v in [('cal', c1), ('wit', w1), ('ext', e1)]}
d2 = {k: discretize(v, noise2, 6) for k, v in [('cal', c2), ('wit', w2), ('ext', e2)]}
P  = {'cal': cp, 'wit': wp, 'ext': ep}

MIN_PTS = min(len(a) for k in d2 for _, a in d2[k])
print(f"common stage-2 timepoints across all three sets: {MIN_PTS}")
for k in d2:
    lens = [len(a) for _, a in d2[k]]
    print(f"  {k}: trajectory lengths {min(lens)}-{max(lens)}")
print()

def build(key):
    par, s1d, s2d = P[key], d1[key], d2[key]
    x1 = pd.DataFrame(
        [[p['CA0_sample'], p['t1_seconds']/60, p['T1'], *s1d[i][1][-1]] for i, p in enumerate(par)],
        columns=['CA0', 't1', 'T1', 'CA1_final', 'CB1_final', 'CC1_final'])
    rows = []
    for i, p in enumerate(par):
        blk = s2d[i][1][:MIN_PTS, :]
        rows.append([p['CD0_sample'], p['t2_seconds']/60, p['T2']] + blk.T.flatten().tolist())
    cols = ['CD0', 't2_final', 'T2'] + [f'{c}_t{t+1}' for c in SPECIES for t in range(MIN_PTS)]
    x2 = pd.DataFrame(rows, columns=cols)
    y = pd.DataFrame([[s2d[i][1][-1][4]/s2d[i][1][-1].sum()*100] for i in range(len(par))],
                     columns=['E_purity'])
    return x1, x2, y

print("=== generated sets ===")
for key, fn in [('cal', 'RG_calibration.xlsx'), ('wit', 'RG_within.xlsx'), ('ext', 'RG_extrap.xlsx')]:
    x1, x2, y = build(key)
    with pd.ExcelWriter(fn) as w:
        x1.to_excel(w, sheet_name='X1', index=False)
        x2.to_excel(w, sheet_name='X2', index=False)
        y.to_excel(w, sheet_name='Y', index=False)
    print(f"{fn:22s} X1{x1.shape} X2{x2.shape} Y{y.shape}  "
          f"T2 {x2.T2.min():.1f}-{x2.T2.max():.1f}  "
          f"Y {y.E_purity.mean():.2f}+/-{y.E_purity.std(ddof=1):.2f} "
          f"[{y.E_purity.min():.2f},{y.E_purity.max():.2f}]")

print("\n=== sanity checks ===")
xc = pd.read_excel('RG_calibration.xlsx', sheet_name='X2')
xw = pd.read_excel('RG_within.xlsx', sheet_name='X2')
xe = pd.read_excel('RG_extrap.xlsx', sheet_name='X2')
print(f"column alignment cal/wit/ext: {list(xc.columns)==list(xw.columns)==list(xe.columns)}")
print(f"calibration T2 percentiles : {np.round(np.percentile(xc.T2,[0,25,50,75,100]),1)}")
print(f"within      T2 percentiles : {np.round(np.percentile(xw.T2,[0,25,50,75,100]),1)}")
print(f"extrap      T2 percentiles : {np.round(np.percentile(xe.T2,[0,25,50,75,100]),1)}")
print(f"within set inside calibration T2 range: "
      f"{((xw.T2>=xc.T2.min())&(xw.T2<=xc.T2.max())).mean()*100:.0f}%")
print(f"extrap set outside calibration T2 range: {(xe.T2>xc.T2.max()).mean()*100:.0f}%")
for fn in ['RG_calibration.xlsx', 'RG_within.xlsx', 'RG_extrap.xlsx']:
    d = pd.read_excel(fn, sheet_name='X2')
    print(f"  {fn:22s} finite={np.isfinite(d.values).all()}  min={d.values.min():.3f}  max={d.values.max():.1f}")
print(f"\nTotal {time.time()-t0:.1f}s")

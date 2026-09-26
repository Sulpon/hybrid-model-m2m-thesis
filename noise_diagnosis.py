"""
TASK 3/4 diagnostics: determine (a) whether X2-M2 residual variance grows
with elapsed time within stage 2 (process/Brownian-noise signature) vs stays
flat (measurement-noise signature), and (b) whether residual magnitude
scales with each batch's OWN concentration (multiplicative/individual-based
noise) or is roughly constant across batches regardless of their own value
(population-mean-based/homoscedastic noise) -- read-only, no files modified.
"""
import numpy as np
import pandas as pd

species = ['A', 'B', 'C', 'D', 'E', 'F']
X2 = pd.read_excel('X2.xlsx')
M2 = pd.read_excel('M2.xlsx')
n_pts = X2.shape[1] // 6

print("===== (1) DOES RESIDUAL VARIANCE GROW ACROSS THE 7 SHARED TIMEPOINTS? =====")
print("(all 100 batches share the SAME 7 timepoints, 0..tau, after truncation -- ")
print(" a growing variance across t1..t7 signals accumulating PROCESS/Brownian noise;")
print(" a flat variance signals i.i.d. per-observation MEASUREMENT noise dominating)\n")
print(f"{'species':<8}" + "".join(f"t{t+1:<9}" for t in range(n_pts)))
resid_var_by_t = {}
for sp in species:
    resids = []
    for t in range(1, n_pts + 1):
        r = (X2[f'{sp}_{t}'] - M2[f'{sp}_fit_{t}']).values
        resids.append(r)
    resid_var_by_t[sp] = np.array([r.var(ddof=1) for r in resids])
    print(f"{sp:<8}" + "".join(f"{v:<9.2f}" for v in resid_var_by_t[sp]))

print("\nratio of variance at t7 (final) to t1 (first), per species:")
for sp in species:
    v = resid_var_by_t[sp]
    print(f"  {sp}: var(t1)={v[0]:.3f}  var(t{n_pts})={v[-1]:.3f}  ratio={v[-1]/v[0]:.2f}")

print("\n===== (2) DOES RESIDUAL MAGNITUDE SCALE WITH EACH BATCH'S OWN CONCENTRATION? =====")
print("(tests multiplicative/individual-concentration-based noise (formula 3) vs")
print(" population-mean-based, homoscedastic noise (formula 1/2), at the final timepoint)\n")
print(f"{'species':<8}{'corr(|resid|, own M2 value)':>30}{'corr(resid^2, own M2 value)':>30}")
for sp in species:
    r = (X2[f'{sp}_{n_pts}'] - M2[f'{sp}_fit_{n_pts}']).values
    own = M2[f'{sp}_fit_{n_pts}'].values
    c1 = np.corrcoef(np.abs(r), own)[0, 1]
    c2 = np.corrcoef(r**2, own)[0, 1]
    print(f"{sp:<8}{c1:>30.3f}{c2:>30.3f}")

print("\n===== (3) IMPLIED CV IF NOISE WERE PURELY MULTIPLICATIVE (sigma = k * own conc.) =====")
print("per-batch |residual| / own concentration, at final timepoint -- if this ratio is")
print("roughly CONSTANT across batches (low CV of the ratio itself), that supports a")
print("multiplicative/individual-based model; if it varies wildly, it does not.\n")
print(f"{'species':<8}{'mean(|resid|/own)':>20}{'sd(|resid|/own)':>18}{'CV of that ratio':>18}")
for sp in species:
    r = (X2[f'{sp}_{n_pts}'] - M2[f'{sp}_fit_{n_pts}']).values
    own = M2[f'{sp}_fit_{n_pts}'].values
    ratio = np.abs(r) / np.maximum(own, 1e-6)
    print(f"{sp:<8}{ratio.mean():>20.4f}{ratio.std(ddof=1):>18.4f}{100*ratio.std(ddof=1)/ratio.mean():>17.1f}%")

print("\n===== (4) POPULATION-MEAN-BASED CV IMPLIED BY THE DATA, PER SPECIES =====")
print("residual sd (final timepoint) as a % of that species' POPULATION MEAN concentration")
print("-- this is what noise_percentage would need to be, under KD_fit.py's own formula\n")
print(f"{'species':<8}{'resid sd':>12}{'pop mean (M2)':>16}{'implied noise_percentage':>26}")
for sp in species:
    r = (X2[f'{sp}_{n_pts}'] - M2[f'{sp}_fit_{n_pts}']).values
    popmean = M2[f'{sp}_fit_{n_pts}'].values.mean()
    print(f"{sp:<8}{r.std(ddof=1):>12.3f}{popmean:>16.2f}{100*r.std(ddof=1)/popmean:>25.2f}%")

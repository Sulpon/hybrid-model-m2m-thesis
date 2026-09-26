"""
Fast harness: reuses Data Generation.py's SDE core (post sigma_A fix) to
produce (1) a 100-batch calibration set (T2 ~ U(330,370), matching the
paper's Table 3 training range) and (2) a 25-batch test set (T2 ~ U(362,382)),
then splits the test set into within-domain (T2<=370) vs extrapolation
(T2>370). Headless (Agg backend), no plots, no curve_fit. Saves outputs to
xlsx files with a DG_ prefix so nothing collides with the author's
X1.xlsx/X2.xlsx/M2.xlsx/Y.xlsx.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import sdeint
import time

# ---- exact core copied from Data Generation.py (post sigma_A fix) ----
V1, V2 = 0.1, 0.2
CA0, CD0 = 2000.0, 1900.0
A1, A2 = 28, 40
A3, A4 = 10, 20
Ea1_mean, Ea2_mean, Ea3_mean, Ea4_mean = 20000, 30000, 50000, 55000
CV = 0.01
R = 8.314
opt_sigma1 = [0.025, 0.9, 0.08]
opt_sigma2 = [0.25, 0.03, 0.022, 0.07, 0.54, 0.04]


def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    return np.array([-k1 * CA, k1 * CA - k2 * CB, k2 * CB])


def stage1_noise(y, t):
    return np.diag(opt_sigma1)


def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    return np.array([0, -k3 * CB * CD ** 2, -k4 * CC * CD ** 2,
                      -2 * k3 * CB * CD ** 2 - 2 * k4 * CC * CD ** 2,
                      k3 * CB * CD ** 2, k4 * CC * CD ** 2])


def stage2_noise(y, t):
    return np.diag(opt_sigma2)


def one_sde_simulation(data_type="training"):
    Ea1 = np.random.normal(Ea1_mean, CV * Ea1_mean)
    Ea2 = np.random.normal(Ea2_mean, CV * Ea2_mean)
    Ea3 = np.random.normal(Ea3_mean, CV * Ea3_mean)
    Ea4 = np.random.normal(Ea4_mean, CV * Ea4_mean)
    T1 = np.random.uniform(290, 310)
    T2 = np.random.uniform(330, 370) if data_type == "training" else np.random.uniform(362, 382)
    t1_seconds = np.random.uniform(350, 650)
    t2_seconds = np.random.uniform(100, 400)
    k1 = A1 * np.exp(-Ea1 / (R * T1)); k2 = A2 * np.exp(-Ea2 / (R * T1))
    k3 = A3 * np.exp(-Ea3 / (R * T2)); k4 = A4 * np.exp(-Ea4 / (R * T2))
    CA0_sample = np.random.normal(CA0, 0.05 * CA0)
    CD0_sample = np.random.normal(CD0, 0.05 * CD0)
    t1v = np.linspace(0, t1_seconds, 1000)
    y0_1 = np.array([CA0_sample, 0.0, 0.0])
    sol1 = sdeint.itoint(lambda y, t: stage1_drift(y, t, k1, k2), lambda y, t: stage1_noise(y, t), y0_1, t1v)
    t2v = np.linspace(0, t2_seconds, 1000)
    CA1f, CB1f, CC1f = sol1[-1]
    dilution = V1 / V2
    y0_2 = np.array([CA1f * dilution, CB1f * dilution, CC1f * dilution, CD0_sample, 0.0, 0.0])
    sol2 = sdeint.itoint(lambda y, t: stage2_drift(y, t, k3, k4), lambda y, t: stage2_noise(y, t), y0_2, t2v)
    params = dict(CA0_sample=CA0_sample, CD0_sample=CD0_sample, T1=T1, T2=T2,
                  t1_seconds=t1_seconds, t2_seconds=t2_seconds, Ea1=Ea1, Ea2=Ea2, Ea3=Ea3, Ea4=Ea4,
                  data_type=data_type)
    return (t1v / 60, sol1), (t2v / 60, sol2), params


def discrete_trajectories(all_stage1, all_stage2, noise_percentage=0.0035, sample_interval=0.25):
    final_c1 = [sol1[-1] for _, sol1 in all_stage1]
    mean1 = np.mean(final_c1, axis=0)
    noise1 = mean1 * noise_percentage
    discrete_stage1 = []
    for t1v, sol1 in all_stage1:
        t1k = np.arange(0, t1v[-1], sample_interval)
        out = np.zeros((len(t1k), 3))
        for j in range(3):
            interp = np.interp(t1k, t1v, sol1[:, j])
            out[:, j] = np.maximum(0, interp + np.random.normal(0, noise1[j], len(t1k)))
        discrete_stage1.append((t1k, out))

    final_c2 = [sol2[-1] for _, sol2 in all_stage2]
    mean2 = np.mean(final_c2, axis=0)
    noise2 = mean2 * noise_percentage
    discrete_stage2 = []
    for t2v, sol2 in all_stage2:
        t2k = np.arange(0, t2v[-1], sample_interval)
        out = np.zeros((len(t2k), 6))
        for j in range(6):
            interp = np.interp(t2k, t2v, sol2[:, j])
            out[:, j] = np.maximum(0, interp + np.random.normal(0, noise2[j], len(t2k)))
        discrete_stage2.append((t2k, out))
    return discrete_stage1, discrete_stage2


# ---------------------------------------------------------------------------
t0 = time.time()
GLOBAL_SEED = 10
np.random.seed(GLOBAL_SEED)
N_calibration = 100
N_test = 25

all_stage1, all_stage2, all_params = [], [], []
for i in range(N_calibration):
    (t1v, sol1), (t2v, sol2), params = one_sde_simulation("training")
    all_stage1.append((t1v, sol1)); all_stage2.append((t2v, sol2)); all_params.append(params)
for i in range(N_test):
    (t1v, sol1), (t2v, sol2), params = one_sde_simulation("testing")
    all_stage1.append((t1v, sol1)); all_stage2.append((t2v, sol2)); all_params.append(params)

print(f"Simulated {N_calibration + N_test} batches in {time.time()-t0:.1f}s")

disc1, disc2 = discrete_trajectories(all_stage1, all_stage2)

data_type = np.array([p['data_type'] for p in all_params])
T2v = np.array([p['T2'] for p in all_params])
CD0v = np.array([p['CD0_sample'] for p in all_params])
CA0v = np.array([p['CA0_sample'] for p in all_params])
T1v = np.array([p['T1'] for p in all_params])
t1v_ = np.array([p['t1_seconds'] for p in all_params])
t2v_ = np.array([p['t2_seconds'] for p in all_params])

cal_idx = np.where(data_type == "training")[0]
test_idx = np.where(data_type == "testing")[0]

# ---------------- sanity checks ----------------
print("\n===== SANITY CHECKS =====")
any_nan = False
neg_conc = False
huge_conc = False
for idx in range(len(disc2)):
    arr = disc2[idx][1]
    if np.isnan(arr).any():
        any_nan = True
    if (arr < -1e-8).any():
        neg_conc = True
    if (arr > 1e6).any():
        huge_conc = True
print(f"Any NaNs in stage-2 discretized data: {any_nan}")
print(f"Any negative concentrations (before clipping floor): {neg_conc}")
print(f"Any absurdly large concentrations (>1e6): {huge_conc}")
print(f"Calibration batches: {len(cal_idx)}  (expected 100)")
print(f"Test batches: {len(test_idx)}  (expected 25)")
print(f"Calibration T2 range: [{T2v[cal_idx].min():.2f}, {T2v[cal_idx].max():.2f}] K  (expected ~[330,370])")
print(f"Test T2 range: [{T2v[test_idx].min():.2f}, {T2v[test_idx].max():.2f}] K  (expected ~[362,382])")

within_idx = test_idx[T2v[test_idx] <= 370]
extrap_idx = test_idx[T2v[test_idx] > 370]
print(f"Test split -> within-domain (T2<=370): {len(within_idx)}   extrapolation (T2>370): {len(extrap_idx)}")

# final concentration summary (species A-F) for calibration set
final_c2_cal = np.array([disc2[i][1][-1] for i in cal_idx])
species = ['A', 'B', 'C', 'D', 'E', 'F']
print("\nFinal-timepoint concentration summary (calibration set):")
for j, sp in enumerate(species):
    col = final_c2_cal[:, j]
    print(f"  {sp}: mean={col.mean():8.2f}  sd={col.std(ddof=1):8.2f}  min={col.min():8.2f}  max={col.max():8.2f}")

purity_cal = final_c2_cal[:, 4] / final_c2_cal.sum(axis=1) * 100
print(f"\nFinal purity of E (calibration set): mean={purity_cal.mean():.2f}%  sd={purity_cal.std(ddof=1):.2f}%  "
      f"min={purity_cal.min():.2f}%  max={purity_cal.max():.2f}%")

# ---------------- build & save output matrices ----------------
def build_X1(idx_list):
    rows = []
    for i in idx_list:
        t1k, sol1d = disc1[i]
        CAf, CBf, CCf = sol1d[-1]
        rows.append([CA0v[i], t1v_[i] / 60, T1v[i], CAf, CBf, CCf])
    return pd.DataFrame(rows, columns=["CA0", "t1", "T1", "CA1_final", "CB1_final", "CC1_final"])


def build_Y(idx_list):
    rows = []
    for i in idx_list:
        _, sol2d = disc2[i]
        CAf, CBf, CCf, CDf, CEf, CFf = sol2d[-1]
        rows.append(CEf / (CAf + CBf + CCf + CDf + CEf + CFf) * 100)
    return pd.DataFrame(rows, columns=["E_purity"])


def build_X2(idx_list):
    min_pts = min(len(disc2[i][1]) for i in idx_list)
    rows = []
    for i in idx_list:
        t2k, sol2d = disc2[i]
        block = sol2d[:min_pts, :]  # truncate to shortest trajectory in this split
        row = [CD0v[i], t2v_[i] / 60, T2v[i]] + block.T.flatten().tolist()
        rows.append(row)
    cols = ['CD0', 't2_final', 'T2']
    for c in species:
        for t in range(min_pts):
            cols.append(f"{c}_t{t+1}")
    return pd.DataFrame(rows, columns=cols), min_pts


X1_cal = build_X1(cal_idx)
Y_cal = build_Y(cal_idx)
X2_cal, min_pts_cal = build_X2(cal_idx)

X1_within = build_X1(within_idx)
Y_within = build_Y(within_idx)
X2_within, min_pts_w = build_X2(within_idx)

X1_extrap = build_X1(extrap_idx)
Y_extrap = build_Y(extrap_idx)
X2_extrap, min_pts_e = build_X2(extrap_idx)

with pd.ExcelWriter("DG_calibration.xlsx") as writer:
    X1_cal.to_excel(writer, sheet_name="X1", index=False)
    X2_cal.to_excel(writer, sheet_name="X2", index=False)
    Y_cal.to_excel(writer, sheet_name="Y", index=False)

with pd.ExcelWriter("DG_test_within_domain.xlsx") as writer:
    X1_within.to_excel(writer, sheet_name="X1", index=False)
    X2_within.to_excel(writer, sheet_name="X2", index=False)
    Y_within.to_excel(writer, sheet_name="Y", index=False)

with pd.ExcelWriter("DG_test_extrapolation.xlsx") as writer:
    X1_extrap.to_excel(writer, sheet_name="X1", index=False)
    X2_extrap.to_excel(writer, sheet_name="X2", index=False)
    Y_extrap.to_excel(writer, sheet_name="Y", index=False)

print(f"\nSaved: DG_calibration.xlsx (100 batches, {min_pts_cal} stage-2 timepoints)")
print(f"Saved: DG_test_within_domain.xlsx ({len(within_idx)} batches, {min_pts_w} stage-2 timepoints)")
print(f"Saved: DG_test_extrapolation.xlsx ({len(extrap_idx)} batches, {min_pts_e} stage-2 timepoints)")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")

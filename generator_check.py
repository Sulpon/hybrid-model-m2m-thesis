"""
TASK 3 implementation: generate a fresh 100-batch calibration dataset using
the logic in KD_fit.py (the most complete / final iteration of the user's
own generator, identified in TASK 1), and compare it STATISTICALLY (not
row-by-row) against the author-provided X1.xlsx/X2.xlsx/M2.xlsx/Y.xlsx.

This is a verbatim copy of KD_fit.py's simulation core (stage1_drift,
stage1_noise, stage2_drift, stage2_noise, one_sde_simulation, and the
discretization / unfolding logic), with the plotting and curve_fit
(misspecified-model) sections removed, so it can run headlessly and be
compared quantitatively. Does not modify or overwrite any existing file --
writes nothing back to X1.xlsx/X2.xlsx/M2.xlsx/Y.xlsx.
"""
import numpy as np
import pandas as pd
import sdeint
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

R = 8.314
V1, V2 = 0.1, 0.2
CA0_mean, CD0_mean = 2000.0, 1900.0
A1, A2 = 28, 40
A3, A4 = 10, 20
Ea1_mean, Ea2_mean, Ea3_mean, Ea4_mean = 20000, 30000, 50000, 55000
CV = 0.01
opt_sigma1 = [0.025, 0.9, 0.08]
opt_sigma2 = [0.006, 0.03, 0.022, 0.07, 0.54, 0.04]


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
    CA0_sample = np.random.normal(CA0_mean, 0.05 * CA0_mean)   # <-- KD_fit.py's literal value (paper states 1% CV, see TASK 4)
    CD0_sample = np.random.normal(CD0_mean, 0.05 * CD0_mean)
    t1v = np.linspace(0, t1_seconds, 1000)
    y0_1 = np.array([CA0_sample, 0.0, 0.0])
    sol1 = sdeint.itoint(lambda y, t: stage1_drift(y, t, k1, k2), lambda y, t: stage1_noise(y, t), y0_1, t1v)
    t2v = np.linspace(0, t2_seconds, 1000)
    CA1f, CB1f, CC1f = sol1[-1]
    dilution = V1 / V2
    y0_2 = np.array([CA1f * dilution, CB1f * dilution, CC1f * dilution, CD0_sample, 0.0, 0.0])
    sol2 = sdeint.itoint(lambda y, t: stage2_drift(y, t, k3, k4), lambda y, t: stage2_noise(y, t), y0_2, t2v)
    params = dict(CA0_sample=CA0_sample, CD0_sample=CD0_sample, T1=T1, T2=T2,
                  t1_seconds=t1_seconds, t2_seconds=t2_seconds, Ea1=Ea1, Ea2=Ea2, Ea3=Ea3, Ea4=Ea4)
    return (t1v / 60, sol1), (t2v / 60, sol2), params


def discretize(all_stage2, noise_percentage=0.0035, sample_interval=0.25):
    final_c2 = [sol2[-1] for _, sol2 in all_stage2]
    mean2 = np.mean(final_c2, axis=0)
    noise2 = mean2 * noise_percentage
    disc = []
    for t2v, sol2 in all_stage2:
        t2k = np.arange(0, t2v[-1], sample_interval)
        out = np.zeros((len(t2k), 6))
        for j in range(6):
            interp = np.interp(t2k, t2v, sol2[:, j])
            out[:, j] = np.maximum(0, interp + np.random.normal(0, noise2[j], len(t2k)))
        disc.append((t2k, out))
    return disc


def stage2_correct(y, t, T2, Ea3, Ea4, A3v, A4v):
    CA, CB, CC, CD, CE, CF = y
    k3 = A3v * np.exp(-Ea3 / (R * T2)); k4 = A4v * np.exp(-Ea4 / (R * T2))
    return [0, -k3 * CB * CD ** 2, -k4 * CC * CD ** 2,
            -2 * k3 * CB * CD ** 2 - 2 * k4 * CC * CD ** 2, k3 * CB * CD ** 2, k4 * CC * CD ** 2]


if __name__ == '__main__':
    GLOBAL_SEED = 10
    np.random.seed(GLOBAL_SEED)
    N = 100
    all_stage1, all_stage2, all_params = [], [], []
    for i in range(N):
        (t1v, sol1), (t2v, sol2), params = one_sde_simulation("training")
        all_stage1.append((t1v, sol1)); all_stage2.append((t2v, sol2)); all_params.append(params)

    disc2 = discretize(all_stage2)
    min_pts = min(len(tk) for tk, _ in disc2)
    print(f"Generated {N} batches. Min stage-2 discretized points across batches: {min_pts} "
          f"(author X2.xlsx has 42/6 = 7 timepoints per species)")

    CA0v = np.array([p['CA0_sample'] for p in all_params])
    CD0v = np.array([p['CD0_sample'] for p in all_params])
    T1v = np.array([p['T1'] for p in all_params])
    T2v = np.array([p['T2'] for p in all_params])
    t1v_ = np.array([p['t1_seconds'] for p in all_params])
    t2v_ = np.array([p['t2_seconds'] for p in all_params])

    Yg = np.array([disc2[i][1][min_pts - 1, 4] / disc2[i][1][min_pts - 1].sum() * 100 for i in range(N)])

    M2g = np.zeros((N, min_pts, 6))
    for i, (t1v_i, sol1) in enumerate(all_stage1):
        CA1f, CB1f, CC1f = sol1[-1]
        dilution = V1 / V2
        y0_2 = [CA1f * dilution, CB1f * dilution, CC1f * dilution, all_params[i]['CD0_sample'], 0.0, 0.0]
        tk = disc2[i][0][:min_pts]
        sol = odeint(stage2_correct, y0_2, tk * 60, args=(all_params[i]['T2'], Ea3_mean, Ea4_mean, A3, A4))
        M2g[i] = sol

    # ================= load author files =================
    X1a = pd.read_excel('X1.xlsx'); X2a = pd.read_excel('X2.xlsx')
    M2a = pd.read_excel('M2.xlsx'); Ya = pd.read_excel('Y.xlsx')

    print("\n===== DIMENSIONS =====")
    print(f"author X1 {X1a.shape}  X2 {X2a.shape}  M2 {M2a.shape}  Y {Ya.shape}")
    print(f"my gen   X1 (100,6)   X2 (100,{6*min_pts})   M2 (100,{6*min_pts}+3)   Y (100,1)")

    print("\n===== OPERATING CONDITIONS: author (from M2.xlsx/X1.xlsx) vs my generator =====")
    print(f"{'var':<8}{'author mean':>14}{'author sd':>12}{'author min':>12}{'author max':>12}   |  "
          f"{'gen mean':>10}{'gen sd':>10}{'gen min':>10}{'gen max':>10}")
    for name, av, gv in [('CA0', X1a['CA0'].values, CA0v), ('T1', X1a['T1'].values, T1v),
                         ('t1', X1a['t1'].values, t1v_), ('T2', M2a['T2'].values, T2v),
                         ('t2', M2a['t2'].values * 60, t2v_), ('D0', M2a['D0'].values, CD0v)]:
        print(f"{name:<8}{av.mean():>14.2f}{av.std(ddof=1):>12.2f}{av.min():>12.2f}{av.max():>12.2f}   |  "
              f"{gv.mean():>10.2f}{gv.std(ddof=1):>10.2f}{gv.min():>10.2f}{gv.max():>10.2f}")

    print(f"\nCA0 realized CV: author={100*X1a['CA0'].std(ddof=1)/X1a['CA0'].mean():.2f}%  "
          f"mine={100*CA0v.std(ddof=1)/CA0v.mean():.2f}%  (sampling sd used: author target 1% per paper Table 3, "
          f"mine hardcodes 0.05*mean = 5%)")
    print(f"D0(CD0) realized CV: author={100*M2a['D0'].std(ddof=1)/M2a['D0'].mean():.2f}%  "
          f"mine={100*CD0v.std(ddof=1)/CD0v.mean():.2f}%")

    print("\n===== RESPONSE Y (purity of E, %) =====")
    print(f"author: mean={Ya['E_pur'].mean():.2f}  sd={Ya['E_pur'].std(ddof=1):.2f}  "
          f"min={Ya['E_pur'].min():.2f}  max={Ya['E_pur'].max():.2f}")
    print(f"mine:   mean={Yg.mean():.2f}  sd={Yg.std(ddof=1):.2f}  min={Yg.min():.2f}  max={Yg.max():.2f}")

    print("\n===== FINAL-TIMEPOINT SPECIES CONCENTRATIONS: author X2 vs my generated X2 (mean +/- sd) =====")
    species = ['A', 'B', 'C', 'D', 'E', 'F']
    n_author_pts = X2a.shape[1] // 6
    for j, sp in enumerate(species):
        a_final = X2a[f'{sp}_{n_author_pts}'].values
        g_final = M2g[:, min_pts - 1, j] if False else None
    # measured (noisy) final point comparison uses X2, not M2g (M2g is the deterministic mechanistic block)
    X2g_final = np.array([disc2[i][1][min_pts - 1] for i in range(N)])
    for j, sp in enumerate(species):
        a_final = X2a[f'{sp}_{n_author_pts}'].values
        g_final = X2g_final[:, j]
        print(f"{sp}: author {a_final.mean():9.2f} +/- {a_final.std(ddof=1):7.2f}   "
              f"mine {g_final.mean():9.2f} +/- {g_final.std(ddof=1):7.2f}")

    print("\n===== MEASUREMENT NOISE MAGNITUDE: residual sd of (X2 - M2) at final point, per species =====")
    M2a_final = np.array([M2a[f'{sp}_fit_{n_author_pts}'].values for sp in species]).T
    X2a_final = np.array([X2a[f'{sp}_{n_author_pts}'].values for sp in species]).T
    resid_author = (X2a_final - M2a_final)
    resid_mine = (X2g_final - M2g[:, min_pts - 1, :])
    for j, sp in enumerate(species):
        print(f"{sp}: author resid sd={resid_author[:, j].std(ddof=1):8.3f}   "
              f"mine resid sd={resid_mine[:, j].std(ddof=1):8.3f}")

    print("\nSaved no files (comparison only).")

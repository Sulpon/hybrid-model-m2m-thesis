"""
Fit the seven initial molar numbers to reproduce HyMech Fig. 7(a) and 7(b).

The charge derived by inverting Bauer's Experiment #1 design variables does not
reproduce the published trajectories: it gives a monotonic nC rising to 0.81,
whereas Fig. 7 shows nC peaking near 0.23, collapsing as the allophanate
reaction consumes it, and recovering only when more butanol is fed. In the
published system nD exceeds nC for most of the batch.

Rather than assume a charge, it is estimated here from the figure itself. The
kinetic parameters are held at their published nominal values (Table A.2) and
the input profiles at the explicit HyMech Fig. 7(c) patterns, so the charge is
the only free quantity.

Targets are anchor points read off the published symbols (the process data, not
the FP lines). Residuals are relative, so the three species contribute on
comparable terms despite spanning three orders of magnitude.
"""
import numpy as np
from scipy.optimize import least_squares
import warnings, json, time
warnings.filterwarnings('ignore')
import urethane_case_study as U

DURATION = U.BAUER_BATCH_H          # 80 h

# ---------------------------------------------------------------- targets
# (time [h], value) read from Fig. 7; index 0 = nC, 1 = nD, 2 = nE
TARGETS = {
    0: {  # Fig. 7(a), batch 1
        0: [(5, 0.230), (10, 0.120), (20, 0.125), (40, 0.155), (80, 0.185)],
        1: [(10, 0.210), (40, 0.190), (80, 0.195)],
        2: [(20, 3.1e-4), (80, 3.4e-4)],
    },
    1: {  # Fig. 7(b), batch 2
        0: [(5, 0.245), (20, 0.020), (40, 0.110), (80, 0.240)],
        1: [(10, 0.330), (40, 0.280), (80, 0.270)],
        2: [(20, 4.7e-3), (80, 5.0e-3)],
    },
}

KEYS = ['n0_A', 'n0_B', 'n0_S', 'nv1_A', 'nv1_S', 'nv2_B', 'nv2_S']
# start from the Bauer Experiment #1 charge
START = U.design_to_moles(U.BAUER_EXPERIMENTS[1])
Z0 = np.log10([max(START[k], 1e-3) for k in KEYS])
LO = np.log10(np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3]))
HI = np.log10(np.array([5.0, 5.0, 20.0, 10.0, 20.0, 10.0, 20.0]))

fns = [U.hymech_inputs(s) for s in U.HYMECH_BATCHES]
t_grid = np.arange(0.0, DURATION + 1e-9, 0.25)

def charge(z):
    return dict(zip(KEYS, 10.0**np.asarray(z)))

def run(ic):
    out = []
    for f in fns:
        try:
            out.append(U.simulate(U.ground_truth_rhs, f, ic, t_grid))
        except Exception:
            out.append(np.full((len(t_grid), 3), np.nan))
    return out

def residuals(z):
    ic = charge(z)
    sims = run(ic)
    r = []
    for b, sim in enumerate(sims):
        if not np.isfinite(sim).all():
            return np.full(24, 10.0)
        for k, pts in TARGETS[b].items():
            for tt, target in pts:
                j = int(np.argmin(np.abs(t_grid - tt)))
                r.append((sim[j, k] - target)/abs(target))
    return np.asarray(r)

if __name__ == '__main__':
    t0 = time.time()
    print('start (Bauer Experiment #1):')
    for k in KEYS:
        print(f'   {k:7s} {START[k]:9.5f} mol')
    print(f'   initial cost {0.5*np.sum(residuals(Z0)**2):.4f}\n', flush=True)

    best, best_cost = None, np.inf
    rng = np.random.default_rng(0)
    for trial in range(6):
        z_init = Z0 if trial == 0 else Z0 + rng.normal(0, 0.45, len(KEYS))
        z_init = np.clip(z_init, LO + 1e-6, HI - 1e-6)
        try:
            res = least_squares(residuals, z_init, bounds=(LO, HI),
                                xtol=1e-12, ftol=1e-12, x_scale='jac', max_nfev=600)
        except Exception as e:
            print(f'  trial {trial}: failed ({e})'); continue
        print(f'  trial {trial}: cost {res.cost:.5f}   ({time.time()-t0:.0f}s)', flush=True)
        if res.cost < best_cost:
            best, best_cost = res.x, res.cost

    ic = charge(best)
    print(f'\nfitted charge (cost {best_cost:.5f}):')
    for k in KEYS:
        print(f'   {k:7s} {ic[k]:9.5f} mol      (Bauer #1: {START[k]:9.5f})')
    tA = ic['n0_A'] + ic['nv1_A']; tB = ic['n0_B'] + ic['nv2_B']
    print(f'   total A {tA:.4f} mol   total B {tB:.4f} mol   B/A {tB/tA:.4f}')
    print(f'   (Bauer #1: total A {START["n0_A"]+START["nv1_A"]:.4f}, '
          f'B {START["n0_B"]+START["nv2_B"]:.4f}, ratio '
          f'{(START["n0_B"]+START["nv2_B"])/(START["n0_A"]+START["nv1_A"]):.4f})')

    sims = run(ic)
    print('\nfit against the digitised targets')
    print(f'{"batch":7s} {"sp":3s} {"t[h]":>5s} {"target":>10s} {"fitted":>10s} {"rel err":>9s}')
    for b, sim in enumerate(sims):
        for k, pts in TARGETS[b].items():
            for tt, target in pts:
                j = int(np.argmin(np.abs(t_grid - tt)))
                v = sim[j, k]
                print(f'{U.HYMECH_BATCHES[b]["name"]:7s} {"CDE"[k]:3s} {tt:5.0f} '
                      f'{target:10.4g} {v:10.4g} {100*(v-target)/target:8.1f}%')
    json.dump({k: float(v) for k, v in ic.items()}, open('urethane_fitted_charge.json', 'w'), indent=2)
    print(f'\nSaved urethane_fitted_charge.json   ({time.time()-t0:.0f}s)')

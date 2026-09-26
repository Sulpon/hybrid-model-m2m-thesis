"""
Recover the Batch_1 feed programme fv1(t), fv2(t) from the authors' figures.

Everything else is already pinned: the kinetic model and parameters are the
authors' own (urethane_authors_replicate), the temperature profile is read
directly off their "Temperature (T)" panel, and the charge follows from the
t=0 readings plus the volume closure. That leaves the two feed ramps, which
the notebooks never plot, as the only unknowns.

They are fitted here to the anchors digitised from the authors' stored
"Baseline Simulation Results" figure -- all six panels, so the fit is
constrained by nA, nB, nC, nD, nE and V simultaneously rather than by the
three measured species alone.

Both ramps are piecewise linear on fixed node times, non-decreasing, and
reach exactly 1 by the end of the batch.
"""
import numpy as np
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')
import urethane_authors_replicate as U

FV1_T = [0.0, 1.5, 3.0, 5.0, 7.0, 50.0, 57.0, 80.0]
FV2_T = [0.0, 5.0, 10.0, 22.0, 30.0, 45.0, 50.0, 57.0, 80.0]
N1, N2 = 4, 6           # free nodes (indices 1..4 of FV1_T, 1..6 of FV2_T)
FLOOR = 2e-3            # absolute floor in the relative residual
SMOOTH = 0.1            # weight on the curvature of the feed increments

# Anchors alone do not pin the shape BETWEEN them: an unregularised fit puts a
# plateau then a jump into fv1, which makes nA and nB spike twice. The authors'
# nB falls monotonically over 0-8 h and their nC has a single clean peak, so a
# curvature penalty on the increments is added and the early decay of nB is
# anchored explicitly. Both are read from the same stored figure.
EXTRA = {
    'A': [(2, 0.040), (3, 0.036)],
    'B': [(1, 0.115), (2, 0.075), (3, 0.045), (5, 0.012)],
    'C': [(3, 0.135), (7, 0.200), (9, 0.118)],
}

t_grid = np.arange(0.0, 80.0 + 1e-9, 0.25)


def build(z):
    """Map the free vector to monotone ramps ending at 1."""
    d1 = np.abs(z[:N1]); d2 = np.abs(z[N1:N1+N2])
    v1 = np.cumsum(d1); v2 = np.cumsum(d2)
    v1 = np.clip(v1, 0.0, 1.0); v2 = np.clip(v2, 0.0, 1.0)
    fv1_v = [0.0] + list(v1) + [v1[-1], 1.0, 1.0]   # hold at 50 h, then to 1
    fv2_v = [0.0] + list(v2) + [1.0, 1.0]           # last free node is at 50 h
    assert len(fv1_v) == len(FV1_T) and len(fv2_v) == len(FV2_T)
    return dict(name='Batch_1', T_t=U.BATCH1_INPUTS['T_t'], T_K=U.BATCH1_INPUTS['T_K'],
                fv1_t=FV1_T, fv1_v=fv1_v, fv2_t=FV2_T, fv2_v=fv2_v)


ANCH = {k: list(v) for k, v in U.ANCHORS.items()}
for k, v in EXTRA.items():
    ANCH[k] = sorted(ANCH[k] + v)
NRES = sum(len(p) for p in ANCH.values())


def residuals(z):
    spec = build(z)
    try:
        out = U.simulate(U.BATCH1_CHARGE, spec, t_grid)
    except Exception:
        return np.full(NRES + (N1 - 2) + (N2 - 2), 10.0)
    r = []
    for sp, pts in ANCH.items():
        for tt, target in pts:
            j = int(np.argmin(np.abs(t_grid - tt)))
            v = out[sp][j]
            if not np.isfinite(v):
                return np.full(NRES + (N1 - 2) + (N2 - 2), 10.0)
            r.append((v - target)/max(abs(target), FLOOR))
    # curvature penalty: keep each feed ramp from kinking between nodes
    d1 = np.abs(z[:N1]); d2 = np.abs(z[N1:N1+N2])
    r.extend(SMOOTH*np.diff(d1, 2))
    r.extend(SMOOTH*np.diff(d2, 2))
    return np.asarray(r)


if __name__ == '__main__':
    # start from the Fig. 7(c) digitisation
    z0 = np.array([0.30, 0.25, 0.20, 0.16,                      # fv1 increments
                   0.45, 0.12, 0.10, 0.08, 0.05, 0.03])         # fv2 increments
    print(f'initial cost {0.5*np.sum(residuals(z0)**2):.4f}', flush=True)

    best, bc = None, np.inf
    rng = np.random.default_rng(0)
    for trial in range(10):
        z = z0 if trial == 0 else np.abs(z0*rng.uniform(0.3, 2.0, len(z0)))
        try:
            r = least_squares(residuals, z, bounds=(0.0, 1.0),
                              xtol=1e-12, ftol=1e-12, x_scale='jac', max_nfev=400)
        except Exception:
            continue
        print(f'  trial {trial}: cost {r.cost:.5f}', flush=True)
        if r.cost < bc:
            best, bc = r.x, r.cost

    spec = build(best)
    print(f'\nfitted feed programme (cost {bc:.5f})')
    print('  fv1_t =', [round(x, 1) for x in spec['fv1_t']])
    print('  fv1_v =', [round(float(x), 4) for x in spec['fv1_v']])
    print('  fv2_t =', [round(x, 1) for x in spec['fv2_t']])
    print('  fv2_v =', [round(float(x), 4) for x in spec['fv2_v']])

    out = U.simulate(U.BATCH1_CHARGE, spec, t_grid)
    print(f'\n{"sp":3s} {"t[h]":>5s} {"figure":>11s} {"fitted":>11s} {"rel err":>9s}')
    n = tot = 0
    for sp, pts in ANCH.items():
        for tt, target in pts:
            j = int(np.argmin(np.abs(t_grid - tt)))
            v = out[sp][j]
            rel = (v - target)/max(abs(target), FLOOR)
            tot += rel**2; n += 1
            print(f'{sp:3s} {tt:5.0f} {target:11.4g} {v:11.4g} {100*rel:8.1f}%')
    print(f'\nRMS relative error {100*np.sqrt(tot/n):.1f}%')

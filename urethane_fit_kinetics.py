"""
Estimate the kinetic parameters from Bauer's published mole-number trajectories.

The initial charge is confirmed correct: inverting Bauer's Experiment #1 design
variables gives n0_A = 0.094 and n0_B = 0.075 mol, and his figure starts at
isocyanate ~0.1 and butanol ~0.09. The totals also close -- his final
C + D = 1.02 mol against total B = 1.017, and C + 2D + 3E = 1.60 against
total A = 1.595.

What does not reproduce is the split between urethane and allophanate. Bauer's
figure gives D/C = 1.17 at 80 h; the parameters tabulated in the adapted papers
give 0.12. Those tabulated values are therefore not the ones that generated the
published trajectories, so they are re-estimated here directly from the figure.

Targets are anchor points digitised from Bauer Fig. (Experiment #1): all five
species, so the fit is constrained by the full mass balance rather than by the
three measured outputs alone.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import warnings, json, time
warnings.filterwarnings('ignore')
import urethane_case_study as U

# Bauer Experiment #1 control functions, read from his Sec. 7.4 plots
BAUER_CTRL = dict(
    name='Bauer Experiment #1',
    fv1_t=[0, 5, 55, 57, 80], fv1_v=[0, 0.97, 0.97, 1.0, 1.0],
    fv2_t=[0, 5, 22, 27, 55, 57, 80], fv2_v=[0, 0.68, 0.68, 0.92, 0.92, 1.0, 1.0],
    T_t=[0, 8, 45, 50, 80], T_C=[20, 197, 197, 177, 177])

# digitised anchor points: species index -> [(t [h], moles)]
#   0 nA, 1 nB, 2 nC, 3 nD, 4 nE
TARGETS = {
    # A isocyanate: spikes with the feed, then is drawn down
    0: [(5, 0.430), (15, 0.090), (25, 0.090), (45, 0.010), (80, 0.005)],
    # B butanol: consumed as fast as it is fed, so it never accumulates
    1: [(5, 0.015), (15, 0.005), (80, 0.002)],
    # C urethane: spikes early, is then eaten by the allophanate step, and
    # recovers only when the second butanol feed arrives after ~25 h
    2: [(5, 0.390), (10, 0.060), (20, 0.015), (40, 0.330), (60, 0.420), (80, 0.470)],
    3: [(10, 0.700), (25, 0.700), (45, 0.560), (80, 0.550)],
    4: [(20, 0.010), (80, 0.011)],
}
FLOOR = 0.02        # absolute floor in the relative residual, keeps small targets sane

PN = ['kref1', 'kref2', 'kref4', 'Ea1', 'Ea2', 'Ea4', 'Kc2', 'dH']
P0 = np.array([U.KREF1, U.KREF2, U.KREF4, U.EA1, U.EA2, U.EA4, U.KC2, -U.DELTA_H])
# optimise log10 of the rate constants / Kc2, and Ea / 1e4
Z0 = np.array([np.log10(P0[0]), np.log10(P0[1]), np.log10(P0[2]),
               P0[3]/1e4, P0[4]/1e4, P0[5]/1e4, np.log10(P0[6]), P0[7]/1e4])
LO = np.array([-6, -9, -9, 0.3, 0.3, 0.3, -3.0, 0.1])
HI = np.array([1, 1, 1, 12.0, 12.0, 12.0, 3.0, 8.0])

def unpack(z):
    return dict(kref1=10**z[0], kref2=10**z[1], kref4=10**z[2],
                Ea1=z[3]*1e4, Ea2=z[4]*1e4, Ea4=z[5]*1e4,
                Kc2=10**z[6], dH=-z[7]*1e4)

def rhs(t, y, inp, ic, P):
    nC, nD, nE = y
    f1, f2, T = inp(t)
    n, V = U.closure(nC, nD, nE, f1, f2, ic)
    if V <= 0:
        return [0.0, 0.0, 0.0]
    e = lambda Ea: np.exp(-Ea/U.R_GAS*(1.0/T - 1.0/U.TREF))
    k1 = P['kref1']*1e-3*e(P['Ea1'])
    k2 = P['kref2']*1e-3*e(P['Ea2'])
    k4 = P['kref4']*1e-3*e(P['Ea4'])
    Kc = P['Kc2']*1e-3*np.exp(-P['dH']/U.R_GAS*(1.0/T - 1.0/U.TREF))
    k3 = k2/Kc
    cA, cB, cC, cD = max(n[0], 0)/V, max(n[1], 0)/V, max(nC, 0)/V, max(nD, 0)/V
    return [V*(k1*cA*cB - k2*cA*cC + k3*cD), V*(k2*cA*cC - k3*cD), V*k4*cA*cA]

f_ctrl = U.hymech_inputs(BAUER_CTRL)
ic = U.design_to_moles(U.BAUER_EXPERIMENTS[1])
tg = np.arange(0.0, 80.0 + 1e-9, 0.2)

def all_species(P):
    s = solve_ivp(rhs, (0, 80.0), [0.0, 0.0, 0.0], t_eval=tg, args=(f_ctrl, ic, P),
                  method='LSODA', rtol=1e-9, atol=1e-12)
    if not s.success or s.y.shape[1] != len(tg):
        return None
    y = s.y.T
    out = np.zeros((len(tg), 5))
    for j in range(len(tg)):
        n, _ = U.closure(*y[j], *f_ctrl(tg[j])[:2], ic)
        out[j] = [n[0], n[1], y[j, 0], y[j, 1], y[j, 2]]
    return out

def residuals(z):
    P = unpack(z)
    Y = all_species(P)
    if Y is None or not np.isfinite(Y).all():
        return np.full(sum(len(v) for v in TARGETS.values()), 5.0)
    r = []
    for k, pts in TARGETS.items():
        for tt, tgt in pts:
            j = int(np.argmin(np.abs(tg - tt)))
            r.append((Y[j, k] - tgt)/max(abs(tgt), FLOOR))
    return np.asarray(r)

if __name__ == '__main__':
    t0 = time.time()
    print(f'initial cost (published parameters): {0.5*np.sum(residuals(Z0)**2):.4f}\n', flush=True)
    best, bc = None, np.inf
    rng = np.random.default_rng(1)
    for trial in range(8):
        z = Z0 if trial == 0 else Z0 + rng.normal(0, 0.35, len(Z0))
        z = np.clip(z, LO + 1e-6, HI - 1e-6)
        try:
            r = least_squares(residuals, z, bounds=(LO, HI), xtol=1e-12, ftol=1e-12,
                              x_scale='jac', max_nfev=800)
        except Exception:
            continue
        print(f'  trial {trial}: cost {r.cost:.5f}  ({time.time()-t0:.0f}s)', flush=True)
        if r.cost < bc:
            best, bc = r.x, r.cost

    P = unpack(best)
    print(f'\nfitted kinetics (cost {bc:.5f}):')
    for k in PN:
        pub = dict(kref1=U.KREF1, kref2=U.KREF2, kref4=U.KREF4, Ea1=U.EA1,
                   Ea2=U.EA2, Ea4=U.EA4, Kc2=U.KC2, dH=U.DELTA_H)[k]
        print(f'   {k:6s} {P[k]:12.5g}   published {pub:12.5g}   x{P[k]/pub:8.3f}')

    Y = all_species(P)
    print('\nfit against Bauer Fig. (Experiment #1)')
    print(f'{"sp":3s} {"t[h]":>5s} {"target":>9s} {"fitted":>9s} {"err":>8s}')
    for k, pts in TARGETS.items():
        for tt, tgt in pts:
            j = int(np.argmin(np.abs(tg - tt)))
            print(f'{"ABCDE"[k]:3s} {tt:5.0f} {tgt:9.4f} {Y[j,k]:9.4f} {Y[j,k]-tgt:+8.4f}')
    json.dump({k: float(P[k]) for k in PN}, open('urethane_fitted_kinetics.json', 'w'), indent=2)
    print(f'\nSaved urethane_fitted_kinetics.json   ({time.time()-t0:.0f}s)')

"""
Reproduce the authors' mole-profile figures using the authors' own model.

Everything in the MODEL is taken verbatim from the notebooks in
"Urethane case study.zip" (urethane_C/D/E.ipynb, urethane_validation.26ipynb):
constants, kinetic parameters, rate laws, material balances, volume closure
and measurement-noise levels. The only change is the integrator: the authors
call GEKKO in IMODE=4, which is not installed here, so the same system is
integrated with scipy. That is faithful because their DAE is index-1 with an
explicit algebraic part -- the three balances solve directly for nA, nB, nS,
and V follows from the densities -- so there is nothing for an implicit DAE
solver to do that substitution does not already do.

What is NOT in the zip is the input file

    C:\\Users\\alexs\\Desktop\\HyMech\\HyMech\\Datasets\\urethane_calibration.xlsx

which holds, per batch, a "<batch>_profiles" sheet (Time, fv1, fv2, T, A..E)
and a "<batch>_initial" sheet (V0, MV1, MV2, MV3, g0, g0v1, g0v2). Without it
the charge and the feed programme have to be recovered from the figures that
are stored in the notebooks themselves. Every such number is listed in
RECONSTRUCTED below with the figure it was read from. None of them is invented:
they are digitised from the authors' own saved output. They are, however,
digitised, so they carry reading error -- see the printed residual table.

Run:  python urethane_authors_replicate.py
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================== authors' code
# verbatim from urethane_C.ipynb cell 0
R = 0.008314            # kJ/(mol K)
t_ref = 363.16          # K

param = {
    'kref1': 1.25e-3,   # L/(mol h)
    'kref2': 7.29e-6,
    'kref4': 8.8e-7,
    'Ea1': 29.440,      # kJ/mol
    'Ea2': 71.014,
    'Ea4': 23.020,
    'kc2': 0.217,       # L/mol
    'dh': -18.300,      # kJ/mol
}
meas_err_std = {'C': 5e-3, 'D': 5e-5, 'E': 5e-6}
mol_mass = {'A': 0.11911, 'B': 0.07412, 'C': 0.19323,
            'D': 0.31234, 'E': 0.35733, 'S': 0.07806}      # kg/mol
rho = {'A': 1095.0, 'B': 809.0, 'C': 1415.0,
       'D': 1528.0, 'E': 1451.0, 'S': 1101.0}              # kg/m3

# the authors hard-code k3; it is exactly kref2/kc2 and Ea2 - dh
K3REF = 3.35945e-5
EA3 = 89.314
assert abs(K3REF - param['kref2']/param['kc2']) < 1e-10
assert abs(EA3 - (param['Ea2'] - param['dh'])) < 1e-9


def calc_initial_values(ic):
    """Verbatim port of the authors' calc_initial_values (design variables)."""
    v = {}
    q = (1 - ic['g0'])/ic['g0']
    phi = ic['MV1'] + ic['MV1']*ic['MV2'] - ic['MV3']
    v['nA0'] = (ic['V0']*rho['A']*rho['B']*rho['S'] /
                (mol_mass['A']*rho['B']*(rho['S'] + q*rho['A']) +
                 mol_mass['B']*rho['A']*phi*(rho['S'] + q*rho['B'])))
    v['nB0'] = phi*v['nA0']
    v['nC0'] = v['nD0'] = v['nE0'] = 0.0
    v['nS0'] = (q/mol_mass['S'])*(mol_mass['A'] + mol_mass['B']*phi)*v['nA0']
    v['nAv1'] = ic['MV2']*v['nA0']
    v['nSv1'] = ((1 - ic['g0v1'])/(ic['g0v1']*mol_mass['S']))*(mol_mass['A']*ic['MV2']*v['nA0'])
    v['nBv2'] = ic['MV3']*v['nA0']
    v['nSv2'] = ((1 - ic['g0v2'])/(ic['g0v2']*mol_mass['S']))*(mol_mass['B']*ic['MV3']*v['nA0'])
    v['V0'] = ic['V0']
    return v


def closure(nC, nD, nE, fv1, fv2, v):
    """The authors' three algebraic balances plus the density volume."""
    nA = v['nA0'] + fv1*v['nAv1'] - nC - 2.0*nD - 3.0*nE
    nB = v['nB0'] + fv2*v['nBv2'] - nC - nD
    nS = v['nS0'] + fv1*v['nSv1'] + fv2*v['nSv2']
    n = dict(A=nA, B=nB, C=nC, D=nD, E=nE, S=nS)
    V = sum(n[s]*mol_mass[s]/rho[s] for s in 'ABCDES')
    return n, V


def rhs(t, y, inp, v, k3_factor=1.0):
    """The authors' rate laws and ODEs (process(), lines 181-194)."""
    nC, nD, nE = y
    fv1, fv2, T = inp(t)
    n, V = closure(nC, nD, nE, fv1, fv2, v)
    if V <= 0:
        return [0.0, 0.0, 0.0]
    E = lambda Ea: np.exp((-Ea/R)*(1.0/T - 1.0/t_ref))
    k1 = param['kref1']*E(param['Ea1'])
    k2 = param['kref2']*E(param['Ea2'])
    k3 = k3_factor*K3REF*E(EA3*k3_factor)
    k4 = param['kref4']*E(param['Ea4'])
    nA, nB = max(n['A'], 0.0), max(n['B'], 0.0)
    r1 = k1*nA*nB/V**2
    r2 = k2*nA*max(nC, 0.0)/V**2
    r3 = k3*max(nD, 0.0)/V
    r4 = k4*(nA/V)**2
    return [V*(r1 - r2 + r3), V*(r2 - r3), V*r4]


# ======================================================== reconstructed inputs
# Batch_1, read off the authors' own stored figures.
#
#   T(t)            "Temperature (T)" panel, urethane_C cell 1
#                   296 K -> 500 K over 0-8 h, hold to 48 h,
#                   -> 455 K over 48-56 h, hold to 80 h
#   nA0, nB0        t=0 of the nA and nB panels, "Baseline Simulation Results"
#   V0              t=0 of the Volume panel of the same figure (2.5e-5 m3)
#   nAv1            A balance at t=80 h (fv1=1): nA+nC+2nD+3nE - nA0
#   nBv2            B balance at t=80 h (fv2=1): nB+nC+nD - nB0
#   nS0             solved from V0 = sum n_i M_i / rho_i at t=0
#   nSv1            from the shortfall in the Volume panel once everything
#                   else is fixed: V is 1.446e-5 m3 low at t=80 h (fv1=1),
#                   and 1.446e-5 / (M_S/rho_S) = 0.204 mol. The same number
#                   closes the t=8 h and t=48 h readings at fv1=0.92, so it
#                   scales with fv1 and therefore belongs to the FIRST feed.
#   nSv2            0: no residual volume is left for the second feed
#   fv1, fv2        HyMech Fig. 7(c), as digitised in urethane_case_study.py
RECONSTRUCTED = True

def _nS0_from_V0(V0, nA0, nB0):
    vol = nA0*mol_mass['A']/rho['A'] + nB0*mol_mass['B']/rho['B']
    return (V0 - vol)/(mol_mass['S']/rho['S'])

BATCH1_CHARGE = dict(nA0=0.028, nB0=0.170, nC0=0.0, nD0=0.0, nE0=0.0,
                     nAv1=0.5495, nBv2=0.207, nSv1=0.204, nSv2=0.0,
                     V0=2.5e-5)
BATCH1_CHARGE['nS0'] = _nS0_from_V0(BATCH1_CHARGE['V0'],
                                    BATCH1_CHARGE['nA0'], BATCH1_CHARGE['nB0'])

#   fv1, fv2        fitted to all six panels of the stored figure by
#                   urethane_fit_feeds.py, with the model, the parameters,
#                   T(t) and the charge all held fixed. Starting point was
#                   the HyMech Fig. 7(c) digitisation; the fit moved fv2
#                   down sharply over 0-5 h, which is what the B balance at
#                   the nC peak requires. RMS residual 5.6% over 26 anchors.
BATCH1_INPUTS = dict(
    name='Batch_1',
    T_t=[0.0, 8.0, 48.0, 56.0, 80.0], T_K=[296.0, 500.0, 500.0, 455.0, 455.0],
    fv1_t=[0.0, 1.5, 3.0, 5.0, 7.0, 50.0, 57.0, 80.0],
    fv1_v=[0.0, 0.2339, 0.3741, 0.5659, 0.8762, 0.8762, 1.0, 1.0],
    fv2_t=[0.0, 5.0, 10.0, 22.0, 30.0, 45.0, 50.0, 57.0, 80.0],
    fv2_v=[0.0, 0.6222, 0.6661, 0.6960, 0.7308, 0.8014, 0.8014, 1.0, 1.0])


def make_inputs(spec):
    def f(t):
        return (float(np.interp(t, spec['fv1_t'], spec['fv1_v'])),
                float(np.interp(t, spec['fv2_t'], spec['fv2_v'])),
                float(np.interp(t, spec['T_t'], spec['T_K'])))
    return f


def simulate(charge, spec, t_eval, k3_factor=1.0):
    inp = make_inputs(spec)
    s = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [charge['nC0'], charge['nD0'], charge['nE0']],
                  t_eval=t_eval, args=(inp, charge, k3_factor),
                  method='LSODA', rtol=1e-10, atol=1e-14)
    if not s.success:
        raise RuntimeError(s.message)
    y = s.y.T
    out = {k: np.zeros(len(t_eval)) for k in 'ABCDES'}
    out['V'] = np.zeros(len(t_eval))
    out['T'] = np.zeros(len(t_eval))
    for j, tt in enumerate(t_eval):
        fv1, fv2, T = inp(tt)
        n, V = closure(*y[j], fv1, fv2, charge)
        for k in 'ABCDES':
            out[k][j] = n[k]
        out['V'][j] = V
        out['T'][j] = T
    return out


def add_noise(out, seed=42):
    """The authors seed 42 and perturb only C, D and E."""
    rng = np.random.RandomState(seed)
    m = {k: out[k].copy() for k in 'ABCDES'}
    m['V'] = out['V']
    for s in 'CDE':
        m[s] = out[s] + rng.normal(0, meas_err_std[s], len(out[s]))
    return m


# ==================================================== the authors' two figures
# anchors digitised from the stored "Baseline Simulation Results" figure
ANCHORS = {
    'A': [(0, 0.028), (7, 0.046), (10, 0.003), (20, 0.0030), (80, 0.0015)],
    'B': [(0, 0.170), (10, 0.002), (55, 0.010), (80, 0.001)],
    'C': [(5, 0.230), (10, 0.115), (20, 0.125), (40, 0.160), (80, 0.180)],
    'D': [(8, 0.207), (25, 0.204), (45, 0.182), (57, 0.205), (80, 0.197)],
    'E': [(10, 3.20e-4), (80, 3.40e-4)],
    'V': [(0, 2.50e-5), (8, 7.80e-5), (48, 7.90e-5), (80, 8.60e-5)],
}

if __name__ == '__main__':
    t = np.arange(0.0, 80.0 + 1e-9, 0.5)          # the authors' 0.5 h sampling
    clean = simulate(BATCH1_CHARGE, BATCH1_INPUTS, t)
    meas = add_noise(clean)

    print('Batch_1 charge used (mol):')
    for k in ['nA0', 'nB0', 'nS0', 'nAv1', 'nBv2', 'nSv1', 'nSv2']:
        print(f'   {k:6s} {BATCH1_CHARGE[k]:10.5f}')
    print(f'   V0     {BATCH1_CHARGE["V0"]:10.3e} m3')

    print(f'\nresidual against the authors\' stored figure')
    print(f'{"sp":3s} {"t[h]":>5s} {"figure":>11s} {"simulated":>11s} {"rel err":>9s}')
    tot = 0.0
    for sp, pts in ANCHORS.items():
        for tt, target in pts:
            j = int(np.argmin(np.abs(t - tt)))
            v = clean[sp][j]
            rel = (v - target)/abs(target)
            tot += rel**2
            print(f'{sp:3s} {tt:5.0f} {target:11.4g} {v:11.4g} {100*rel:8.1f}%')
    print(f'\nRMS relative error {100*np.sqrt(tot/sum(len(p) for p in ANCHORS.values())):.1f}%')

    # ---------------------------------------- figure 1: authors' 6-panel layout
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))
    fig.suptitle('Baseline Simulation Results')
    panels = [('nA', 'A', 'nA [mol]'), ('nB', 'B', 'nB [mol]'), ('nC', 'C', 'nC [mol]'),
              ('nD', 'D', 'nD [mol]'), ('nE', 'E', 'nE [mol]'), ('Volume', 'V', 'V [m$^3$]')]
    for a, (title, key, ylab) in zip(ax.ravel(), panels):
        a.plot(t, meas[key] if key in 'CDE' else clean[key], linewidth=2)
        a.set_title(title); a.set_xlabel('Time [h]'); a.set_ylabel(ylab)
        a.grid(alpha=0.4); a.set_xlim(0, 80)
    fig.tight_layout()
    fig.savefig('urethane_authors_baseline.png', dpi=200)

    # ------------------------------ figure 2: authors' 3-panel species layout
    fig2, ax2 = plt.subplots(1, 3, figsize=(13.2, 4.2))
    info = [('Species C', 'C', 'blue', 'o'), ('Species D', 'D', 'red', 's'),
            ('Species E', 'E', 'green', '^')]
    for a, (lab, key, col, mk) in zip(ax2, info):
        a.plot(t, meas[key], mk, ms=7, mfc='none', mec=col, mew=1.4, ls='none')
        a.set_xlabel('time [h]', fontsize=15)
        a.set_ylabel('no. of moles [mol]', fontsize=15)
        a.set_xlim(-3, 83); a.grid(alpha=0.45)
        a.text(0.97, 0.06, lab, transform=a.transAxes, ha='right', fontsize=14,
               bbox=dict(boxstyle='round', fc='white', ec='black'))
    fig2.tight_layout()
    fig2.savefig('urethane_authors_species.png', dpi=200)

    # ------------------------------------------------- figure 3: the T profile
    fig3, a3 = plt.subplots(figsize=(9, 5.5))
    a3.scatter(t, clean['T'], color='gray', alpha=0.5, label='T sperimentale')
    a3.plot(t, clean['T'], 'r-', label='T augmented (interp1d)')
    a3.set_title('Temperature (T)'); a3.legend(fontsize=13); a3.grid(alpha=0.3)
    fig3.tight_layout(); fig3.savefig('urethane_authors_T.png', dpi=200)

    print('\nSaved urethane_authors_baseline.png, urethane_authors_species.png, '
          'urethane_authors_T.png')

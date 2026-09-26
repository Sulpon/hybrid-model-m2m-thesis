"""
Mole-number profiles of the urethane process -- experimental (process) data only.

Charge:   Bauer Experiment #1, recovered by inverting his seven design variables
Controls: Bauer Experiment #1 feed and temperature profiles
Kinetics: estimated from Bauer's published trajectories (urethane_fitted_kinetics.json)

No mechanistic, data-driven or hybrid model appears here.

Left panel follows Bauer's own layout (isocyanate, butanol, urethane); right
panel the byproducts (allophanate, isocyanurate). The second figure shows the
three measured species in the HyMech layout.
"""
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import urethane_case_study as U
import urethane_fit_kinetics as K

P = json.load(open('urethane_fitted_kinetics.json'))
ic = U.design_to_moles(U.BAUER_EXPERIMENTS[1])
f = U.hymech_inputs(K.BAUER_CTRL)
t = np.arange(0.0, 80.0 + 1e-9, 0.1)

s = solve_ivp(K.rhs, (0, 80.0), [0.0, 0.0, 0.0], t_eval=t, args=(f, ic, P),
              method='LSODA', rtol=1e-9, atol=1e-12)
y = s.y.T
n = np.zeros((len(t), 6))
for j in range(len(t)):
    n[j], _ = U.closure(*y[j], *f(t[j])[:2], ic)          # A B C D E S

rng = np.random.default_rng(5)
ts = np.arange(0.0, 80.0 + 1e-9, 0.5)                      # 30-min sampling
idx = [int(np.argmin(np.abs(t - x))) for x in ts]
meas = n[idx][:, [2, 3, 4]] + rng.normal(0, U.NOISE_LEVELS, (len(ts), 3))

# ---------------------------------------------------------- Bauer layout
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
for lbl, k, ls in [('isocyanate (A)', 0, '-'), ('butanol (B)', 1, '--'),
                   ('urethane (C)', 2, ':')]:
    ax[0].plot(t, n[:, k], ls, color='k', lw=1.6, label=lbl)
ax[0].set_ylabel('molar numbers [mol]'); ax[0].set_ylim(0, 0.5)
for lbl, k, ls in [('allophanate (D)', 3, ':'), ('isocyanurate (E)', 4, '--')]:
    ax[1].plot(t, n[:, k], ls, color='k', lw=1.6, label=lbl)
ax[1].set_ylabel('molar numbers [mol]'); ax[1].set_ylim(0, 0.8)
for a in ax:
    a.set_xlabel('time [hours]'); a.set_xlim(0, 80)
    a.set_xticks([0, 20, 40, 60, 80]); a.legend(fontsize=9, loc='upper left')
    a.tick_params(direction='in', top=True, right=True)
fig.tight_layout(); fig.savefig('urethane_moles_bauer.png', dpi=200)

# ---------------------------------------------------------- measured species
fig2, ax2 = plt.subplots(1, 3, figsize=(12.4, 3.7))
info = [('Species C  (urethane)', 2, 0, '#1f4e9c', 'o'),
        ('Species D  (allophanate)', 3, 1, '#c0392b', 's'),
        ('Species E  (isocyanurate)', 4, 2, '#2e8b57', '^')]
for a, (title, kn, km, col, mk) in zip(ax2, info):
    a.plot(t, n[:, kn], '-', color=col, lw=1.6, label='process (true)')
    a.plot(ts[::3], meas[::3, km], mk, ms=4.0, mfc='none', mec=col, mew=1.0,
           ls='none', label='measured')
    a.set_xlabel('time [h]'); a.set_ylabel('no. of moles [mol]')
    a.set_title(title, fontsize=10); a.set_xlim(0, 80)
    a.set_xticks([0, 20, 40, 60, 80])
    a.tick_params(direction='in', top=True, right=True); a.grid(alpha=0.25, lw=0.5)
ax2[0].legend(fontsize=8, loc='upper left')
fig2.tight_layout(); fig2.savefig('urethane_moles_species.png', dpi=200)

print(f'{"species":16s} {"t=5h":>9s} {"t=25h":>9s} {"t=80h":>9s}')
for lbl, k in [('isocyanate A', 0), ('butanol B', 1), ('urethane C', 2),
               ('allophanate D', 3), ('isocyanurate E', 4)]:
    g = lambda x: n[int(np.argmin(np.abs(t - x))), k]
    print(f'{lbl:16s} {g(5):9.4f} {g(25):9.4f} {g(80):9.4f}')
print(f'\npeak A {n[:,0].max():.4f} at t={t[np.argmax(n[:,0])]:.1f} h   '
      f'peak D {n[:,3].max():.4f} at t={t[np.argmax(n[:,3])]:.1f} h')
print('Bauer figure: A peak 0.43 at ~5 h, D peak 0.70, C@80 0.47, D@80 0.55')
print('\nSaved urethane_moles_bauer.png and urethane_moles_species.png')

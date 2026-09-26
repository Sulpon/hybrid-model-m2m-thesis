"""
Process data only: mole profiles of nC, nD and nE over time.

The two batches are the explicit HyMech Fig. 7(c) input patterns, run through
the GROUND-TRUTH model (the reversible A + C <-> D network). Lines are the
noise-free trajectories; symbols are the simulated measurements, sampled every
30 min with the published measurement noise.

No mechanistic, data-driven or hybrid model appears here -- this is the process
alone. The initial charge is Bauer Experiment #1, which is the only published
source for it; nothing else about the run follows Bauer.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import urethane_case_study as U

DURATION = U.BAUER_BATCH_H            # 80 h, as in HyMech
DT = 0.5                              # sampling interval [h]
SEED = 3

t = np.arange(0.0, DURATION + 1e-9, DT)
ic = U.design_to_moles(U.BAUER_EXPERIMENTS[1])
rng = np.random.default_rng(SEED)

runs = []
for spec in U.HYMECH_BATCHES:
    f = U.hymech_inputs(spec)
    clean = U.simulate(U.ground_truth_rhs, f, ic, t)
    runs.append(dict(name=spec['name'], clean=clean, meas=U.add_noise(clean, rng)))

every = 6                             # show a symbol every 3 h
species = [('$n_C$  urethane', 0, '#1f4e9c'),
           ('$n_D$  allophanate', 1, '#c0392b'),
           ('$n_E$  isocyanurate', 2, '#2e8b57')]
styles = ['-', '--']
marks = ['o', 's']

fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.7))
for a, (title, k, colour) in zip(ax, species):
    for r, ls, mk in zip(runs, styles, marks):
        a.plot(t, r['clean'][:, k], ls, color=colour, lw=1.7,
               label=f"{r['name']} (true)")
        a.plot(t[::every], r['meas'][::every, k], mk, ms=4.0, mfc='none',
               mec=colour, mew=1.0, ls='none', label=f"{r['name']} (measured)")
    a.set_xlabel('time [h]')
    a.set_ylabel('moles [mol]')
    a.set_title(title, fontsize=10)
    a.set_xlim(0, DURATION)
    a.set_xticks([0, 20, 40, 60, 80])
    a.tick_params(direction='in', top=True, right=True)
    a.grid(alpha=0.25, lw=0.5)
ax[0].legend(fontsize=7.5, loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig('urethane_process_moles.png', dpi=200)

print(f'{"batch":9s} {"species":9s} {"final [mol]":>13s} {"noise sd":>10s} {"noise/final":>12s}')
for r in runs:
    for nm, k in [('nC', 0), ('nD', 1), ('nE', 2)]:
        v = r['clean'][-1, k]
        print(f'{r["name"]:9s} {nm:9s} {v:13.5f} {U.NOISE_LEVELS[k]:10.1e} '
              f'{U.NOISE_LEVELS[k]/v*100:11.2f}%')
print('\nSaved urethane_process_moles.png')

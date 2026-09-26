"""
Reproduction of Rossi et al. (HyMech) Fig. 7(c): the input excitation patterns
of the two batches used in that study -- accumulated feed ratios from the two
vessels and the reactor temperature.

The profiles are the explicit ones defined in urethane_case_study.HYMECH_BATCHES,
read off the published figure, not random draws from the Geremia band.
Batch length is 80 h and temperature is plotted in degrees Celsius, both as in
the source figure.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import urethane_case_study as U

DURATION = U.BAUER_BATCH_H          # 80 h, as in HyMech
tg = np.arange(0.0, DURATION + 1e-9, 0.05)
fns = [U.hymech_inputs(s) for s in U.HYMECH_BATCHES]
prof = [np.array([f(x) for x in tg]) for f in fns]

fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
panels = [(0, 'feed ratio from V1 [-]', '#9a9a2a', (0.0, 1.0)),
          (1, 'feed ratio from V2 [-]', '#8e3a8e', (0.0, 1.0)),
          (2, 'temperature [°C]',  '#c0392b', (0.0, 250.0))]
styles = ['-', '--']
for a, (col, ylab, colour, ylim) in zip(ax, panels):
    for p, ls, spec in zip(prof, styles, U.HYMECH_BATCHES):
        y = p[:, col] - 273.15 if col == 2 else p[:, col]
        a.plot(tg, y, ls, color=colour, lw=1.7, label=spec['name'])
    a.set_xlabel('time [h]')
    a.set_ylabel(ylab)
    a.set_xlim(0, DURATION)
    a.set_ylim(*ylim)
    a.set_xticks([0, 20, 40, 60, 80])
    if col < 2:
        a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        a.set_yticks([0, 50, 100, 150, 200, 250])
    a.legend(loc='lower right', fontsize=8.5, frameon=True)
    a.tick_params(direction='in', top=True, right=True)
    a.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig('urethane_inputs.png', dpi=200)

print(f'{"batch":9s} {"fv1 plateau":>12s} {"fv2 plateaus":>16s} {"T plateaus [C]":>17s}')
for p, spec in zip(prof, U.HYMECH_BATCHES):
    tC = p[:, 2] - 273.15
    print(f'{spec["name"]:9s} {spec["fv1_v"][1]:12.2f} '
          f'{spec["fv2_v"][1]:7.2f} /{spec["fv2_v"][3]:7.2f} '
          f'{spec["T_C"][1]:9.0f} /{spec["T_C"][3]:7.0f}')
print('\nSaved urethane_inputs.png')

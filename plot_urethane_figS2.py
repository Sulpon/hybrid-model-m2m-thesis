"""
Regeneration of Geremia et al. Figure S.2: validation results for one batch.

Three panels (nC, nD, nE) showing the in-silico process data against the three
models compared in the source study:
    - purely mechanistic  (FP model, parameters fitted on calibration)
    - purely data-driven  (random forest on the process inputs)
    - hybrid              (random forest on inputs + mechanistic outputs)

The kinetic parameters are taken from urethane_fitted_params.json, written by
urethane_replicate.py, so this script does not refit them.
"""
import numpy as np
import json, warnings, time
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
import urethane_case_study as U
import urethane_replicate as RP

BATCH = 0            # validation batch #1
SEED = 7

t0 = time.time()
th = np.array([json.load(open('urethane_fitted_params.json'))[k]
               for k in ['kref1', 'kref2', 'kref4', 'Ea1', 'Ea2', 'Ea4']])
print('fitted mechanistic parameters:', np.round(th, 6))

t, inp_c, meas_c, clean_c, _, ic = U.generate(U.N_CALIBRATION, 101)
_, inp_v, meas_v, clean_v, _, _ = U.generate(U.N_VALIDATION, 202)

mech_c = np.array([RP.sim_fp(th, f, ic, t) for f in inp_c])
mech_v = np.array([RP.sim_fp(th, f, ic, t) for f in inp_v])

grid = {'max_depth': [None, 12, 20], 'min_samples_leaf': [1, 2, 5]}
cv = KFold(5, shuffle=True, random_state=SEED)
preds = {}
for tag, mc, mv in [('data-driven', None, None), ('hybrid', mech_c, mech_v)]:
    Xtr = RP.features(inp_c, t, mc)
    Xte = RP.features(inp_v, t, mv)
    gs = GridSearchCV(RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1),
                      grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
    gs.fit(Xtr, RP.flatten_targets(meas_c))
    preds[tag] = gs.best_estimator_.predict(Xte).reshape(meas_v.shape)
    print(f'  {tag:12s} RF {gs.best_params_}   ({time.time()-t0:.0f}s)', flush=True)

every = 6                                      # show a measurement every 3 h
lab = [r'$n_C$ [mol]', r'$n_D$ [mol]', r'$n_E$ [mol]']
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
for k, a in enumerate(ax):
    a.plot(t[::every], meas_v[BATCH, ::every, k], 'o', ms=4.2, mfc='none',
           mec='0.25', mew=1.0, label='process data', zorder=5)
    a.plot(t, mech_v[BATCH, :, k], '-', color='#C44E52', lw=1.7, label='mechanistic')
    a.plot(t, preds['data-driven'][BATCH, :, k], '--', color='#55A868', lw=1.7,
           label='data-driven')
    a.plot(t, preds['hybrid'][BATCH, :, k], ':', color='#1f4e9c', lw=2.4, label='hybrid')
    a.set_xlabel('time [h]')
    a.set_ylabel(lab[k])
    a.set_xlim(0, U.BATCH_DURATION_H)
    a.tick_params(direction='in', top=True, right=True)
    a.set_title(f'({chr(97+k)})', loc='left', fontsize=10)
ax[0].legend(loc='upper left', fontsize=8, frameon=True)
fig.tight_layout()
fig.savefig('urethane_figS2.png', dpi=200)

print('\nMAE on this batch [mol]')
print(f'{"model":14s} {"nC":>11s} {"nD":>11s} {"nE":>11s}')
for tag, p in [('mechanistic', mech_v), ('data-driven', preds['data-driven']),
               ('hybrid', preds['hybrid'])]:
    e = np.abs(p[BATCH] - meas_v[BATCH]).mean(0)
    print(f'{tag:14s} {e[0]:11.3e} {e[1]:11.3e} {e[2]:11.3e}')
print('\nSaved urethane_figS2.png')

"""
Test: X1.xlsx/M2.xlsx and X1_mis.xlsx/M2_mis.xlsx share IDENTICAL operating
conditions (CA0,T1,t1,T2,D0,t2 all match exactly) but DIFFERENT X2/Y --
confirmed by direct comparison. So the author drew conditions once and reused
them, but ran the SDE simulation separately per case. This feeds the REAL
case-(ii) conditions into a fresh SDE run (corrected sigma1/sigma2) and
compares to X2_mis.xlsx/Y_mis.xlsx, to see how much closer this gets versus
drawing fresh random conditions (which is what every previous attempt did).
"""
import numpy as np
import pandas as pd
import sdeint
import warnings
warnings.filterwarnings('ignore')

R = 8.314
V1, V2 = 0.1, 0.2
A1, A2, A3, A4 = 28, 40, 10, 20
Ea1_m, Ea2_m, Ea3_m, Ea4_m = 20000, 30000, 50000, 55000
CV = 0.01
sigma1 = [0.75, 0.75, 0.25]
sigma2 = [0.25, 0.75, 0.25, 0.75, 0.75, 0.25]

def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    return np.array([-k1*CA, k1*CA - k2*CB, k2*CB])
def stage1_noise(y, t): return np.diag(sigma1)
def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    return np.array([0, -k3*CB*CD**2, -k4*CC*CD**2, -2*k3*CB*CD**2-2*k4*CC*CD**2, k3*CB*CD**2, k4*CC*CD**2])
def stage2_noise(y, t): return np.diag(sigma2)

X1m = pd.read_excel('X1_mis.xlsx'); M2m = pd.read_excel('M2_mis.xlsx')
X2m = pd.read_excel('X2_mis.xlsx'); Ym = pd.read_excel('Y_mis.xlsx')
N = len(X1m)
species = ['A','B','C','D','E','F']

np.random.seed(10)   # try the same seed=10 as case (i), just applied to this separate pass
disc2_rows = []
for i in range(N):
    CA0s, T1, t1min = X1m['CA0'].iloc[i], X1m['T1'].iloc[i], X1m['t1'].iloc[i]
    T2, CD0s, t2min = M2m['T2'].iloc[i], M2m['D0'].iloc[i], M2m['t2'].iloc[i]
    t1s, t2s = t1min*60, t2min*60

    Ea1 = np.random.normal(Ea1_m, CV*Ea1_m); Ea2 = np.random.normal(Ea2_m, CV*Ea2_m)
    Ea3 = np.random.normal(Ea3_m, CV*Ea3_m); Ea4 = np.random.normal(Ea4_m, CV*Ea4_m)
    k1 = A1*np.exp(-Ea1/(R*T1)); k2 = A2*np.exp(-Ea2/(R*T1))
    k3 = A3*np.exp(-Ea3/(R*T2)); k4 = A4*np.exp(-Ea4/(R*T2))

    t1v = np.linspace(0, t1s, 1000)
    sol1 = sdeint.itoint(lambda y,t: stage1_drift(y,t,k1,k2), stage1_noise, np.array([CA0s,0.,0.]), t1v)
    CA1f, CB1f, CC1f = sol1[-1]
    d = V1/V2
    y0 = np.array([CA1f*d, CB1f*d, CC1f*d, CD0s, 0., 0.])
    t2v = np.linspace(0, t2s, 1000)
    sol2 = sdeint.itoint(lambda y,t: stage2_drift(y,t,k3,k4), stage2_noise, y0, t2v)
    disc2_rows.append(sol2[-1])   # true final point (own t2s, matches Y convention)

final = np.array(disc2_rows)
Yg = final[:,4]/final.sum(axis=1)*100

print("===== fixed-conditions regen (case ii) vs X2_mis.xlsx/Y_mis.xlsx, final point =====")
for j, sp in enumerate(species):
    a = X2m[f'{sp}_{X2m.shape[1]//6}'].values
    print(f"{sp}: real {a.mean():9.2f}+/-{a.std(ddof=1):6.2f}   fixed-cond gen {final[:,j].mean():9.2f}+/-{final[:,j].std(ddof=1):6.2f}   "
          f"corr(real,gen)={np.corrcoef(a, final[:,j])[0,1]:.3f}")
print(f"\nY: real {Ym['E_pur'].values.mean():.2f}+/-{Ym['E_pur'].values.std(ddof=1):.2f}   "
      f"gen {Yg.mean():.2f}+/-{Yg.std(ddof=1):.2f}   corr={np.corrcoef(Ym['E_pur'].values, Yg)[0,1]:.3f}")

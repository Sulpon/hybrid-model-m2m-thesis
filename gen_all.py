import pandas as pd, numpy as np
from scipy.integrate import odeint
import warnings; warnings.filterwarnings('ignore')
 
R=8.314
# true parameters from Data_Generation.py
A3,A4=10.0,20.0; Ea3,Ea4=50000.0,55000.0
 
X2=pd.read_excel('X2.xlsx'); M2ref=pd.read_excel('M2.xlsx'); Y=pd.read_excel('Y.xlsx')
X1=pd.read_excel('X1.xlsx')
sp=['A','B','C','D','E','F']; n=len(X2)
 
IC=np.array([[X2[f'{s}_1'].iloc[i] for s in sp] for i in range(n)])
T2=M2ref['T2'].values; t2=M2ref['t2'].values; D0=M2ref['D0'].values
 
def build(rhs, extra_cols=True, label=''):
    """integrate rhs for each batch, return M2-style DataFrame."""
    out=np.zeros((n,6,7))
    for i in range(n):
        sol=odeint(rhs, IC[i], np.linspace(0,t2[i]*60,7), args=(T2[i],))
        out[i]=sol.T
    cols={}
    for j,s in enumerate(sp):
        for k in range(7):
            cols[f'{s}_fit_{k+1}']=out[:,j,k]
    df=pd.DataFrame(cols)
    df['t2']=t2; df['T2']=T2; df['D0']=D0
    return df
 
# ===== MODEL DEFINITIONS =====
# M0: CORRECT - full mechanism
def m_correct(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea4/(R*T2v))
    return [0,-k3*CB*CD**2,-k4*CC*CD**2,-2*k3*CB*CD**2-2*k4*CC*CD**2,k3*CB*CD**2,k4*CC*CD**2]
 
# M1: DELETED SIDE REACTION (the original 'mis') - engineer missed C+2D->F entirely
def m_no_side(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v))
    return [0,-k3*CB*CD**2,0,-2*k3*CB*CD**2,k3*CB*CD**2,0]
 
# M2: WRONG ORDER IN D - engineer assumed 1st order in D (couldn't resolve the D^2 dependence)
def m_order1_D(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea4/(R*T2v))
    return [0,-k3*CB*CD,-k4*CC*CD,-2*k3*CB*CD-2*k4*CC*CD,k3*CB*CD,k4*CC*CD]
 
# M3: WRONG ACTIVATION ENERGY on k4 - couldn't measure Ea4, guessed it equal to Ea3
def m_wrong_Ea4(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v)); k4=A4*np.exp(-Ea3/(R*T2v))  # Ea3 used for k4
    return [0,-k3*CB*CD**2,-k4*CC*CD**2,-2*k3*CB*CD**2-2*k4*CC*CD**2,k3*CB*CD**2,k4*CC*CD**2]
 
# M4: LUMPED - engineer couldn't separate E and F, modelled one combined product from B+C
def m_lumped(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v))
    rate=k3*(CB+CC)*CD**2
    return [0,-k3*CB*CD**2,-k3*CC*CD**2,-2*rate,k3*CB*CD**2,k3*CC*CD**2]
 
# M5: NO D DEPENDENCE - engineer treated D as excess/constant (pseudo-first-order in B,C only)
def m_no_D(y,t,T2v):
    CA,CB,CC,CD,CE,CF=y
    k3=A3*np.exp(-Ea3/(R*T2v))*1e6; k4=A4*np.exp(-Ea4/(R*T2v))*1e6  # rescale since dropping D^2
    return [0,-k3*CB,-k4*CC,-2*k3*CB-2*k4*CC,k3*CB,k4*CC]
 
models={
 'M0_correct':      (m_correct,  'Full mechanism (correct)'),
 'M1_no_side':      (m_no_side,  'Deleted side reaction C+2D->F'),
 'M2_order1_D':     (m_order1_D, 'Wrong order in D (1st not 2nd)'),
 'M3_wrong_Ea4':    (m_wrong_Ea4,'Wrong activation energy Ea4=Ea3'),
 'M4_lumped_EF':    (m_lumped,   'Lumped E+F into one product'),
 'M5_no_D':         (m_no_D,     'Ignored D dependence (D in excess)'),
}
 
import pickle
blocks={}
for name,(rhs,desc) in models.items():
    df=build(rhs,label=name)
    blocks[name]=df
    # quick sanity: variance of E trajectory
    Evar=df[[f'E_fit_{k}' for k in range(1,8)]].values.std()
    print(f'{name:16s} {desc:38s}  E-traj sd={Evar:8.2f}')
pickle.dump({'blocks':blocks,'X1':X1,'X2':X2,'Y':Y}, open('all_models.pkl','wb'))
print('saved all_models.pkl')
 
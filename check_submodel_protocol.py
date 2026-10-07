"""
Which sub-model allocation protocol is safe?

Re-optimising each reduced model on its own grid is the fairer comparison, but
it breaks the nesting that guarantees non-negative uniques: the full model is
charged a parsimony penalty at its own allocation while the reduced models are
charged theirs, and the penalties are not comparable across models with
different numbers of blocks. When the reduced model's pick lands better, the
unique contribution goes negative and the ratio becomes meaningless.

Three protocols are compared on both case studies:

  inherited    reduced models evaluated at the full model's per-block counts
  1-SE         every model at the parsimonious member of its own 1-SE region
  argmin       every model at its own minimum-RMSECV allocation

The last puts all three models at their best attainable cross-validated
performance, so the comparison is like for like and nesting is far more likely
to hold.
"""
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import case_study_1 as A
import case_study_2 as B

BAD1 = {'M1_no_side', 'M5_no_D', 'M6_author_mis'}


def amin(rmse):
    return np.unravel_index(rmse.argmin(), rmse.shape)


def cs1():
    d = A.load_C('RG_calibration.xlsx')
    cache = json.load(open('regen_fitted_params_paperIC.json'))
    Y = d['Y']
    qX1, rfX1 = A.cv_1(d['X1'], Y, A.MAX_X1)
    rows = []
    for name, rhs in A.MODELS.items():
        M2, _ = A.build_M2_C(rhs, d, cache[name])
        qF, rf = A.cv_3(d['X1'], M2, d['X2'], Y, A.MAX_X1, A.MAX_M2, A.MAX_X2)
        qKD, rfKD = A.cv_2(d['X1'], M2, Y, A.MAX_X1, A.MAX_M2)
        qDD, rfDD = A.cv_2(d['X1'], d['X2'], Y, A.MAX_X1, A.MAX_X2)
        p1se, *_ = A.pick_1se(rf.mean(0), rf)
        pkd1, *_ = A.pick_1se(rfKD.mean(0), rfKD)
        pdd1, *_ = A.pick_1se(rfDD.mean(0), rfDD)
        pf, pkdA, pddA = amin(rf.mean(0)), amin(rfKD.mean(0)), amin(rfDD.mean(0))
        a, b, c = p1se
        for tag, qf, kd, dd in [
                ('inherited', qF[p1se], qKD[a, b], qDD[a, c]),
                ('1-SE',      qF[p1se], qKD[pkd1], qDD[pdd1]),
                ('argmin',    qF[pf],   qKD[pkdA], qDD[pddA])]:
            uM, uX = float(qf - dd), float(qf - kd)
            rows.append(dict(case='CS1', model=name, protocol=tag, uM=uM, uX=uX,
                             ratio=uM/uX if uX > 1e-9 else np.nan,
                             sound='sound' if name not in BAD1 else 'BROKEN'))
    return rows


def cs2():
    charges, inps, clean, meas = B.generate()
    nb = len(inps)
    keep = B.TGRID <= 70.0
    inp_full = np.array([[f(tt) for tt in B.TGRID] for f in inps]
                        ).transpose(0, 2, 1).reshape(nb, -1)
    cache = json.load(open(B.CACHE))
    rows = []
    for rname, ri in [('nC', 0), ('nD', 1)]:
        Y = meas[:, -1, ri].reshape(-1, 1)
        for name, (rev, npar) in B.CANDIDATES.items():
            th = np.array(cache[name]['th'])
            P = np.array([B.simulate(B.make_rhs(rev, th), f, c)
                          for f, c in zip(inps, charges)])
            Mb = np.hstack([P[:, keep, :].transpose(0, 2, 1).reshape(nb, -1), inp_full])
            Xb = meas[:, keep, :].transpose(0, 2, 1).reshape(nb, -1)
            qF, rf = B.cv2(Mb, Xb, Y, B.MAX_M, B.MAX_X)
            qM, rfM = B.cv1(Mb, Y, B.MAX_M)
            qX, rfX = B.cv1(Xb, Y, B.MAX_X)
            p1se, *_ = B.pick_1se(rf.mean(0), rf)
            pm1, *_ = B.pick_1se(rfM.mean(0), rfM)
            px1, *_ = B.pick_1se(rfX.mean(0), rfX)
            pf, pmA, pxA = amin(rf.mean(0)), amin(rfM.mean(0)), amin(rfX.mean(0))
            a, b = p1se
            for tag, qf, m_, x_ in [
                    ('inherited', qF[p1se], qM[a], qX[b]),
                    ('1-SE',      qF[p1se], qM[pm1], qX[px1]),
                    ('argmin',    qF[pf],   qM[pmA], qX[pxA])]:
                uM, uX = float(qf - x_), float(qf - m_)
                rows.append(dict(case=f'CS2 {rname}', model=name, protocol=tag,
                                 uM=uM, uX=uX,
                                 ratio=uM/uX if uX > 1e-9 else np.nan,
                                 sound='sound' if name == 'U0_correct' else 'BROKEN'))
    return rows


if __name__ == '__main__':
    df = pd.DataFrame(cs1() + cs2())
    df.to_excel('check_submodel_protocol.xlsx', index=False)

    print('\n=== negative uniques (these make the ratio meaningless) ===')
    bad = df[(df.uM < 0) | (df.uX < 0)]
    if len(bad):
        print(bad[['case', 'model', 'protocol', 'uM', 'uX', 'ratio']].round(4).to_string(index=False))
    else:
        print('  none')

    print('\n=== does each protocol separate sound from broken? ===')
    for case in df.case.unique():
        for pr in ['inherited', '1-SE', 'argmin']:
            g = df[(df.case == case) & (df.protocol == pr)]
            lo = g.loc[g.sound == 'sound', 'ratio'].min()
            hi = g.loc[g.sound == 'BROKEN', 'ratio'].max()
            neg = int(((g.uM < 0) | (g.uX < 0)).sum())
            ok = 'SEPARATES' if lo > hi else '*** OVERLAPS ***'
            print(f'  {case:9s} {pr:10s} worst sound {lo:9.3f}  best broken {hi:9.3f}  '
                  f'{ok}{"   (negative uniques!)" if neg else ""}')
    print('\nSaved check_submodel_protocol.xlsx')

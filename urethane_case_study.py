"""
Case study 2: urethane manufacturing semi-batch reactor.

Replication of the in-silico data-generating process and the mechanistic (FP)
model, and assembly of the multiblock data blocks for the M2M methodology.

SOURCES
  [G] Geremia, Marella et al. (2026), Supplementary material S.1  -> urethane2.pdf
      50 calibration + 10 validation batches, 90 h, sampling every 30 min.
  [H] Rossi, Bezzo, Barolo (2026), HyMech, Sec. 4.2 + Appendix A.2 -> urethane.pdf
      Same chemistry; only 2 batches, 80 h. Used here for cross-checking the
      equations and parameter values, not for the experimental design.

The design follows [G] because the M2M methodology needs many batches.

------------------------------------------------------------------ CHEMISTRY
Species: A phenylisocyanate, B butanol, C urethane (product),
         D allophanate, E isocyanurate, S solvent (dimethylsulfoxide).

GROUND-TRUTH ("the process"), [G] S.2-S.17 / [H] A.7-A.21:
    A + B  -> C          r1 = k1 cA cB
    A + C <-> D          r2 = k2 cA cC   (forward),  r3 = k3 cD  (reverse)
    3A     -> E          r4 = k4 cA^2

    dnC/dt = V (r1 - r2 + r3)
    dnD/dt = V (r2 - r3)
    dnE/dt = V  r4

MECHANISTIC / FP MODEL, [G] S.18-S.29 / [H] 15-26:
    identical except the second reaction is treated as IRREVERSIBLE (r3 = 0):
    dnC/dt = V (r1 - r2)
    dnD/dt = V  r2
    dnE/dt = V  r4
  -> the process-model mismatch is exactly the missing reverse step D -> A + C.

Algebraic closure (both models), [G] S.5-S.7, S.9:
    nA = n0A + fv1 nv1A - nC - 2 nD - 3 nE
    nB = n0B + fv2 nv2B - nC - nD
    nS = n0S + fv1 nv1S + fv2 nv2S
    V  = sum_i n_i M_i / rho_i
so the DAE collapses to an explicit 3-state ODE in (nC, nD, nE).

------------------------------------------------------------------ UNITS
kref is tabulated in L/(mol h); concentrations here are mol/m3 and V is m3,
so rate constants are converted by 1e-3 (1 L = 1e-3 m3). kc,2 in L/mol is
converted the same way, which makes k3 = k2/kc come out in 1/h as required.
"""
import numpy as np

R_GAS = 8.314                    # J/(mol K)

# ---------------------------------------------------------------- KNOWN [G] Table S.2
KREF1, KREF2, KREF4 = 1.25e-3, 7.29e-6, 8.80e-7    # L/(mol h)
EA1, EA2, EA4       = 2.94e4, 7.10e4, 2.30e4       # J/mol
KC2                 = 0.217                         # L/mol  ([H] Table A.2; [G] rounds to 0.22)
DELTA_H             = -1.83e4                       # J/mol

# VALIDATION of the kinetics and unit conversions, independent of any data.
# Combining k3 = k2/Kc with both Arrhenius forms gives
#     k3 = (kref2/Kc2) * exp(-(Ea2 - dH)/R * (1/T - 1/Tref))
# i.e. a reverse step with pre-exponential kref2/Kc2 and activation energy
# Ea2 - dH. HyMech DISCOVERED this same reverse reaction from data alone and
# reports (Table 4, 30-min sampling):
#     beta  = (3.3486 +/- 0.1604)e-5 1/h ,  gamma = 89331.67 +/- 533.31 J/mol
# The values implied here are
#     kref2/Kc2 = 3.3594e-5 1/h  (0.07 sd from theirs)
#     Ea2 - dH  = 89314.0  J/mol (0.03 sd from theirs)
# so the rate expressions, Arrhenius parameterisation and L->m3 conversions in
# this file are confirmed correct by an independent route.
TREF                = 363.16                        # K  (Tref,1 = Tref,2 = Tref,4 = Tg2)

# [H] Table A.2 gives the same values at higher precision; [H] also states
# Tref = 363.16 K explicitly. Differences are rounding only:
#   kref1 1.25e-3, kref2 7.29e-6, kref4 8.80e-7, KC2 0.217,
#   Ea1 29440, Ea2 71014, Ea4 23020, dH -18300
KREF1_H, KREF2_H, KREF4_H = 1.25e-3, 7.29e-6, 8.80e-7
EA1_H, EA2_H, EA4_H, KC2_H, DH_H = 29440.0, 71014.0, 23020.0, 0.217, -18300.0

# ---------------------------------------------------------------- KNOWN [H] Table A.3
SPECIES = ['A', 'B', 'C', 'D', 'E', 'S']
M_MOLAR = np.array([0.11911, 0.07412, 0.19323, 0.31234, 0.35733, 0.07806])   # kg/mol
RHO     = np.array([1095.0,  809.0,   1415.0,  1528.0,  1451.0,  1101.0])    # kg/m3
# [G] Table S.1 rounds these (M_A 0.12, rho_A 1.10e3, ...); [H] values used.

# ---------------------------------------------------------------- DESIGN [G] S.1, p.4
BATCH_DURATION_H = 90.0          # h
SAMPLE_INTERVAL_H = 0.5          # h  -> 181 samples per batch
N_CALIBRATION = 50
N_VALIDATION = 10

# Measurement noise. NOTE A CONTRADICTION BETWEEN THE SOURCES:
#   [G] p.4  "Gaussian noise ... with variance sigma^2 = 5e-2 mol for C,
#             5e-5 for D, 5e-6 for E"
#   [H] p.9  "standard deviations sigma_C = 5e-2 mol, sigma_D = 5e-5,
#             sigma_E = 5e-6"
# These differ by a square root. NOISE_IS_VARIANCE selects the reading; it is
# NOT a free parameter to tune -- pick the source you cite and say which.
NOISE_LEVELS = np.array([5e-2, 5e-5, 5e-6])       # for (nC, nD, nE)
NOISE_IS_VARIANCE = False   # [H] reading; see the Bauer-charge note below for why

# ---------------------------------------------------------------- Bauer et al. (2000)
# Sec. 7.1 defines the experiment through seven DESIGN VARIABLES rather than
# through the molar numbers directly:
#     MV1 = (na2 + na2eb)/(na1 + na1ea)          MV2 = na1ea/na1
#     MV3 = na2eb/na1                            Va  = na1 M1/rho1 + na2 M2/rho2 + na6 M6/rho6
#     ga   = (na1 M1 + na2 M2)/(na1 M1 + na2 M2 + na6 M6)
#     gaea = na1ea M1/(na1ea M1 + na6ea M6)      gaeb = na2eb M2/(na2eb M2 + na6eb M6)
# with 1=A, 2=B, 6=S; na* = initial in reactor, na*ea = vessel 1, na*eb = vessel 2.
# These invert in closed form (see design_to_moles). Bauer Sec. 7.4 reports the
# two A-optimal experiments; Geremia et al. take Experiment #1 as nominal.
BAUER_EXPERIMENTS = {
    # MV1, MV2, MV3, ga, gaea, gaeb, Va [m3]          Bauer Sec. 7.4
    1: dict(MV1=0.637635, MV2=15.9264, MV3=10.0, ga=0.8, gaea=0.9, gaeb=1.0, Va=2.09045e-5),
    2: dict(MV1=0.370994, MV2=25.9546, MV3=10.0, ga=0.8, gaea=0.9, gaeb=1.0, Va=9.00762e-6),
    # Bauer Sec. 7.3, starting point of the optimisation (not an experiment)
    'init': dict(MV1=1.0, MV2=0.3, MV3=0.3, ga=0.75, gaea=0.5, gaeb=0.4, Va=2.75e-5),
}

# Bauer Sec. 7.1: batch length, temperature bounds, and the feed parameterisation.
BAUER_BATCH_H = 80.0
T_BOUNDS = (293.16, 473.16)          # K
# feeda, feedb : [0, tf] -> [0, 1], monotonically increasing accumulated feeds;
# fed molar numbers are na1ea*feeda, na6ea*feeda, na2eb*feedb, na6eb*feedb.
# Safety rule: feed rates and heating/cooling rate are zero overnight.

def design_to_moles(d):
    """Invert Bauer's design variables to the seven initial molar numbers.

    Returns a dict in this file's naming: n0_A, n0_B, n0_S (reactor charge),
    nv1_A, nv1_S (vessel 1), nv2_B, nv2_S (vessel 2).
    """
    M1, M2, M6 = M_MOLAR[0], M_MOLAR[1], M_MOLAR[5]
    r1, r2, r6 = RHO[0], RHO[1], RHO[5]
    MV1, MV2, MV3 = d['MV1'], d['MV2'], d['MV3']
    ga, gaea, gaeb, Va = d['ga'], d['gaea'], d['gaeb'], d['Va']

    # everything scales linearly with na1; solve the scale from Va
    u_na1ea = MV2                       # na1ea / na1
    u_na2eb = MV3                       # na2eb / na1
    u_na2 = MV1*(1.0 + MV2) - MV3       # na2  / na1
    # Bauer's Experiment #2 gives u_na2 = -1.3e-4, i.e. zero initial butanol in
    # the reactor to within the precision of the reported design variables.
    # Clamp that rounding artefact; reject anything genuinely negative.
    if -1e-3 < u_na2 < 0.0:
        u_na2 = 0.0
    elif u_na2 < 0:
        raise ValueError('MV1/MV2/MV3 imply a negative initial butanol charge')
    u_na6 = (M1 + u_na2*M2)*(1.0 - ga)/(ga*M6)          # na6 / na1
    u_Va = M1/r1 + u_na2*M2/r2 + u_na6*M6/r6            # Va  / na1
    na1 = Va/u_Va

    na2, na6 = u_na2*na1, u_na6*na1
    na1ea, na2eb = u_na1ea*na1, u_na2eb*na1
    na6ea = na1ea*M1*(1.0 - gaea)/(gaea*M6) if gaea > 0 else float('inf')
    na6eb = na2eb*M2*(1.0 - gaeb)/(gaeb*M6) if gaeb > 0 else float('inf')
    return dict(n0_A=na1, n0_B=na2, n0_S=na6,
                nv1_A=na1ea, nv1_S=na6ea, nv2_B=na2eb, nv2_S=na6eb)

def moles_to_design(m):
    """Forward map, for round-trip verification of design_to_moles."""
    M1, M2, M6 = M_MOLAR[0], M_MOLAR[1], M_MOLAR[5]
    r1, r2, r6 = RHO[0], RHO[1], RHO[5]
    a, b, s = m['n0_A'], m['n0_B'], m['n0_S']
    ae, se, be, sb = m['nv1_A'], m['nv1_S'], m['nv2_B'], m['nv2_S']
    return dict(MV1=(b + be)/(a + ae), MV2=ae/a, MV3=be/a,
                ga=(a*M1 + b*M2)/(a*M1 + b*M2 + s*M6),
                gaea=ae*M1/(ae*M1 + se*M6),
                gaeb=be*M2/(be*M2 + sb*M6) if (be*M2 + sb*M6) > 0 else float('nan'),
                Va=a*M1/r1 + b*M2/r2 + s*M6/r6)

# ================================================================= NOT SPECIFIED
# Neither PDF reports these. They come from Bauer et al. (2000), Experiment #1,
# which is cited but not reproduced in either source. Figure S.1 of [G] shows
# the input ranges and switching times graphically only -- the numbers are not
# in the extractable text.
#
# Nothing below is invented. The script refuses to generate data until they are
# supplied, so that no fabricated constant can silently enter the case study.
# RESOLVED from Bauer et al. (2000) Sec. 7.4, Experiment #1, via design_to_moles:
#   n0_A  = 0.094246   nv1_A = 1.500999   nv2_B = 0.942460
#   n0_B  = 0.074724   nv1_S = 0.254482   nv2_S = 0.0   (gaeb = 1, pure butanol)
#   n0_S  = 0.053690
# Total A available 1.5952 mol, total B 1.0172 mol; reactor fills 20.9 -> 288.6 mL.
# Since B is limiting, nC cannot exceed ~1.02 mol. That settles the noise
# contradiction flagged above: sigma_C = 5e-2 mol is ~5% of the maximum, whereas
# variance 5e-2 (sd 0.224) would be ~22% -- so [H]'s reading (standard
# deviations) is the consistent one and NOISE_IS_VARIANCE should be False.
# ---------------------------------------------------------------- CONTROL PROFILES
# READ OFF GEREMIA FIGURE S.1 (nominal solid line, admissible band dashed).
# These are the only quantities in this file taken from a graphic rather than
# from text or a table, so they carry a reading uncertainty of roughly
# +/-0.02 on the fractions, +/-10 K on temperature and +/-2 h on switching times.
# Bauer Sec. 7.4's own plots for Experiment #1 agree with the nominal values
# (isocyanate plateau ~0.97 vs 0.96, butanol ~0.68/0.92 vs 0.63/0.82,
# T ~470 then ~450 vs 473 then 437), which cross-checks the reading.
#
# Each input is piecewise linear through plateaus. Ranges are (lo, nominal, hi);
# a batch is drawn by sampling each entry uniformly on [lo, hi].
FV1 = dict(t_ramp=(5.0, 7.0, 9.0), p1=(0.88, 0.96, 1.00),
           t_hold=(44.0, 48.0, 52.0), t_rise=(55.0, 57.0, 60.0), p2=1.0)
FV2 = dict(t_ramp=(4.0, 6.0, 8.0), p1=(0.45, 0.63, 0.80),
           t_hold1=(20.0, 22.0, 25.0), t_rise1=(27.0, 30.0, 34.0),
           p2=(0.72, 0.82, 0.92),
           t_hold2=(44.0, 48.0, 52.0), t_rise2=(55.0, 57.0, 60.0), p3=1.0)
# TEMPERATURE RANGE. Bauer Sec. 7.1 constrains T to [293.16, 473.16] K as part
# of his optimum-experimental-design problem. The adapted case studies do not
# carry that constraint over: Geremia Fig. S.1c places the upper dashed limit
# near 500 K, and HyMech Fig. 7(c) shows a batch peaking at about 225 C
# (= 498 K). The upper limit is therefore taken from the adapted studies, not
# from Bauer. This matters because the process-model mismatch is strongly
# temperature-dependent -- roughly 0.1 % at 363 K, 5 % at 423 K, 24 % at 453 K
# and 48 % at 473 K -- so capping at Bauer's bound would suppress exactly the
# region where the missing reverse reaction becomes observable.
TPROF = dict(T0=293.16, t_ramp=(5.0, 7.0, 9.0), p1=(455.0, 473.0, 500.0),
             t_hold=(42.0, 45.0, 50.0), t_fall=(55.0, 57.0, 60.0),
             p2=(432.0, 437.0, 460.0))

# ---------------------------------------------------------------- HyMech Fig. 7(c)
# The two batches shown in Rossi et al. Fig. 7(c), read directly off that figure.
# Temperature there is plotted in degrees Celsius; it is converted to Kelvin here.
# These are explicit profiles, not draws from the Geremia band, and they are the
# ones to use when reproducing that figure. Batch length is 80 h, as in [H].
HYMECH_BATCHES = [
    dict(name='Batch 1',
         fv1_t=[0.0, 7.0, 50.0, 57.0, 80.0], fv1_v=[0.0, 0.92, 0.92, 1.00, 1.00],
         fv2_t=[0.0, 5.0, 22.0, 28.0, 50.0, 57.0, 80.0],
         fv2_v=[0.0, 0.72, 0.72, 0.85, 0.85, 1.00, 1.00],
         T_t=[0.0, 7.0, 48.0, 57.0, 80.0], T_C=[20.0, 225.0, 225.0, 180.0, 180.0]),
    dict(name='Batch 2',
         fv1_t=[0.0, 6.0, 48.0, 55.0, 80.0], fv1_v=[0.0, 0.93, 0.93, 1.00, 1.00],
         fv2_t=[0.0, 5.0, 25.0, 32.0, 55.0, 60.0, 80.0],
         fv2_v=[0.0, 0.62, 0.62, 0.78, 0.78, 1.00, 1.00],
         T_t=[0.0, 7.0, 48.0, 57.0, 80.0], T_C=[20.0, 195.0, 195.0, 165.0, 165.0]),
]

def hymech_inputs(spec):
    """Input function for one of the explicit HyMech Fig. 7(c) batches."""
    T_K = [c + 273.15 for c in spec['T_C']]
    def f(t):
        return (float(np.interp(t, spec['fv1_t'], spec['fv1_v'])),
                float(np.interp(t, spec['fv2_t'], spec['fv2_v'])),
                float(np.interp(t, spec['T_t'], T_K)))
    f.spec = spec
    return f

def _draw(rng, spec, nominal_only):
    """Uniform draw from a (lo, nominal, hi) triple, or the nominal value."""
    if not isinstance(spec, tuple):
        return spec
    lo, nom, hi = spec
    return nom if nominal_only else float(rng.uniform(lo, hi))

def sample_inputs(rng, duration_h, nominal_only=False):
    """Build one batch's piecewise-linear f_v1, f_v2 and T profiles.

    Beyond the last breakpoint (~60 h) every profile is constant, so extending
    the batch from Bauer's 80 h to Geremia's stated 90 h simply holds the final
    plateau -- no extrapolation of the figure is involved.
    """
    d = lambda s: _draw(rng, s, nominal_only)
    a_t1, a_p1, a_t2, a_t3 = d(FV1['t_ramp']), d(FV1['p1']), d(FV1['t_hold']), d(FV1['t_rise'])
    b_t1, b_p1 = d(FV2['t_ramp']), d(FV2['p1'])
    b_t2, b_t3, b_p2 = d(FV2['t_hold1']), d(FV2['t_rise1']), d(FV2['p2'])
    # f_v2 is an ACCUMULATED feed fraction and must not decrease. The two plateau
    # ranges read off Fig. S.1b overlap ([0.45,0.80] and [0.72,0.92]), so a draw
    # with p1 > p2 would make the profile run backwards. Enforce the ordering.
    b_p2 = max(b_p2, b_p1)
    b_t4, b_t5 = d(FV2['t_hold2']), d(FV2['t_rise2'])
    T_t1, T_p1, T_t2, T_t3, T_p2 = (d(TPROF['t_ramp']), d(TPROF['p1']),
                                    d(TPROF['t_hold']), d(TPROF['t_fall']), d(TPROF['p2']))
    end = duration_h
    fa_t = [0.0, a_t1, a_t2, a_t3, end]
    fa_v = [0.0, a_p1, a_p1, FV1['p2'], FV1['p2']]
    fb_t = [0.0, b_t1, b_t2, b_t3, b_t4, b_t5, end]
    fb_v = [0.0, b_p1, b_p1, b_p2, b_p2, FV2['p3'], FV2['p3']]
    T_t = [0.0, T_t1, T_t2, T_t3, end]
    T_v = [TPROF['T0'], T_p1, T_p1, T_p2, T_p2]

    def f(t):
        return (float(np.interp(t, fa_t, fa_v)),
                float(np.interp(t, fb_t, fb_v)),
                float(np.interp(t, T_t, T_v)))
    f.spec = dict(fv1_t=fa_t, fv1_v=fa_v, fv2_t=fb_t, fv2_v=fb_v, T_t=T_t, T_v=T_v)
    return f

def missing():  # retained for provenance; all entries now resolved
    return [k for k, v in UNKNOWN.items() if v is None]

# ---------------------------------------------------------------- kinetics
def rate_constants(T):
    """Arrhenius constants at temperature T [K], converted to m3/(mol h) or 1/h."""
    arr = lambda kref, Ea: kref*1e-3*np.exp(-Ea/R_GAS*(1.0/T - 1.0/TREF))
    k1 = arr(KREF1, EA1)
    k2 = arr(KREF2, EA2)
    k4 = arr(KREF4, EA4)
    Kc = KC2*1e-3*np.exp(-DELTA_H/R_GAS*(1.0/T - 1.0/TREF))   # m3/mol
    k3 = k2/Kc                                                 # 1/h
    return k1, k2, k3, k4

def closure(nC, nD, nE, fv1, fv2, ic):
    """Algebraic part: recover nA, nB, nS and the reactor volume."""
    nA = ic['n0_A'] + fv1*ic['nv1_A'] - nC - 2.0*nD - 3.0*nE
    nB = ic['n0_B'] + fv2*ic['nv2_B'] - nC - nD
    nS = ic['n0_S'] + fv1*ic['nv1_S'] + fv2*ic['nv2_S']
    n = np.array([nA, nB, nC, nD, nE, nS])
    V = float(np.sum(n*M_MOLAR/RHO))
    return n, V

def _rhs(t, y, inputs, ic, reversible):
    nC, nD, nE = y
    fv1, fv2, T = inputs(t)
    n, V = closure(nC, nD, nE, fv1, fv2, ic)
    if V <= 0:
        return [0.0, 0.0, 0.0]
    nA, nB = max(n[0], 0.0), max(n[1], 0.0)
    cA, cB, cC, cD = nA/V, nB/V, max(nC, 0.0)/V, max(nD, 0.0)/V
    k1, k2, k3, k4 = rate_constants(T)
    r1 = k1*cA*cB
    r2 = k2*cA*cC
    r3 = k3*cD if reversible else 0.0
    r4 = k4*cA*cA
    return [V*(r1 - r2 + r3), V*(r2 - r3), V*r4]

def ground_truth_rhs(t, y, inputs, ic):
    """The process: second reaction is an equilibrium."""
    return _rhs(t, y, inputs, ic, reversible=True)

def fp_model_rhs(t, y, inputs, ic):
    """The available first-principles model: second reaction irreversible."""
    return _rhs(t, y, inputs, ic, reversible=False)

# ---------------------------------------------------------------- simulation
def make_input_fn(breakpoints, fv1_vals, fv2_vals, T_vals):
    """Piecewise-linear interpolation of the three manipulated inputs."""
    bp = np.asarray(breakpoints, float)
    def f(t):
        return (float(np.interp(t, bp, fv1_vals)),
                float(np.interp(t, bp, fv2_vals)),
                float(np.interp(t, bp, T_vals)))
    return f

def simulate(rhs, inputs, ic, t_eval, rtol=1e-8, atol=1e-10):
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), [0.0, 0.0, 0.0],
                    t_eval=t_eval, args=(inputs, ic), method='LSODA',
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T                     # (n_times, 3) -> nC, nD, nE

def add_noise(traj, rng):
    sd = np.sqrt(NOISE_LEVELS) if NOISE_IS_VARIANCE else NOISE_LEVELS
    return traj + rng.normal(0.0, sd, traj.shape)

# ---------------------------------------------------------------- block assembly
def build_blocks(inputs_per_batch, meas_traj, fp_traj, t_grid, y_index=-1,
                 response='nC_final'):
    """Assemble the multiblock matrices for the M2M methodology.

    Mapping from the two-stage reactor case study to this one:

      X1  operating-conditions block  -- the manipulated input trajectories
          fv1(t), fv2(t), T(t), unfolded batch-wise. Known before the batch runs.
      M   mechanistic block           -- FP-model-predicted nC, nD, nE
          trajectories, unfolded; the KD inputs (fv1, fv2, T) are appended here
          and removed from X, per the causal-ordering rule of the methodology.
      X   measured block              -- measured nC, nD, nE trajectories.
      Y   response                    -- final urethane, or selectivity.

    Because every batch has the same duration and sampling grid, no trajectory
    truncation is needed -- unlike the two-stage reactor case study.
    """
    n_b = len(meas_traj)
    n_t = len(t_grid)
    unfold = lambda A: A.transpose(0, 2, 1).reshape(n_b, -1)   # species-major

    U = np.array([[inputs_per_batch[i](t) for t in t_grid] for i in range(n_b)])
    X1 = U.transpose(0, 2, 1).reshape(n_b, -1)                 # fv1|fv2|T unfolded
    M = np.hstack([unfold(np.asarray(fp_traj)), X1])           # KD inputs moved into M
    X = unfold(np.asarray(meas_traj))

    last = np.asarray(meas_traj)[:, y_index, :]                # (n_b, 3) at t_end
    if response == 'nC_final':
        Y = last[:, 0:1]
    elif response == 'selectivity':
        Y = (last[:, 0]/last.sum(1)).reshape(-1, 1)*100.0
    else:
        raise ValueError(response)

    cols = dict(
        X1=[f'{v}_t{j+1}' for v in ('fv1', 'fv2', 'T') for j in range(n_t)],
        M=[f'{s}_fit_t{j+1}' for s in ('nC', 'nD', 'nE') for j in range(n_t)]
          + [f'{v}_t{j+1}' for v in ('fv1', 'fv2', 'T') for j in range(n_t)],
        X=[f'{s}_t{j+1}' for s in ('nC', 'nD', 'nE') for j in range(n_t)],
        Y=[response])
    return X1, M, X, Y, cols

# ---------------------------------------------------------------- self-test
def smoke_test():
    """Verify the equations integrate and the mass balances close, using the
    REAL Bauer Experiment #1 charge. The temperature profile is illustrative
    (Bauer's is graphical), so this is a structural check, not the case-study
    data."""
    ic = design_to_moles(BAUER_EXPERIMENTS[1])
    t = np.arange(0.0, BAUER_BATCH_H + 1e-9, SAMPLE_INTERVAL_H)
    bp = [0.0, 20.0, 50.0, 80.0]
    inp = make_input_fn(bp, [0.0, 0.4, 0.8, 1.0], [0.0, 0.4, 0.8, 1.0],
                        [423.0, 453.0, 453.0, 433.0])
    gt = simulate(ground_truth_rhs, inp, ic, t)
    fp = simulate(fp_model_rhs, inp, ic, t)
    print("smoke test: Bauer Experiment #1 charge, illustrative temperature profile")
    print(f"  samples per batch          : {len(t)}")
    for lbl, tr in [('ground truth', gt), ('FP model', fp)]:
        print(f"  {lbl:13s} final nC={tr[-1,0]:.5f}  nD={tr[-1,1]:.5f}  nE={tr[-1,2]:.6e}")
    fv1, fv2, T = inp(t[-1])
    n, V = closure(*gt[-1], fv1, fv2, ic)
    print(f"  closure at t_end           : nA={n[0]:.4f} nB={n[1]:.4f} nS={n[5]:.4f} V={V:.6f} m3")
    bal_A = n[0] + gt[-1,0] + 2*gt[-1,1] + 3*gt[-1,2] - (ic['n0_A'] + fv1*ic['nv1_A'])
    bal_B = n[1] + gt[-1,0] + gt[-1,1] - (ic['n0_B'] + fv2*ic['nv2_B'])
    print(f"  atom balance residuals     : A {bal_A:+.2e}   B {bal_B:+.2e}")
    print(f"  mismatch (GT - FP) final nC: {gt[-1,0]-fp[-1,0]:+.5f} mol"
          f"   nD: {gt[-1,1]-fp[-1,1]:+.5f} mol")
    k1, k2, k3, k4 = rate_constants(TREF)
    print(f"  k at Tref                  : k1={k1:.4e} k2={k2:.4e} k3={k3:.4e} k4={k4:.4e}")

# ---------------------------------------------------------------- generation
def generate(n_batches, seed, duration_h=BATCH_DURATION_H, nominal_only=False):
    """Simulate n_batches of the PROCESS (ground truth) and of the FP model.

    The FP model is driven by the same inputs and the same initial charge, so the
    only difference between the two is the missing reverse reaction.
    """
    rng = np.random.default_rng(seed)
    ic = design_to_moles(BAUER_EXPERIMENTS[1])
    t = np.arange(0.0, duration_h + 1e-9, SAMPLE_INTERVAL_H)
    inputs, meas, clean, fp = [], [], [], []
    for _ in range(n_batches):
        f = sample_inputs(rng, duration_h, nominal_only)
        gt = simulate(ground_truth_rhs, f, ic, t)
        mm = simulate(fp_model_rhs, f, ic, t)
        inputs.append(f); clean.append(gt); fp.append(mm)
        meas.append(add_noise(gt, rng))
    return t, inputs, np.array(meas), np.array(clean), np.array(fp), ic

def save(path, X1, M, X, Y, cols):
    import pandas as pd
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(X1, columns=cols['X1']).to_excel(w, sheet_name='X1', index=False)
        pd.DataFrame(M,  columns=cols['M']).to_excel(w,  sheet_name='M',  index=False)
        pd.DataFrame(X,  columns=cols['X']).to_excel(w,  sheet_name='X',  index=False)
        pd.DataFrame(Y,  columns=cols['Y']).to_excel(w,  sheet_name='Y',  index=False)

if __name__ == '__main__':
    print(__doc__.split('SOURCES')[0].strip())
    print()
    smoke_test()

    print("\n--- Bauer Experiment #1 initial charge (derived from the design variables) ---")
    ic = design_to_moles(BAUER_EXPERIMENTS[1])
    for k, v in ic.items():
        print(f"   {k:7s} = {v:10.6f} mol")

    print("\n--- generating datasets ---")
    for tag, n, seed, fn in [('calibration', N_CALIBRATION, 101, 'UR_calibration.xlsx'),
                             ('validation',  N_VALIDATION,  202, 'UR_validation.xlsx')]:
        t, inp, meas, clean, fp, _ = generate(n, seed)
        X1, M, X, Y, cols = build_blocks(inp, meas, fp, t)
        save(fn, X1, M, X, Y, cols)
        err = np.abs(clean - fp)
        print(f"  {tag:11s} n={n:3d}  X1{X1.shape} M{M.shape} X{X.shape} Y{Y.shape}  -> {fn}")
        print(f"              Y (final nC): {Y.mean():.4f} +/- {Y.std(ddof=1):.4f} mol"
              f"  [{Y.min():.4f}, {Y.max():.4f}]")
        print(f"              mean |process - FP| over trajectory:"
              f"  nC {err[:,:,0].mean():.4f}   nD {err[:,:,1].mean():.5f}   nE {err[:,:,2].mean():.3e} mol")
    print("\nBlocks: X1 = inputs, M = FP-predicted trajectories + KD inputs,"
          " X = measured trajectories, Y = final nC.")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# PARAMETERS
# =========================
R = 8.314
T_ref = 363.15

kref1 = 1.25e-3
kref2 = 7.29e-6
kref4 = 8.80e-7

Kc2 = 0.217

Ea1 = 29440
Ea2 = 71014
Ea4 = 23020

dH = 18300

# =========================
# CONSTANT TEMPERATURE
# =========================
T = 393.15

# =========================
# RATE CONSTANTS
# =========================
k1 = kref1 * np.exp(
    -Ea1 / R * (1/T - 1/T_ref)
)

k2 = kref2 * np.exp(
    -Ea2 / R * (1/T - 1/T_ref)
)

Kc = Kc2 * np.exp(
    -dH / R * (1/T - 1/T_ref)
)

k3 = k2 / Kc

k4 = kref4 * np.exp(
    -Ea4 / R * (1/T - 1/T_ref)
)

# =========================
# REACTOR VOLUME
# =========================
V = 1.0  # L

# =========================
# ODE SYSTEM
# =========================
def reactor(t, y):

    CA, CB, CC, CD, CE = y

    r1 = k1 * CA * CB
    r2 = k2 * CA * CC
    r3 = k3 * CD
    r4 = k4 * CA**2

    dCA = -r1 - r2 - 2*r4
    dCB = -r1
    dCC = r1 - r2 + r3
    dCD = r2 - r3
    dCE = r4

    return [dCA, dCB, dCC, dCD, dCE]

# =========================
# INITIAL CONDITIONS
# =========================
CA0 = 2.0
CB0 = 2.0
CC0 = 0.0
CD0 = 0.0
CE0 = 0.0

y0 = [CA0, CB0, CC0, CD0, CE0]

# =========================
# SIMULATION
# =========================
t_eval = np.arange(
    0,
    80.5,
    0.5
)  # 80 h, every 30 min

sol = solve_ivp(
    reactor,
    [0, 80],
    y0,
    t_eval=t_eval,
    method="LSODA"
)

# =========================
# EXTRACT RESULTS
# =========================
time = sol.t

CA = sol.y[0]
CB = sol.y[1]
CC = sol.y[2]
CD = sol.y[3]
CE = sol.y[4]

# =========================
# PLOT C, D, E
# =========================
plt.figure(figsize=(10,6))

plt.plot(
    time,
    CC,
    linewidth=2,
    label="C"
)

plt.plot(
    time,
    CD,
    linewidth=2,
    label="D"
)

plt.plot(
    time,
    CE,
    linewidth=2,
    label="E"
)

plt.xlabel(
    "Time [h]",
    fontsize=14
)

plt.ylabel(
    "Concentration [mol/L]",
    fontsize=14
)

plt.title(
    "Concentration Profiles of Species C, D and E",
    fontsize=16
)

plt.grid(alpha=0.3)

plt.legend(
    fontsize=12
)

plt.tight_layout()
plt.show()
import numpy as np
import sdeint
np.random.seed(42)

# --- Parameters from paper ---
V1 = 0.1  # m³ (Stage 1 reactor volume)
V2 = 0.2  # m³ (Stage 2 reactor volume)
VA0 = 0.1  # m³ (initial volume of A solution)
VD0 = 0.1  # m³ (volume of D solution added)

# Initial concentrations
CA0 = 2000.0  # mol/m³ (A in stage 1)
CD0 = 1900.0  # mol/m³ (D in the solution added to stage 2)

# Kinetic parameters
A1, A2 = 28, 40      # s⁻¹
A3, A4 = 10, 20      # m⁶/(mol²·s)
Ea1_mean, Ea2_mean, Ea3_mean, Ea4_mean = 20000, 30000, 50000, 55000   # J/mol
CV = 0.01           # Coefficient of Variation
R = 8.314           # J/(mol·K)

# Drift term of SDE
def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    # Stage 1 model equations
    dCAdt = -k1 * CA
    dCBdt = k1 * CA - k2 * CB
    dCCdt = k2 * CB
    return np.array([dCAdt, dCBdt, dCCdt])

# Noise term of SDE
def stage1_noise(y, t, sigma1):
    # sigma is the diffusion coefficient
    return np.diag(sigma1)

def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    # Stage 2 model equations
    dCAdt = 0
    dCBdt = -k3 * CB * CD**2
    dCCdt = -k4 * CC * CD**2
    dCDdt = -2 * k3 * CB * CD**2 - 2 * k4 * CC * CD**2
    dCEdt = k3 * CB * CD**2
    dCFdt = k4 * CC * CD**2
    return np.array([dCAdt, dCBdt, dCCdt, dCDdt, dCEdt, dCFdt])

def stage2_noise(y, t, sigma2):
    return np.diag(sigma2)

def simulation(sigma1, sigma2):
    # Fix activation energies
    Ea1 = Ea1_mean
    Ea2 = Ea2_mean
    Ea3 = Ea3_mean
    Ea4 = Ea4_mean
    # Fix temperature and time as the average of the range
    T1 = 300
    T2 = 350
    t1_seconds = 500
    t2_seconds = 250
    # Rate constants
    k1 = A1 * np.exp(-Ea1 / (R * T1))
    k2 = A2 * np.exp(-Ea2 / (R * T1))
    k3 = A3 * np.exp(-Ea3 / (R * T2))
    k4 = A4 * np.exp(-Ea4 / (R * T2))

    # Generate time trajectories for Stage 1
    t1v = np.linspace(0, t1_seconds, 1000)
    # Initial input to Stage 1
    y0_1 = [CA0, 0, 0]
    # Retrieving solutions of Stage 1 SDEs
    sol1 = sdeint.itoint(lambda y, t: stage1_drift(y, t, k1, k2),
                        lambda y, t: stage1_noise(y,t, sigma1), y0_1, t1v)
    # Generate time trajectories for Stage 2
    t2v = np.linspace(0, t2_seconds, 1000)
    # Concentrations at Stage 1 completion
    CA1f, CB1f, CC1f = sol1[-1]
    # Dilution due to the addition of CD0
    dilution = V1 / V2
    CA2_0 = CA1f * dilution
    CB2_0 = CB1f * dilution
    CC2_0 = CC1f * dilution
    CD2_0 = CD0
    CE2_0 = 0.0
    CF2_0 = 0.0
    # Initial input to Stage 2
    y0_2 = [CA2_0, CB2_0, CC2_0, CD2_0, CE2_0, CF2_0]
    # Retrieving solutions of Stage 2 SDEs
    sol2 = sdeint.itoint(lambda y, t: stage2_drift(y, t, k3, k4),
        lambda y, t: stage2_noise(y, t, sigma2), y0_2, t2v)
    return (t1v/60, sol1), (t2v/60, sol2)

N = 100   # Number of simulations
target_cv = 0.01  # Target coefficient of variation of 1%
tol = 1e-3    # Tolerance level for diffusion coefficient convergence
max_iter = 20   # Number of iterations for diffusion coefficient calibration
# Initial guess of diffusion coefficients
sigma1 = [0.0] * 3
sigma2 = [0.0] * 6
# Optimal values of diffusion coefficients after calibration
opt_sigma1 = [0.0] * 3
opt_sigma2 = [0.0] * 6
# Stage 1 diffusion coefficients calibration:
for t in range(len(sigma1)):    # For loop for each component (3) of stage 1
    # Reset diffusion coefficients of all other components
    sigma1 = [0.0] * 3
    sigma2 = [0.0] * 6
    # Initial guess of diffusion coefficient of current component
    sigma1[t] = 1e-3
    for j in range(max_iter):   # For loop for calibration
        final_c1 = []
        for i in range(N):  # For loop to generate N simulations
            (t1v, sol1), (t2v, sol2) = simulation(sigma1, sigma2)
            # Store concentrations at final trajectory for each simulation
            final_c1.append(sol1[-1])
        final_c1 = np.asarray(final_c1)

        # Stage 1 CV
        mean1 = np.mean(final_c1, axis=0)
        std1 = np.std(final_c1, axis=0, ddof=1)
        cv1 = std1 / mean1

        print(f"\nComponent {t}, Iteration {j}")
        print(f"Stage 1 CVs: {cv1}")
        print(f"sigma[{t}] = {sigma1[t]:.6f}, CV[{t}] = {cv1[t]:.5f}")

        # Avoid division by zero
        if cv1[t] == 0:
            sigma1[t] *= 10
            continue

        # Update rule
        sigma1[t] *= target_cv / cv1[t]

        # Check convergence
        if abs(cv1[t] - target_cv) < tol:
            break
    # Store calibrated diffusion coefficient values
    opt_sigma1[t]=sigma1[t]

# Stage 2 diffusion coefficients calibration:
for t in range(len(sigma2)):    # For loop for each component (6) of stage 2
    # Reset diffusion coefficients of all other components
    sigma1 = [0.0] * 3
    sigma2 = [0.0] * 6
    # Initial guess of diffusion coefficient of current component
    sigma2[t] = 1e-3
    for j in range(max_iter):   # For loop for calibration
        final_c2 = []
        for i in range(N):  # For loop to generate N simulations
            (t1v, sol1), (t2v, sol2) = simulation(sigma1, sigma2)
            # Store concentrations at final trajectory for each simulation
            final_c2.append(sol2[-1])
        final_c2=np.asarray(final_c2)

        # Stage 2 CV
        mean2 = np.mean(final_c2, axis=0)
        std2 = np.std(final_c2, axis=0, ddof=1)
        cv2 = std2 / mean2

        print(f"\nComponent {t}, Iteration {j}")
        print(f"Stage 2 CVs: {cv2}")
        print(f"sigma[{t}] = {sigma2[t]:.6f}, CV[{t}] = {cv2[t]:.5f}")

        # Avoid division by zero
        if cv2[t] == 0:
            sigma2[t] *= 10
            continue

        # Update rule
        sigma2[t] *= target_cv / cv2[t]

        # Check convergence
        if abs(cv2[t] - target_cv) < tol:
            break
    # Store calibrated diffusion coefficient values
    opt_sigma2[t]=sigma2[t]
print(np.asarray(opt_sigma1))
print(np.asarray(opt_sigma2))

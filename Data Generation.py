import numpy as np
import matplotlib.pyplot as plt
import sdeint
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# --- Parameters from paper ---
V1 = 0.1  # m³ (Stage 1 reactor volume)
V2 = 0.2  # m³ (Stage 2 reactor volume)
VA0 = 0.1  # m³ (initial volume of A solution)
VD0 = 0.1  # m³ (volume of D solution added)

# Initial concentrations
CA0 = 2000.0  # mol/m³ (A in stage 1)
CD0 = 1900.0  # mol/m³ (D in the solution added to stage 2)

# Kinetic parameters
A1, A2 = 28, 40  # s⁻¹
A3, A4 = 10, 20  # m⁶/(mol²·s)
Ea1_mean, Ea2_mean, Ea3_mean, Ea4_mean = 20000, 30000, 50000, 55000  # J/mol
CV = 0.01  # Coefficient of Variation
R = 8.314  # J/(mol·K)

opt_sigma1 = [0.025, 0.9, 0.08]
opt_sigma2 = [0.25, 0.03, 0.022, 0.07, 0.54, 0.04]  # sigma_A corrected from 0.006 -> 0.25 (see Var[A(t)]=sigma_A^2*t vs author X2-M2 residual variance)

def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    # Stage 1 model equations
    dCAdt = -k1 * CA
    dCBdt = k1 * CA - k2 * CB
    dCCdt = k2 * CB
    return np.array([dCAdt, dCBdt, dCCdt])


def stage1_noise(y, t):
    # Diffusion coefficients obtained from calibration
    return np.diag(opt_sigma1)


def stage2_drift(y, t, k3, k4):
    CA, CB, CC, CD, CE, CF = y
    # Stage 2 model equations
    dCAdt = 0
    dCBdt = -k3 * CB * CD ** 2
    dCCdt = -k4 * CC * CD ** 2
    dCDdt = -2 * k3 * CB * CD ** 2 - 2 * k4 * CC * CD ** 2
    dCEdt = k3 * CB * CD ** 2
    dCFdt = k4 * CC * CD ** 2
    return np.array([dCAdt, dCBdt, dCCdt, dCDdt, dCEdt, dCFdt])


def stage2_noise(y, t):
    # Diffusion coefficients obtained from calibration
    return np.diag(opt_sigma2)


def one_sde_simulation(data_type="training"):
    # Generate sample activation energies
    Ea1 = np.random.normal(Ea1_mean, CV * Ea1_mean)
    Ea2 = np.random.normal(Ea2_mean, CV * Ea2_mean)
    Ea3 = np.random.normal(Ea3_mean, CV * Ea3_mean)
    Ea4 = np.random.normal(Ea4_mean, CV * Ea4_mean)

    # Generate sample time and temperature
    T1 = np.random.uniform(290, 310)
    # Use different T2 range based on data type
    if data_type == "training":
        T2 = np.random.uniform(330, 370)
    else:  # testing
        T2 = np.random.uniform(362, 382)
    t1_seconds = np.random.uniform(350, 650)
    t2_seconds = np.random.uniform(100, 400)
    # Rate constants
    k1 = A1 * np.exp(-Ea1 / (R * T1))
    k2 = A2 * np.exp(-Ea2 / (R * T1))
    k3 = A3 * np.exp(-Ea3 / (R * T2))
    k4 = A4 * np.exp(-Ea4 / (R * T2))

    # --- ADD NOISE TO INITIAL CONDITIONS HERE ---
    CA0_sample = np.random.normal(CA0, 0.05 * CA0)
    CD0_sample = np.random.normal(CD0, 0.05 * CD0)

    # Generate time trajectories for Stage 1
    t1v = np.linspace(0, t1_seconds, 1000)
    # Initial input of A, B, C concentrations to Stage 1
    y0_1 = np.array([CA0_sample, 0.0, 0.0])
    # Retrieving solutions of Stage 1 SDEs
    sol1 = sdeint.itoint(lambda y, t: stage1_drift(y, t, k1, k2),
                         lambda y, t: stage1_noise(y, t), y0_1, t1v)
    # Generate time trajectories for Stage 2
    t2v = np.linspace(0, t2_seconds, 1000)
    # Concentrations at Stage 1 completion
    CA1f, CB1f, CC1f = sol1[-1]
    # Dilution of concentrations due to the addition of CD0
    dilution = V1 / V2
    CA2_0 = CA1f * dilution
    CB2_0 = CB1f * dilution
    CC2_0 = CC1f * dilution
    CD2_0 = CD0_sample
    CE2_0 = 0.0
    CF2_0 = 0.0
    # Initial input of A, B, C, D, E, F concentrations to Stage 2
    y0_2 = np.array([CA2_0, CB2_0, CC2_0, CD2_0, CE2_0, CF2_0])
    # Retrieving solutions of Stage 2 SDEs
    sol2 = sdeint.itoint(lambda y, t: stage2_drift(y, t, k3, k4),
                         lambda y, t: stage2_noise(y, t), y0_2, t2v)
    # Store parameters
    params = {
        'CA0_sample': CA0_sample,
        'CD0_sample': CD0_sample,
        'T1': T1,
        'T2': T2,
        't1_seconds': t1_seconds,
        't2_seconds': t2_seconds,
        'Ea1': Ea1,
        'Ea2': Ea2,
        'Ea3': Ea3,
        'Ea4': Ea4,
        'data_type': data_type
    }
    # Return solutions and parameters
    return (t1v / 60, sol1), (t2v / 60, sol2), params


if __name__ == "__main__":
    # SET GLOBAL SEED HERE FOR REPRODUCIBILITY
    GLOBAL_SEED = 10  # Change this to any integer you want
    np.random.seed(GLOBAL_SEED)
    N_training = 100
    N_testing = 25
    N_total = N_training + N_testing

    # Introduce arrays to store solutions and parameters across all simulations
    all_stage1 = []
    all_stage2 = []
    all_params = []  # Store all parameters
    data_labels = []  # Store data type labels

    # Generate training data
    for i in range(N_training):
        (t1v, sol1), (t2v, sol2), params = one_sde_simulation(data_type="training")
        all_stage1.append((t1v, sol1))
        all_stage2.append((t2v, sol2))
        all_params.append(params)
        data_labels.append("training")

    # Generate testing data
    for i in range(N_testing):
        (t1v, sol1), (t2v, sol2), params = one_sde_simulation(data_type="testing")
        all_stage1.append((t1v, sol1))
        all_stage2.append((t2v, sol2))
        all_params.append(params)
        data_labels.append("testing")

    # Extract parameter arrays for easy access
    CA0_values = np.array([p['CA0_sample'] for p in all_params])
    T1_values = np.array([p['T1'] for p in all_params])
    t1_values = np.array([p['t1_seconds'] / 60 for p in all_params])  # Convert to minutes
    T2_values = np.array([p['T2'] for p in all_params])
    t2_values = np.array([p['t2_seconds'] / 60 for p in all_params])  # Convert to minutes
    CD0_values = np.array([p['CD0_sample'] for p in all_params])
    data_labels = np.array(data_labels)

    # Separate training and testing indices
    training_indices = np.where(data_labels == "training")[0]
    testing_indices = np.where(data_labels == "testing")[0]

# Create visualization with separate colors for training and testing data
    plt.figure(figsize=(18, 10))

    # Stage 1 concentrations
    plt.subplot(2, 3, 1)
    for i, (t1v, sol1) in enumerate(all_stage1):
        CA1, CB1, CC1 = sol1.T
        alpha = 0.08 if data_labels[i] == "training" else 0.3
        color_suffix = '' if data_labels[i] == "training" else 'dark'
        plt.plot(t1v, CA1, color='blue' if data_labels[i] == "training" else 'navy', alpha=alpha)
        plt.plot(t1v, CB1, color='green' if data_labels[i] == "training" else 'darkgreen', alpha=alpha)
        plt.plot(t1v, CC1, color='red' if data_labels[i] == "training" else 'darkred', alpha=alpha)
    plt.title("Stage 1 (SDE)")
    plt.xlabel("Time [min]")
    plt.ylabel("Concentration [mol/m³]")
    plt.grid(True, alpha=0.3)
    plt.plot([], [], 'b-', alpha=0.5, label='A (training)')
    plt.plot([], [], 'g-', alpha=0.5, label='B (training)')
    plt.plot([], [], 'r-', alpha=0.5, label='C (training)')
    plt.plot([], [], 'navy', alpha=0.8, label='A (testing)')
    plt.plot([], [], 'darkgreen', alpha=0.8, label='B (testing)')
    plt.plot([], [], 'darkred', alpha=0.8, label='C (testing)')
    plt.legend()

    # Stage 2 concentrations
    plt.subplot(2, 3, 2)
    for i, (t2v, sol2) in enumerate(all_stage2):
        CA2, CB2, CC2, CD2, CE2, CF2 = sol2.T
        alpha = 0.08 if data_labels[i] == "training" else 0.3
        plt.plot(t2v, CB2, color='green' if data_labels[i] == "training" else 'darkgreen', alpha=alpha)
        plt.plot(t2v, CC2, color='red' if data_labels[i] == "training" else 'darkred', alpha=alpha)
        plt.plot(t2v, CD2, color='magenta' if data_labels[i] == "training" else 'purple', alpha=alpha)
        plt.plot(t2v, CE2, color='blue' if data_labels[i] == "training" else 'navy', alpha=alpha)
        plt.plot(t2v, CF2, color='orange' if data_labels[i] == "training" else 'darkorange', alpha=alpha)
    plt.title("Stage 2 (SDE)")
    plt.xlabel("Time [min]")
    plt.ylabel("Concentration [mol/m³]")
    plt.grid(True, alpha=0.3)
    plt.plot([], [], 'g-', alpha=0.5, label='B (training)')
    plt.plot([], [], 'r-', alpha=0.5, label='C (training)')
    plt.plot([], [], 'm-', alpha=0.5, label='D (training)')
    plt.plot([], [], 'b-', alpha=0.5, label='E (training)')
    plt.plot([], [], 'orange', alpha=0.5, label='F (training)')
    plt.legend()

    # Purity E
    plt.subplot(2, 3, 3)
    for i, (t2v, sol2) in enumerate(all_stage2):
        CA2, CB2, CC2, CD2, CE2, CF2 = sol2.T
        total_conc_profile = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
        purity_profile = CE2 / (total_conc_profile + 1e-10) * 100
        alpha = 0.08 if data_labels[i] == "training" else 0.3
        color = 'blue' if data_labels[i] == "training" else 'navy'
        plt.plot(t2v, purity_profile, color=color, alpha=alpha)
    plt.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
    plt.title("Purity of E")
    plt.xlabel("Time [min]")
    plt.ylabel("Purity [%]")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.legend()

    # T2 distribution comparison
    plt.subplot(2, 3, 4)
    plt.hist(T2_values[training_indices], bins=20, alpha=0.7, label='Training Data', color='blue', density=True)
    plt.hist(T2_values[testing_indices], bins=20, alpha=0.7, label='Testing Data', color='red', density=True)
    plt.xlabel('T2 Temperature [K]')
    plt.ylabel('Density')
    plt.title('T2 Temperature Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final purity comparison
    plt.subplot(2, 3, 5)
    final_purities_train = []
    final_purities_test = []

    for i, (t2v, sol2) in enumerate(all_stage2):
        CA2, CB2, CC2, CD2, CE2, CF2 = sol2.T
        total_conc_final = CA2[-1] + CB2[-1] + CC2[-1] + CD2[-1] + CE2[-1] + CF2[-1]
        purity_final = CE2[-1] / (total_conc_final + 1e-10) * 100

        if data_labels[i] == "training":
            final_purities_train.append(purity_final)
        else:
            final_purities_test.append(purity_final)

    plt.hist(final_purities_train, bins=15, alpha=0.7, label='Training Data', color='blue', density=True)
    plt.hist(final_purities_test, bins=15, alpha=0.7, label='Testing Data', color='red', density=True)
    plt.axvline(x=80, color='black', linestyle='--', label='Target: 80%')
    plt.xlabel('Final Purity [%]')
    plt.ylabel('Density')
    plt.title('Final Purity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # T2 vs Final Purity scatter plot
    plt.subplot(2, 3, 6)
    final_purities_all = final_purities_train + final_purities_test
    plt.scatter(T2_values[training_indices], final_purities_train, alpha=0.6, label='Training Data', color='blue', s=20)
    plt.scatter(T2_values[testing_indices], final_purities_test, alpha=0.8, label='Testing Data', color='red', s=30)
    plt.axhline(y=80, color='black', linestyle='--', label='Target: 80%')
    plt.xlabel('T2 Temperature [K]')
    plt.ylabel('Final Purity [%]')
    plt.title('T2 vs Final Purity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def discrete_trajectories(all_stage1, all_stage2, noise_percentage=0.0035, sample_interval=0.25):
    # Introduce arrays to store final concentrations and discretized solutions
    final_c1 = []
    final_c2 = []
    discrete_stage1 = []
    discrete_stage2 = []
    discrete_labels = []

    # Stage 1
    # For each solution in all_stage1
    for t1v, sol1 in all_stage1:
        # Store final concentrations of each simulation
        final_c1.append(sol1[-1])
    # Find mean concentrations across all simulations
    mean1 = np.mean(final_c1, axis=0)
    # Adjust the noise term with given percentage
    noise1 = mean1 * noise_percentage
    # For each simulation
    for i, (t1v, sol1) in enumerate(all_stage1):
        # Discretize the time interval from 0 to final batch time, with sample_interval
        t1k = np.arange(0, t1v[-1], sample_interval)
        # Introduce a matrix to store the discretized solutions
        sol1_discrete = np.zeros((len(t1k), 3))
        # For each component of stage 1
        for j in range(3):
            # Interpolate the existing solution to the discretized time
            sol1_interp = np.interp(t1k, t1v, sol1[:, j])
            # Introduce random noise value from normal distribution
            noise = np.random.normal(0, noise1[j], len(t1k))
            # Update the discretized solution adding the noise term
            sol1_discrete[:, j] = np.maximum(0, sol1_interp + noise)
        # Repeat for each simulation and store in discrete_stage1 matrix
        discrete_stage1.append((t1k, sol1_discrete))

    # Stage 2
    # For each simulation solutions inside the all_stage2
    for t2v, sol2 in all_stage2:
        # Store final concentrations of each simulation
        final_c2.append(sol2[-1])
    # Find mean concentrations across all simulations
    mean2 = np.mean(final_c2, axis=0)
    # Adjust the noise term with given percentage
    noise2 = mean2 * noise_percentage
    # For each simulation
    for i, (t2v, sol2) in enumerate(all_stage2):
        # Discretize the time interval from 0 to final batch time, with sample_interval
        t2k = np.arange(0, t2v[-1], sample_interval)
        # Introduce a matrix to store the discretized solutions
        sol2_discrete = np.zeros((len(t2k), 6))
        # For each component of stage 2
        for j in range(6):
            # Interpolate the existing solution to the discretized time
            sol2_interp = np.interp(t2k, t2v, sol2[:, j])
            # Introduce random noise value from normal distribution
            noise = np.random.normal(0, noise2[j], len(t2k))
            # Update the discretized solution adding the noise term
            sol2_discrete[:, j] = np.maximum(0, sol2_interp + noise)
        # Repeat for each simulation and store in discrete_stage1 matrix
        discrete_stage2.append((t2k, sol2_discrete))
        discrete_labels.append(data_labels[i])
    return discrete_stage1, discrete_stage2, discrete_labels


# Call the function
discrete_stage1, discrete_stage2, discrete_labels = discrete_trajectories(all_stage1, all_stage2,
                                                                          noise_percentage=0.0035, sample_interval=0.25)
discrete_labels = np.array(discrete_labels)
# Separate training and testing indices
training_indices_discrete = np.where(discrete_labels == "training")[0]
testing_indices_discrete = np.where(discrete_labels == "testing")[0]

# Now plot discrete samples with proper training/testing distinction
plt.figure(figsize=(18, 10))

# Stage 1 discrete samples
plt.subplot(2, 3, 1)
for i, (t1k, sol1_discrete) in enumerate(discrete_stage1):
    CA1, CB1, CC1 = sol1_discrete.T
    if discrete_labels[i] == "training":
        plt.plot(t1k, CA1, 'bo', alpha=0.08, markersize=1)
        plt.plot(t1k, CB1, 'go', alpha=0.08, markersize=1)
        plt.plot(t1k, CC1, 'ro', alpha=0.08, markersize=1)
    else:  # testing
        plt.plot(t1k, CA1, 'o', color='navy', alpha=0.3, markersize=1.5)
        plt.plot(t1k, CB1, 'o', color='darkgreen', alpha=0.3, markersize=1.5)
        plt.plot(t1k, CC1, 'o', color='darkred', alpha=0.3, markersize=1.5)

plt.title("Stage 1 (Discrete Samples)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.grid(True, alpha=0.3)
plt.plot([], [], 'bo', label='A (training)', markersize=4)
plt.plot([], [], 'go', label='B (training)', markersize=4)
plt.plot([], [], 'ro', label='C (training)', markersize=4)
plt.plot([], [], 'o', color='navy', label='A (testing)', markersize=4)
plt.plot([], [], 'o', color='darkgreen', label='B (testing)', markersize=4)
plt.plot([], [], 'o', color='darkred', label='C (testing)', markersize=4)
plt.legend()

# Stage 2 discrete samples
plt.subplot(2, 3, 2)
for i, (t2k, sol2_discrete) in enumerate(discrete_stage2):
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_discrete.T
    if discrete_labels[i] == "training":
        plt.plot(t2k, CB2, 'go', alpha=0.08, markersize=1)
        plt.plot(t2k, CC2, 'ro', alpha=0.08, markersize=1)
        plt.plot(t2k, CD2, 'mo', alpha=0.08, markersize=1)
        plt.plot(t2k, CE2, 'bo', alpha=0.08, markersize=1)
        plt.plot(t2k, CF2, 'o', color='orange', alpha=0.08, markersize=1)
    else:  # testing
        plt.plot(t2k, CB2, 'o', color='darkgreen', alpha=0.3, markersize=1.5)
        plt.plot(t2k, CC2, 'o', color='darkred', alpha=0.3, markersize=1.5)
        plt.plot(t2k, CD2, 'o', color='purple', alpha=0.3, markersize=1.5)
        plt.plot(t2k, CE2, 'o', color='navy', alpha=0.3, markersize=1.5)
        plt.plot(t2k, CF2, 'o', color='darkorange', alpha=0.3, markersize=1.5)

plt.title("Stage 2 (Discrete Samples)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.grid(True, alpha=0.3)
plt.plot([], [], 'go', label='B (training)', markersize=4)
plt.plot([], [], 'ro', label='C (training)', markersize=4)
plt.plot([], [], 'mo', label='D (training)', markersize=4)
plt.plot([], [], 'bo', label='E (training)', markersize=4)
plt.plot([], [], 'o', color='orange', label='F (training)', markersize=4)
plt.legend()

# Purity E discrete
plt.subplot(2, 3, 3)
for i, (t2k, sol2_discrete) in enumerate(discrete_stage2):
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_discrete.T
    total_conc = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
    purity = CE2 / (total_conc + 1e-10) * 100
    if discrete_labels[i] == "training":
        plt.plot(t2k, purity, 'bo', alpha=0.08, markersize=1)
    else:  # testing
        plt.plot(t2k, purity, 'o', color='navy', alpha=0.3, markersize=1.5)

plt.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
plt.title("Purity of E (Discrete)")
plt.xlabel("Time [min]")
plt.ylabel("Purity [%]")
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.plot([], [], 'bo', label='Training', markersize=4)
plt.plot([], [], 'o', color='navy', label='Testing', markersize=4)
plt.legend()

# Add comparison plots for discrete data
# Discrete sampling frequency comparison
plt.subplot(2, 3, 4)
sample_counts_train = [len(t1k) for i, (t1k, _) in enumerate(discrete_stage1) if discrete_labels[i] == "training"]
sample_counts_test = [len(t1k) for i, (t1k, _) in enumerate(discrete_stage1) if discrete_labels[i] == "testing"]
plt.hist(sample_counts_train, bins=15, alpha=0.7, label='Training Data', color='blue', density=True)
plt.hist(sample_counts_test, bins=15, alpha=0.7, label='Testing Data', color='red', density=True)
plt.xlabel('Number of Discrete Samples per Trajectory')
plt.ylabel('Density')
plt.title('Stage 1: Discrete Sampling Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Final E purity comparison
plt.subplot(2, 3, 5)
final_CE_train = []
final_CE_test = []

for i, (tk2, sol2_discrete) in enumerate(discrete_stage2):
    CAf, CBf, CCf, CDf, CEf, CFf = sol2_discrete[-1]
    purity = CEf / (CAf + CBf + CCf + CDf + CEf + CFf) * 100
    if discrete_labels[i] == "training":
        final_CE_train.append(purity)
    else:
        final_CE_test.append(purity)

plt.hist(final_CE_train, bins=15, alpha=0.7, label='Training Data', color='blue', density=True)
plt.hist(final_CE_test, bins=15, alpha=0.7, label='Testing Data', color='red', density=True)
plt.axvline(x=80, color='k', linestyle='--', label='Target: 80%')
plt.xlabel('Final E Purity [%]')
plt.ylabel('Density')
plt.title('Final E Purity (Discrete)')
plt.legend()
plt.grid(True, alpha=0.3)

# Stage 2 duration comparison
plt.subplot(2, 3, 6)
stage2_lengths_train = [tk2[-1] for i, (tk2, _) in enumerate(discrete_stage2) if discrete_labels[i] == "training"]
stage2_lengths_test = [tk2[-1] for i, (tk2, _) in enumerate(discrete_stage2) if discrete_labels[i] == "testing"]

plt.hist(stage2_lengths_train, bins=15, alpha=0.7, label='Training Data', color='blue', density=True)
plt.hist(stage2_lengths_test, bins=15, alpha=0.7, label='Testing Data', color='red', density=True)
plt.xlabel('Stage 2 Duration [min]')
plt.ylabel('Density')
plt.title('Stage 2 Duration Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def create_X1_matrix(discrete_stage1, CA0_values, t1_values, T1_values, discrete_labels):
    X1_train = []
    X1_test = []
    for i, (t1k, sol1_discrete) in enumerate(discrete_stage1):
        CA0 = CA0_values[i]
        t1 = t1_values[i]
        T1 = T1_values[i]
        CAf, CBf, CCf = sol1_discrete[-1]
        row = [CA0, t1, T1, CAf, CBf, CCf]
        if discrete_labels[i] == "training":
            X1_train.append(row)
        else:  # testing
            X1_test.append(row)
    X1_train = np.array(X1_train)
    X1_test = np.array(X1_test)
    return X1_train, X1_test


X1_train, X1_test = create_X1_matrix(discrete_stage1, CA0_values, t1_values, T1_values, discrete_labels)
df1 = pd.DataFrame(X1_train, columns=["CA0", "t1", "T1", "CA1_final", "CB1_final", "CC1_final"])
df2 = pd.DataFrame(X1_test, columns=["CA0", "t1", "T1", "CA1_final", "CB1_final", "CC1_final"])
#with pd.ExcelWriter("X1_mat.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="X1_train")
#    df2.to_excel(writer, sheet_name="X1_test")


def create_Y_matrix(discrete_stage2, discrete_labels):
    Y_train = []
    Y_test = []
    for i, (t2k, sol2_discrete) in enumerate(discrete_stage2):
        CAf, CBf, CCf, CDf, CEf, CFf = sol2_discrete[-1]
        purity = CEf / (CAf + CBf + CCf + CDf + CEf + CFf) * 100
        if discrete_labels[i] == "training":
            Y_train.append(purity)
        else:  # testing
            Y_test.append(purity)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return Y_train, Y_test


Y_train, Y_test = create_Y_matrix(discrete_stage2, discrete_labels)
df1 = pd.DataFrame(Y_train, columns=["E_purity"])
df2 = pd.DataFrame(Y_test, columns=["E_purity"])
#with pd.ExcelWriter("Y_mat.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="Y_train")
#    df2.to_excel(writer, sheet_name="Y_test")


def create_X2_tensor_matrix(discrete_stage2, CD0_values, t2_values, T2_values, discrete_labels):
    n_samples_train = len(training_indices_discrete)
    n_samples_test = len(testing_indices_discrete)
    max_points = max(len(sol2_discrete) for (_, sol2_discrete) in discrete_stage2)
    min_points = min(len(sol2_discrete) for (_, sol2_discrete) in discrete_stage2)
    X2_tensor_train = np.full((n_samples_train, max_points, 6), np.nan)
    X2_tensor_test = np.full((n_samples_test, max_points, 6), np.nan)
    for i, (t2k, sol2_discrete) in enumerate(discrete_stage2):
        actual_length = len(sol2_discrete)
        if discrete_labels[i] == "training":
            X2_tensor_train[i, :actual_length, :] = sol2_discrete
        else:  # testing
            X2_tensor_test[i - n_samples_train, :actual_length, :] = sol2_discrete
    X2_matrix_train = X2_tensor_train[:, 0:min_points + 1, :]
    X2_matrix_test = X2_tensor_test[:, 0:min_points + 1, :]
    return X2_tensor_train, X2_tensor_test, X2_matrix_train, X2_matrix_test


# Convert 3D to 2D by reshaping
def flatten_X2_tensor_matrix(X2_tensor_train, X2_tensor_test, X2_matrix_train, X2_matrix_test, CD0_values, t2_values,
                             T2_values, training_indices_discrete, testing_indices_discrete):
    X2_flat_train = X2_tensor_train.transpose(0, 2, 1).reshape(X2_tensor_train.shape[0], -1)  # (100, 27*6=162)
    X2_flat_test = X2_tensor_test.transpose(0, 2, 1).reshape(X2_tensor_test.shape[0], -1)  # (100, 27*6=162)
    X2_flat_train_matrix = X2_matrix_train.transpose(0, 2, 1).reshape(X2_matrix_train.shape[0], -1)  # (100, 27*6=162)
    X2_flat_test_matrix = X2_matrix_test.transpose(0, 2, 1).reshape(X2_matrix_test.shape[0], -1)  # (100, 27*6=162)
    CD0_train = CD0_values[training_indices_discrete]
    t2_train = t2_values[training_indices_discrete]
    T2_train = T2_values[training_indices_discrete]
    CD0_test = CD0_values[testing_indices_discrete]
    t2_test = t2_values[testing_indices_discrete]
    T2_test = T2_values[testing_indices_discrete]
    scalars_train = np.column_stack([CD0_train, t2_train, T2_train])
    scalars_test = np.column_stack([CD0_test, t2_test, T2_test])
    X2_full_train = np.hstack([scalars_train, X2_flat_train])
    X2_full_test = np.hstack([scalars_test, X2_flat_test])
    X2_full_train_matrix = np.hstack([scalars_train, X2_flat_train_matrix])
    X2_full_test_matrix = np.hstack([scalars_test, X2_flat_test_matrix])
    return X2_full_train, X2_full_test, X2_full_train_matrix, X2_full_test_matrix


X2_tensor_train, X2_tensor_test, X2_matrix_train, X2_matrix_test = create_X2_tensor_matrix(discrete_stage2, CD0_values,
                                                                                           t2_values, T2_values,
                                                                                           discrete_labels)
X2_full_train, X2_full_test, X2_full_train_matrix, X2_full_test_matrix = flatten_X2_tensor_matrix(X2_tensor_train,
                                                                                                  X2_tensor_test,
                                                                                                  X2_matrix_train,
                                                                                                  X2_matrix_test,
                                                                                                  CD0_values, t2_values,
                                                                                                  T2_values,
                                                                                                  training_indices_discrete,
                                                                                                  testing_indices_discrete)
cols = ['CD0', 't2_final', 'T2']
cols2 = ['CD0', 't2_final', 'T2']
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((X2_tensor_train.shape[1])):
        cols.append(f"{c}_t{t + 1}")
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((X2_matrix_train.shape[1])):
        cols2.append(f"{c}_t{t + 1}")
df1 = pd.DataFrame(X2_full_train, columns=cols)
df2 = pd.DataFrame(X2_full_test, columns=cols)
#with pd.ExcelWriter("X2_tensor.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="X2_tensor_train")
#    df2.to_excel(writer, sheet_name="X2_tensor_test")
df1 = pd.DataFrame(X2_full_train_matrix, columns=cols2)
df2 = pd.DataFrame(X2_full_test_matrix, columns=cols2)
#with pd.ExcelWriter("X2_matrix.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="X2_matrix_train")
#    df2.to_excel(writer, sheet_name="X2_matrix_test")


def stage2_correct(y, t, T2, Ea3, Ea4, A3, A4):
    CA, CB, CC, CD, CE, CF = y

    k3 = A3 * np.exp(-Ea3 / (R * T2))
    k4 = A4 * np.exp(-Ea4 / (R * T2))

    dCB = -k3 * CB * CD**2
    dCC = -k4 * CC * CD**2
    dCD = -2 * k3 * CB * CD**2
    dCE = k3 * CB * CD**2
    dCF = k4 * CC * CD**2

    return [0, dCB, dCC, dCD, dCE, dCF]

def stage2_mis(y, t, T2, Ea3, Ea4, A3, A4):
    CA, CB, CC, CD, CE, CF = y

    k3 = A3 * np.exp(-Ea3 / (R * T2))
    k4 = A4 * np.exp(-Ea4 / (R * T2))

    dCB = -k3 * CB * CD
    dCD = -2 * k3 * CB * CD
    dCE = k3 * CB * CD

    return [0, dCB, 0, dCD, dCE, 0]


def model_stage2(params_to_fit, data_dict_list):
    """
    Model function that simulates only stage 2 for all trajectories
    params_to_fit: [Ea3, Ea4, A3, A4] - activation energies and pre-exponential factors
    """
    Ea3_fit, Ea4_fit, A3_fit, A4_fit = params_to_fit

    all_predictions = []

    for data_dict in data_dict_list:
        # Extract parameters for this trajectory
        T2 = data_dict['T2']

        # Get initial conditions for stage 2 from the data
        initial_concentrations = data_dict['stage2_data']['concentrations'][0]  # First time point

        # Stage 2 simulation
        tk2 = data_dict['stage2_data']['time'] * 60  # Convert to seconds
        y0_2 = initial_concentrations  # [CA, CB, CC, CD, CE, CF]

        sol2_pred = odeint(stage2_correct, y0_2, tk2, args=(T2, Ea3_fit, Ea4_fit, A3_fit, A4_fit))

        # Flatten predictions for this trajectory
        trajectory_pred = sol2_pred.ravel()
        all_predictions.append(trajectory_pred)

    return np.concatenate(all_predictions)

def model_stage2_mis(params_to_fit, data_dict_list):
    """
    Model function that simulates only stage 2 for all trajectories
    params_to_fit: [Ea3, Ea4, A3, A4] - activation energies and pre-exponential factors
    """
    Ea3_fit, Ea4_fit, A3_fit, A4_fit = params_to_fit

    all_predictions = []

    for data_dict in data_dict_list:
        # Extract parameters for this trajectory
        T2 = data_dict['T2']

        # Get initial conditions for stage 2 from the data
        initial_concentrations = data_dict['stage2_data']['concentrations'][0]  # First time point

        # Stage 2 simulation
        tk2 = data_dict['stage2_data']['time'] * 60  # Convert to seconds
        y0_2 = initial_concentrations  # [CA, CB, CC, CD, CE, CF]

        sol2_pred = odeint(stage2_mis, y0_2, tk2, args=(T2, Ea3_fit, Ea4_fit, A3_fit, A4_fit))

        # Flatten predictions for this trajectory
        trajectory_pred = sol2_pred.ravel()
        all_predictions.append(trajectory_pred)

    return np.concatenate(all_predictions)


# Convert discrete data to fitting format (stage 2 only)
data_dict_list = []
for i, (tk2, sol2_discrete) in enumerate(discrete_stage2):
    params = all_params[i]

    data_dict = {
        'T2': params['T2'],
        'stage2_data': {
            'time': tk2,  # in minutes
            'concentrations': sol2_discrete  # [CA, CB, CC, CD, CE, CF]
        }
    }
    data_dict_list.append(data_dict)

# Prepare observed data (flatten all stage 2 discrete measurements)
all_y_observed = []
for data_dict in data_dict_list:
    stage2_flat = data_dict['stage2_data']['concentrations'].ravel()
    all_y_observed.append(stage2_flat)

all_y_observed = np.concatenate(all_y_observed)

# Initial parameter guesses [Ea3, Ea4, A3, A4]
Ea3_guess = 50000  # Adjust based on your system
Ea4_guess = 55000  # Adjust based on your system
A3_guess = 10  # Adjust based on your system
A4_guess = 20  # Adjust based on your system

param_guesses = [Ea3_guess, Ea4_guess, A3_guess, A4_guess]

print(f"Initial parameter guesses [Ea3, Ea4, A3, A4]: {param_guesses}")
print(f"Total data points to fit: {len(all_y_observed)}")

# Fit the model
try:
    fitted_params, pcov = curve_fit(
        lambda dummy, Ea3, Ea4, A3, A4: model_stage2([Ea3, Ea4, A3, A4], data_dict_list),
        np.zeros_like(all_y_observed),  # Dummy x values
        all_y_observed,
        p0=param_guesses,
        maxfev=10000,  # Increased for more parameters
        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, 100])  # Reasonable bounds
    )
    fitted_params_mis, pcov_mis = curve_fit(
        lambda dummy, Ea3, Ea4, A3, A4: model_stage2_mis([Ea3, Ea4, A3, A4], data_dict_list),
        np.zeros_like(all_y_observed),  # Dummy x values
        all_y_observed,
        p0=param_guesses,
        maxfev=10000,  # Increased for more parameters
        bounds=([0, 0, 0, 0], [np.inf, np.inf, 100, 100])  # Reasonable bounds
    )
    print("For correct model:")
    print(f"Fitted parameters [Ea3, Ea4, A3, A4]: {fitted_params}")
    print(f"True parameters [Ea3, Ea4, A3, A4]: {[Ea3_guess, Ea4_guess, A3_guess, A4_guess]}")

    # Calculate parameter errors
    param_errors = np.sqrt(np.diag(pcov))
    print(f"Parameter errors (std): {param_errors}")

    # Calculate relative errors
    rel_errors = param_errors / fitted_params * 100
    print(f"Relative errors (%): {rel_errors}")

    print("For misspecified model")
    print(f"Fitted parameters [Ea3, Ea4, A3, A4]: {fitted_params_mis}")
    print(f"True parameters [Ea3, Ea4, A3, A4]: {[Ea3_guess, Ea4_guess, A3_guess, A4_guess]}")

    # Calculate parameter errors
    param_errors = np.sqrt(np.diag(pcov_mis))
    print(f"Parameter errors (std): {param_errors}")

    # Calculate relative errors
    rel_errors = param_errors / fitted_params_mis * 100
    print(f"Relative errors (%): {rel_errors}")

except Exception as e:
    print(f"Fitting failed: {e}")
    fitted_params = param_guesses
    fitted_params_mis = param_guesses
# No fitting !*******************!
# fitted_params = param_guesses
# !******************************!

# Generate fitted trajectories using fitted parameters (stage 2 only)
all_stage2_fitted = []
all_stage2_fitted_mis = []
for i, data_dict in enumerate(data_dict_list):
    params = all_params[i]
    T2 = params['T2']
    t2_seconds = params['t2_seconds']

    # Use initial conditions from discrete data
    y0_2 = data_dict['stage2_data']['concentrations'][0]

    # Stage 2 with fitted parameters
    t2v = np.linspace(0, t2_seconds, 1000)
    sol2_fitted = odeint(stage2_correct, y0_2, t2v,
                         args=(T2, fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3]))
    sol2_fitted_mis = odeint(stage2_mis, y0_2, t2v,
                                args=(T2, fitted_params_mis[0], fitted_params_mis[1], fitted_params_mis[2], fitted_params_mis[3]))

    all_stage2_fitted.append((t2v / 60, sol2_fitted))
    all_stage2_fitted_mis.append((t2v / 60, sol2_fitted_mis))

print(f"Generated {len(all_stage2_fitted)} fitted stage 2 correct dynamics")
print(f"Generated {len(all_stage2_fitted_mis)} fitted stage 2 misspecified dynamics")

def fitted_trajectory_into_discrete(all_stage2_fitted, all_stage2_fitted_mis, discrete_stage2):
    sampled_fitted_trajectories_correct =[]
    sampled_fitted_trajectories_mis = []
    for i, ((t2v_fitted, sol2_fitted), (t2v_fitted_mis, sol2_fitted_mis), (t2k_discrete, sol2_discrete)) in enumerate(zip(all_stage2_fitted, all_stage2_fitted_mis, discrete_stage2)):
        sol2_fit_discrete_correct = np.zeros((len(t2k_discrete), 6))
        sol2_fit_discrete_mis = np.zeros((len(t2k_discrete), 6))
        # For each component of stage 1
        for j in range(6):
            # Interpolate the existing solution to the discretized time
            sol2_interp_correct = np.interp(t2k_discrete, t2v_fitted, sol2_fitted[:, j])
            sol2_interp_mis = np.interp(t2k_discrete, t2v_fitted_mis, sol2_fitted_mis[:, j])
            sol2_fit_discrete_correct[:, j] = sol2_interp_correct
            sol2_fit_discrete_mis[:, j] = sol2_interp_mis
        # Repeat for each simulation and store in discrete_stage1 matrix
        sampled_fitted_trajectories_correct.append((t2v_fitted, sol2_fit_discrete_correct))
        sampled_fitted_trajectories_mis.append((t2v_fitted, sol2_fit_discrete_mis))
    return sampled_fitted_trajectories_correct, sampled_fitted_trajectories_mis

fitted_correct_stage2, fitted_mis_stage2 = fitted_trajectory_into_discrete(all_stage2_fitted, all_stage2_fitted_mis, discrete_stage2)

M2_tensor_train, M2_tensor_test, M2_matrix_train, M2_matrix_test = create_X2_tensor_matrix(fitted_correct_stage2, CD0_values,
                                                                                           t2_values, T2_values,
                                                                                           discrete_labels)
M2_full_train, M2_full_test, M2_full_train_matrix, M2_full_test_matrix = flatten_X2_tensor_matrix(M2_tensor_train,
                                                                                                  M2_tensor_test,
                                                                                                  M2_matrix_train,
                                                                                                  M2_matrix_test,
                                                                                                  CD0_values, t2_values,
                                                                                                  T2_values,
                                                                                                  training_indices_discrete,
                                                                                                  testing_indices_discrete)

M2_mis_tensor_train, M2_mis_tensor_test, M2_mis_matrix_train, M2_mis_matrix_test = create_X2_tensor_matrix(fitted_mis_stage2, CD0_values,
                                                                                           t2_values, T2_values,
                                                                                           discrete_labels)
M2_mis_full_train, M2_mis_full_test, M2_mis_full_train_matrix, M2_mis_full_test_matrix = flatten_X2_tensor_matrix(M2_mis_tensor_train,
                                                                                                  M2_mis_tensor_test,
                                                                                                  M2_mis_matrix_train,
                                                                                                  M2_mis_matrix_test,
                                                                                                  CD0_values, t2_values,
                                                                                                  T2_values,
                                                                                                  training_indices_discrete,
                                                                                                  testing_indices_discrete)

cols = ['CD0', 't2_final', 'T2']
cols2 = ['CD0', 't2_final', 'T2']
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((M2_tensor_train.shape[1])):
        cols.append(f"{c}_t{t + 1}")
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((M2_matrix_train.shape[1])):
        cols2.append(f"{c}_t{t + 1}")
df1 = pd.DataFrame(M2_full_train, columns=cols)
df2 = pd.DataFrame(M2_full_test, columns=cols)
#with pd.ExcelWriter("M2_tensor.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="M2_tensor_train")
#    df2.to_excel(writer, sheet_name="M2_tensor_test")
df1 = pd.DataFrame(M2_full_train_matrix, columns=cols2)
df2 = pd.DataFrame(M2_full_test_matrix, columns=cols2)
#with pd.ExcelWriter("M2_matrix.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="M2_matrix_train")
#    df2.to_excel(writer, sheet_name="M2_matrix_test")

cols = ['CD0', 't2_final', 'T2']
cols2 = ['CD0', 't2_final', 'T2']
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((M2_mis_tensor_train.shape[1])):
        cols.append(f"{c}_t{t + 1}")
for c in ['CA', 'CB', 'CC', 'CD', 'CE', 'CF']:
    for t in range((M2_mis_matrix_train.shape[1])):
        cols2.append(f"{c}_t{t + 1}")
df1 = pd.DataFrame(M2_mis_full_train, columns=cols)
df2 = pd.DataFrame(M2_mis_full_test, columns=cols)
#with pd.ExcelWriter("M2_mis_tensor.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="M2_mis_tensor_train")
#    df2.to_excel(writer, sheet_name="M2_mis_tensor_test")
df1 = pd.DataFrame(M2_mis_full_train_matrix, columns=cols2)
df2 = pd.DataFrame(M2_mis_full_test_matrix, columns=cols2)
#with pd.ExcelWriter("M2_mis_matrix.xlsx") as writer:
#    df1.to_excel(writer, sheet_name="M2_mis_matrix_train")
#    df2.to_excel(writer, sheet_name="M2_mis_matrix_test")

# Plot comparison: Stage 2 Discrete Data vs Fitted Model
plt.figure(figsize=(18, 12))

# Component E
plt.subplot(2, 3, 1)
for tk2, sol2_discrete in discrete_stage2:
    CE2_discrete = sol2_discrete[:, 4]
    plt.plot(tk2, CE2_discrete, 'bo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted in all_stage2_fitted:
    CE2_fitted = sol2_fitted[:, 4]
    plt.plot(t2v, CE2_fitted, 'b-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component E (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component F
plt.subplot(2, 3, 2)
for tk2, sol2_discrete in discrete_stage2:
    CF2_discrete = sol2_discrete[:, 5]
    plt.plot(tk2, CF2_discrete, 'o', color='orange', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted in all_stage2_fitted:
    CF2_fitted = sol2_fitted[:, 5]
    plt.plot(t2v, CF2_fitted, '-', color='darkorange', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title("Stage 2: Component F (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component B (reactant)
plt.subplot(2, 3, 3)
for tk2, sol2_discrete in discrete_stage2:
    CB2_discrete = sol2_discrete[:, 1]
    plt.plot(tk2, CB2_discrete, 'go', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted in all_stage2_fitted:
    CB2_fitted = sol2_fitted[:, 1]
    plt.plot(t2v, CB2_fitted, 'g-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component B (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component D (reactant)
plt.subplot(2, 3, 4)
for tk2, sol2_discrete in discrete_stage2:
    CD2_discrete = sol2_discrete[:, 3]
    plt.plot(tk2, CD2_discrete, 'mo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted in all_stage2_fitted:
    CD2_fitted = sol2_fitted[:, 3]
    plt.plot(t2v, CD2_fitted, 'm-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component D (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)
# Purity comparison
plt.subplot(2, 3, 5)
for tk2, sol2_discrete in discrete_stage2:
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_discrete.T
    total_conc = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
    purity_discrete = CE2 / (total_conc + 1e-10) * 100
    plt.plot(tk2, purity_discrete, 'bo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted in all_stage2_fitted:
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_fitted.T
    total_conc = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
    purity_fitted = CE2 / (total_conc + 1e-10) * 100
    plt.plot(t2v, purity_fitted, 'b-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
plt.title("Purity of E (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Purity [%]")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)

# Parameter comparison plot
plt.subplot(2, 3, 6)
param_names = ['Ea3', 'Ea4', 'A3', 'A4']
true_params = [Ea3_mean, Ea4_mean, A3, A4]
plt.show()

# Print fitting summary
print("\n" + "="*50)
print("STAGE 2 FITTING SUMMARY (Ea and A factors)")
print("="*50)
param_names_full = ['Ea3 [J/mol]', 'Ea4 [J/mol]', 'A3 [1/s]', 'A4 [1/s]']
for i, name in enumerate(param_names_full):
    print(f"{name}:")
    print(f"  True:   {true_params[i]:.2e}")
    print(f"  Fitted: {fitted_params[i]:.2e}")
    print(f"  Error:  {abs(fitted_params[i] - true_params[i]):.2e}")
    if true_params[i] != 0:
        print(f"  Rel Error: {abs(fitted_params[i] - true_params[i])/abs(true_params[i])*100:.2f}%")
    print()

# Calculate rate constants at a reference temperature for comparison
T_ref = 350  # K, adjust as needed
print(f"\nRate constants at T = {T_ref} K:")
print("True:")
k3_true = A3 * np.exp(-Ea3_mean / (R * T_ref))
k4_true = A4 * np.exp(-Ea4_mean / (R * T_ref))
print(f"  k3 = {k3_true:.2e}")
print(f"  k4 = {k4_true:.2e}")

print("Fitted:")
k3_fitted = fitted_params[2] * np.exp(-fitted_params[0] / (R * T_ref))
k4_fitted = fitted_params[3] * np.exp(-fitted_params[1] / (R * T_ref))
print(f"  k3 = {k3_fitted:.2e}")
print(f"  k4 = {k4_fitted:.2e}")


# Plot comparison: Stage 2 Discrete Data vs Fitted Model
plt.figure(figsize=(18, 12))

# Component E
plt.subplot(2, 3, 1)
for tk2, sol2_discrete in discrete_stage2:
    CE2_discrete = sol2_discrete[:, 4]
    plt.plot(tk2, CE2_discrete, 'bo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted_mis in all_stage2_fitted_mis:
    CE2_fitted = sol2_fitted_mis[:, 4]
    plt.plot(t2v, CE2_fitted, 'b-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component E (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component F
plt.subplot(2, 3, 2)
for tk2, sol2_discrete in discrete_stage2:
    CF2_discrete = sol2_discrete[:, 5]
    plt.plot(tk2, CF2_discrete, 'o', color='orange', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted_mis in all_stage2_fitted_mis:
    CF2_fitted = sol2_fitted_mis[:, 5]
    plt.plot(t2v, CF2_fitted, '-', color='darkorange', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.title("Stage 2: Component F (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component B (reactant)
plt.subplot(2, 3, 3)
for tk2, sol2_discrete in discrete_stage2:
    CB2_discrete = sol2_discrete[:, 1]
    plt.plot(tk2, CB2_discrete, 'go', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted_mis in all_stage2_fitted_mis:
    CB2_fitted = sol2_fitted_mis[:, 1]
    plt.plot(t2v, CB2_fitted, 'g-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component B (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)

# Component D (reactant)
plt.subplot(2, 3, 4)
for tk2, sol2_discrete in discrete_stage2:
    CD2_discrete = sol2_discrete[:, 3]
    plt.plot(tk2, CD2_discrete, 'mo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted_mis in all_stage2_fitted_mis:
    CD2_fitted = sol2_fitted_mis[:, 3]
    plt.plot(t2v, CD2_fitted, 'm-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Stage 2: Component D (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Concentration [mol/m³]")
plt.legend()
plt.grid(True, alpha=0.3)
# Purity comparison
plt.subplot(2, 3, 5)
for tk2, sol2_discrete in discrete_stage2:
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_discrete.T
    total_conc = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
    purity_discrete = CE2 / (total_conc + 1e-10) * 100
    plt.plot(tk2, purity_discrete, 'bo', alpha=0.4, markersize=3, label='Discrete Data' if 'Discrete Data' not in plt.gca().get_legend_handles_labels()[1] else "")

for t2v, sol2_fitted_mis in all_stage2_fitted_mis:
    CA2, CB2, CC2, CD2, CE2, CF2 = sol2_fitted_mis.T
    total_conc = CA2 + CB2 + CC2 + CD2 + CE2 + CF2
    purity_fitted = CE2 / (total_conc + 1e-10) * 100
    plt.plot(t2v, purity_fitted, 'b-', alpha=0.3, linewidth=1.5, label='Fitted Model' if 'Fitted Model' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.axhline(y=80, color='r', linestyle='--', label='Target: 80%')
plt.title("Purity of E (Discrete vs Fitted)")
plt.xlabel("Time [min]")
plt.ylabel("Purity [%]")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, alpha=0.3)

# Parameter comparison plot
plt.subplot(2, 3, 6)
param_names = ['Ea3', 'Ea4', 'A3', 'A4']
true_params = [Ea3_mean, Ea4_mean, A3, A4]
plt.show()

# Print fitting summary
print("\n" + "="*50)
print("STAGE 2 FITTING SUMMARY (Ea and A factors)")
print("="*50)
param_names_full = ['Ea3 [J/mol]', 'Ea4 [J/mol]', 'A3 [1/s]', 'A4 [1/s]']
for i, name in enumerate(param_names_full):
    print(f"{name}:")
    print(f"  True:   {true_params[i]:.2e}")
    print(f"  Fitted: {fitted_params_mis[i]:.2e}")
    print(f"  Error:  {abs(fitted_params_mis[i] - true_params[i]):.2e}")
    if true_params[i] != 0:
        print(f"  Rel Error: {abs(fitted_params_mis[i] - true_params[i])/abs(true_params[i])*100:.2f}%")
    print()

# Calculate rate constants at a reference temperature for comparison
T_ref = 350  # K, adjust as needed
print(f"\nRate constants at T = {T_ref} K:")
print("True:")
k3_true = A3 * np.exp(-Ea3_mean / (R * T_ref))
k4_true = A4 * np.exp(-Ea4_mean / (R * T_ref))
print(f"  k3 = {k3_true:.2e}")
print(f"  k4 = {k4_true:.2e}")

print("Fitted:")
k3_fitted = fitted_params_mis[2] * np.exp(-fitted_params_mis[0] / (R * T_ref))
k4_fitted = fitted_params_mis[3] * np.exp(-fitted_params_mis[1] / (R * T_ref))
print(f"  k3 = {k3_fitted:.2e}")
print(f"  k4 = {k4_fitted:.2e}")

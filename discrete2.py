import numpy as np
import matplotlib.pyplot as plt
import sdeint

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

def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    # Stage 1 model equations
    dCAdt = -k1 * CA
    dCBdt = k1 * CA - k2 * CB
    dCCdt = k2 * CB
    return np.array([dCAdt, dCBdt, dCCdt])

def stage1_noise(y, t):
    # Diffusion coefficients obtained from calibration
    return np.diag([0.025, 0.9, 0.08])

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

def stage2_noise(y, t):
    # Diffusion coefficients obtained from calibration
    return np.diag([0.006, 0.03, 0.022, 0.07, 0.54, 0.04])

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
                         lambda y, t: stage1_noise(y,t), y0_1, t1v)
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
                         lambda y, t: stage2_noise(y,t), y0_2, t2v)
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
    return (t1v/60, sol1), (t2v/60, sol2), params

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
    for i, (t1v,sol1) in enumerate(all_stage1):
        # Discretize the time interval from 0 to final batch time, with sample_interval
        t1k = np.arange(0, t1v[-1], sample_interval)
        # Introduce a matrix to store the discretized solutions
        sol1_discrete = np.zeros((len(t1k),3))
        # For each component of stage 1
        for j in range (3):
            # Interpolate the existing solution to the discretized time
            sol1_interp = np.interp(t1k, t1v, sol1[:,j])
            # Introduce random noise value from normal distribution
            noise = np.random.normal(0, noise1[j], len(t1k))
            # Update the discretized solution adding the noise term
            sol1_discrete[:,j] = np.maximum(0, sol1_interp + noise)
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
    for i, (t2v,sol2) in enumerate(all_stage2):
        # Discretize the time interval from 0 to final batch time, with sample_interval
        t2k = np.arange(0, t2v[-1], sample_interval)
        # Introduce a matrix to store the discretized solutions
        sol2_discrete = np.zeros((len(t2k),6))
        # For each component of stage 2
        for j in range (6):
            # Interpolate the existing solution to the discretized time
            sol2_interp = np.interp(t2k, t2v, sol2[:,j])
            # Introduce random noise value from normal distribution
            noise = np.random.normal(0, noise2[j], len(t2k))
            # Update the discretized solution adding the noise term
            sol2_discrete[:,j] = np.maximum(0, sol2_interp + noise)
        # Repeat for each simulation and store in discrete_stage1 matrix
        discrete_stage2.append((t2k, sol2_discrete))
        discrete_labels.append(data_labels[i])
    return discrete_stage1, discrete_stage2, discrete_labels

# Call the function
discrete_stage1, discrete_stage2, discrete_labels = discrete_trajectories(all_stage1, all_stage2, noise_percentage=0.0035, sample_interval=0.25)
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
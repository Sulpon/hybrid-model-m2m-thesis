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
opt_sigma1 = [0.025, 0.9, 0.08]
opt_sigma2 = [0.006, 0.03, 0.022, 0.07, 0.54, 0.04]
def stage1_drift(y, t, k1, k2):
    CA, CB, CC = y
    # Stage 1 model equations
    dCAdt = -k1 * CA
    dCBdt = k1 * CA - k2 * CB
    dCCdt = k2 * CB
    return np.array([dCAdt, dCBdt, dCCdt])

def stage1_noise(y, t):
    return np.diag(opt_sigma1)  # scale by current concentration

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
    # Initial input to Stage 1
    y0_1 = np.array([CA0_sample, 0.0, 0.0])
    # Retrieving solutions of Stage 1 ODEs
    sol1 = sdeint.itoint(lambda y, t: stage1_drift(y, t, k1, k2),
                         lambda y, t: stage1_noise(y,t), y0_1, t1v)
    # Generate time trajectories for Stage 2
    t2v = np.linspace(0, t2_seconds, 1000)
    # Concentrations at Stage 1 completion
    CA1f, CB1f, CC1f = sol1[-1]
    # Dilution due to the addition of CD0
    dilution = V1 / V2
    CA2_0 = CA1f * dilution
    CB2_0 = CB1f * dilution
    CC2_0 = CC1f * dilution
    CD2_0 = CD0_sample
    CE2_0 = 0.0
    CF2_0 = 0.0
    # Initial input to Stage 2
    y0_2 = np.array([CA2_0, CB2_0, CC2_0, CD2_0, CE2_0, CF2_0])
    # Retrieving solutions of Stage 2 ODEs
    sol2 = sdeint.itoint(lambda y, t: stage2_drift(y, t, k3, k4),
                         lambda y, t: stage2_noise(y,t), y0_2, t2v)
    # Return solutions AND parameters
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
    return (t1v/60, sol1), (t2v/60, sol2), params

if __name__ == "__main__":
    # SET GLOBAL SEED HERE FOR REPRODUCIBILITY
    GLOBAL_SEED = 10  # Change this to any integer you want
    np.random.seed(GLOBAL_SEED)
    N_training = 100
    N_testing = 25
    N_total = N_training + N_testing

    all_stage1 = []
    all_stage2 = []
    all_params = []  # Store all parameters
    data_labels = []  # Store data type labels
    final_CA1 = []
    final_CC1 = []
    final_CB1 = []
    final_CA2 = []
    final_CC2 = []
    final_CB2 = []
    final_CD2 = []
    final_CE2 = []
    final_CF2 = []

    # Generate training data
    for i in range(N_training):
        (t1v, sol1), (t2v, sol2), params = one_sde_simulation(data_type="training")
        CA1f, CB1f, CC1f = sol1[-1]
        CA2f, CB2f, CC2f, CD2f, CE2f, CF2f = sol2[-1]
        final_CA1.append(CA1f)
        final_CB1.append(CB1f)
        final_CC1.append(CC1f)
        final_CA2.append(CA2f)
        final_CB2.append(CB2f)
        final_CC2.append(CC2f)
        final_CD2.append(CD2f)
        final_CE2.append(CE2f)
        final_CF2.append(CF2f)
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
    cv_CA1 = np.std(final_CA1) / np.mean(final_CA1)
    cv_CB1 = np.std(final_CB1) / np.mean(final_CB1)
    cv_CC1 = np.std(final_CC1) / np.mean(final_CC1)
    cv_CA2 = np.std(final_CA2) / np.mean(final_CA2)
    cv_CB2 = np.std(final_CB2) / np.mean(final_CB2)
    cv_CC2 = np.std(final_CC2) / np.mean(final_CC2)
    cv_CD2 = np.std(final_CD2) / np.mean(final_CD2)
    cv_CE2 = np.std(final_CE2) / np.mean(final_CE2)
    cv_CF2 = np.std(final_CF2) / np.mean(final_CF2)

    print("Final CVs for Stage 1 components:")
    print(f"CA: {cv_CA1:.4f}, CB: {cv_CB1:.4f}, CC: {cv_CC1:.4f}")
    print("Final CVs for Stage 2 components:")
    print(f"CA: {cv_CA2:.4f}, CB: {cv_CB2:.4f}, CC: {cv_CC2:.4f}, CD: {cv_CD2:.4f},CE: {cv_CE2:.4f},CF: {cv_CF2:.4f}")

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
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from itertools import combinations
import nnls_plots
import os

#==== senario selection ======
NofSou = 4
case = 3

#case 1 - Equal contributions
#case 2 - Slightly different contributions
#case 3 - Largely varied contributions


# ========================================================================================
# OUTPUT DIRECTORY
# ========================================================================================
OUTPUT_DIR = f"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/NNLS_out_2_3_4_sources/{NofSou}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================================================================================
# NNLS SOLVER CLASS
# ========================================================================================
class NNLSsolver:

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        # autodetect metals (first 5 columns)
        metals = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]
        self.A_full = df[metals].values.T   # (5 × Nsources)

        # number of sources automatically detected
        self.n_sources = df.shape[0]

        # Source ID labels
        self.source_ids = df["source_ID"].astype(str).values

        # True contributions from CSV
        self.x_true_full = df["contribution_x"].values


    # ------------------------------------------------------
    def condition_number(self, A):
        return np.linalg.cond(A)

    def metrics(self, x_est, x_true, A, b):
        residual_vec = A @ x_est - b
        rmse = np.sqrt(np.mean(residual_vec**2))

        if np.linalg.norm(x_true) == 0:
            rel_err = np.nan
        else:
            rel_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)
        return rmse, rel_err

    # ------------------------------------------------------
    def pick_combos(self, k):
        """Return ALL combinations for current number of sources"""
        return list(combinations(range(self.n_sources), k))


    # ========================================================================================
    # 1. IDEAL CONDITION
    # ========================================================================================
    def idealcondition(self, k_list=[1,2,3, 4, 8]):
        rows = []

        for k in k_list:
            combos = self.pick_combos(k)

            for combo in combos:

                A_sub = self.A_full[:, combo]
                x_true = self.x_true_full[list(combo)]

                b = A_sub @ x_true  #b= signature x contribution
                x_est, residual = nnls(A_sub, b)  #<---- NNLS solver

                condA = self.condition_number(A_sub)
                rmse, rel_err = self.metrics(x_est, x_true, A_sub, b)

                rows.append({
                    "k": k,
                    "combination": "-".join(self.source_ids[list(combo)]),   # FIXED
                    "cond_A": condA,
                    "rmse": rmse,
                    "relerr": rel_err,
                    "true_x": x_true,
                    "mean_x_est": x_est
                })

        return pd.DataFrame(rows)


    # ========================================================================================
    # 2. NOISE ON OBSERVATION
    # ========================================================================================
    def noisetoobs(self, k_list=[1,2,3,4,8], mc_runs=100):
        rows = []

        for k in k_list:
            combos = self.pick_combos(k)

            for combo in combos:

                A_sub = self.A_full[:, combo]
                x_true = self.x_true_full[list(combo)]
                b_clean = A_sub @ x_true

                rmse_list, err_list, x_list = [], [], []
                condA = self.condition_number(A_sub)

                for _ in range(mc_runs):

                    noise = np.random.uniform(-0.1, 0.1, size=b_clean.shape)
                    b_noisy = b_clean * (1 + noise)

                    x_est, _ = nnls(A_sub, b_noisy)
                    #rmse, rel_err = self.metrics(x_est, x_true, A_sub, b_clean)
                    rmse, rel_err = self.metrics(x_est, x_true, A_sub, b_noisy)


                    rmse_list.append(rmse)
                    err_list.append(rel_err)
                    x_list.append(x_est)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": "-".join(self.source_ids[list(combo)]),
                    "cond_A": condA,
                    "rmse": np.mean(rmse_list),
                    "relerr": np.mean(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "true_x": x_true,
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12)
                })

        return pd.DataFrame(rows)


    # ========================================================================================
    # 3. NOISE TO SIGNATURE MATRIX
    # ========================================================================================
    def noisetosignature(self, k_list=[1,2, 3, 4, 5], mc_runs=200):
        rows = []

        for k in k_list:
            combos = self.pick_combos(k)

            for combo in combos:
                A_sub = self.A_full[:, combo]
                x_true = self.x_true_full[list(combo)]
                b_clean = A_sub @ x_true

                rmse_list, err_list, x_list = [], [], []

                for _ in range(mc_runs):

                    noiseA = np.random.uniform(-0.1, 0.1, size=A_sub.shape)
                    A_noisy = A_sub * (1 + noiseA)

                    x_est, _ = nnls(A_noisy, b_clean)
                    rmse, rel_err = self.metrics(x_est, x_true, A_noisy, b_clean)

                    rmse_list.append(rmse)
                    err_list.append(rel_err)
                    x_list.append(x_est)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": "-".join(self.source_ids[list(combo)]),
                    "rmse": np.mean(rmse_list),
                    "relerr": np.mean(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "true_x": x_true,
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12)
                })

        return pd.DataFrame(rows)


    # ========================================================================================
    # 4. NOISE TO SIGNATURE + OBSERVATION
    # ========================================================================================
    def noisetosignatureandobs(self, k_list=[1,2, 3, 4, 5], mc_runs=100):
        rows = []

        for k in k_list:
            combos = self.pick_combos(k)

            for combo in combos:

                A_sub = self.A_full[:, combo]
                x_true = self.x_true_full[list(combo)]
                b_clean = A_sub @ x_true

                rmse_list, err_list, x_list = [], [], []

                for _ in range(mc_runs):

                    noiseA = np.random.uniform(-0.1, 0.1, size=A_sub.shape)
                    A_noisy = A_sub * (1 + noiseA)

                    noiseB = np.random.uniform(-0.1, 0.1, size=b_clean.shape)
                    b_noisy = b_clean * (1 + noiseB)

                    x_est, _ = nnls(A_noisy, b_noisy)
                    rmse, rel_err = self.metrics(x_est, x_true, A_noisy, b_clean)

                    rmse_list.append(rmse)
                    err_list.append(rel_err)
                    x_list.append(x_est)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": "-".join(self.source_ids[list(combo)]),
                    "rmse": np.mean(rmse_list),
                    "relerr": np.mean(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "true_x": x_true,
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12)
                })

        return pd.DataFrame(rows)



# ========================================================================================
# RUN
# ========================================================================================

#2-8 Sources Analysis
#Flexible
solver = NNLSsolver(f"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_{NofSou}sources_case{case}.csv")
#4Sources
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_4sources.csv")
#8Sources
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_4sources.csv")

df_ideal = solver.idealcondition()
df_ideal["scenario"] = "Ideal"

df_noise_obs = solver.noisetoobs()
df_noise_obs["scenario"] = "Noise Obs"

df_noise_sig = solver.noisetosignature()
df_noise_sig["scenario"] = "Noise Sig"

df_noise_both = solver.noisetosignatureandobs()
df_noise_both["scenario"] = "Noise Both"

# Merge
df_all = pd.concat([df_ideal, df_noise_obs, df_noise_sig, df_noise_both], ignore_index=True)
#df_all = pd.concat([df_ideal, df_noise_obs], ignore_index=True)

# Save
df_ideal.to_csv(os.path.join(OUTPUT_DIR, f"ideal_{NofSou}sources_case{case}.csv"), index=False)
df_noise_obs.to_csv(os.path.join(OUTPUT_DIR, f"noise_obs_{NofSou}sources_case{case}.csv"), index=False)
df_noise_sig.to_csv(os.path.join(OUTPUT_DIR, f"noise_sig_{NofSou}sources_case{case}.csv"), index=False)
df_noise_both.to_csv(os.path.join(OUTPUT_DIR, f"noise_both_{NofSou}sources_case{case}.csv"), index=False)

df_all.to_csv(os.path.join(OUTPUT_DIR, f"all_senarios_{NofSou}sources_case{case}.csv"), index=False)

print("Completed all analyses.")

# ========================================================================================
# Plotting
# ========================================================================================

surfix = f"K{NofSou}_case_{case}.png"

#==combined plots 
nnls_plots.plot_rmse_all_scenarios(df_all, rmse_col="rmse", title_suffix=f"- {NofSou} Sources -case {case}")

#==Box plots for rell_err and rmse
nnls_plots.plot_box_relerr(df_ideal, relerr_col="relerr", title_suffix=f"- Ideal con., {NofSou} Sources -case {case}")
nnls_plots.plot_box_rmse(df_ideal, rmse_col="rmse", title_suffix=f"- Ideal con., {NofSou} Sources -case {case}")

nnls_plots.plot_box_relerr(df_noise_obs, relerr_col="relerr", title_suffix=f"- Noise to Obs., {NofSou} Sources -case {case}")
nnls_plots.plot_box_rmse(df_noise_obs, rmse_col="rmse", title_suffix=f"- Noise to Obs., {NofSou} Sources -case {case}")

nnls_plots.plot_box_relerr(df_noise_sig, relerr_col="relerr", title_suffix=f"- Noise to Sig., {NofSou} Sources -case {case}")
nnls_plots.plot_box_rmse(df_noise_sig, rmse_col="rmse", title_suffix=f"- Noise to Sig., {NofSou} Sources -case {case}")

nnls_plots.plot_box_relerr(df_noise_both, relerr_col="relerr", title_suffix=f"- Noise to both, {NofSou} Sources -case {case}")
nnls_plots.plot_box_rmse(df_noise_both, rmse_col="rmse", title_suffix=f"- Noise to both, {NofSou} Sources -case {case}")







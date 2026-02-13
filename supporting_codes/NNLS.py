import numpy as np
import pandas as pd
from scipy.optimize import nnls
from itertools import combinations
import random
import nnls_plots
import os

# ============================================================
# DEFINE OUTPUT DIRECTORY
# ============================================================

#20positvesources
#OUTPUT_DIR = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/NNLS_out_20positvesources"

#1 positive source
#OUTPUT_DIR = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/NNLS_out_1outof20sources"

#2-8 sources analysis
OUTPUT_DIR = r"C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/NNLS_out/NNLS_out_2_to_8_sources"


# Make sure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


class NNLSsolver:

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Signature matrix A (5 × 20)
        metals = ["Metal1", "Metal2", "Metal3", "Metal4", "Metal5"]
        self.A_full = df[metals].values.T

        # Source IDs
        self.source_ids = df["source_ID"].values

        # TRUE contributions from CSV  (20 values)
        self.x_true_full = df["contribution_x"].values  # <<=== IMPORTANT

    # ---------------------------------------------
    def condition_number(self, A):
        return np.linalg.cond(A)

    def metrics(self, x_est, x_true, A, b):
        residual_vec = A @ x_est - b
        rmse = np.sqrt(np.mean(residual_vec**2))
        rel_err = np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)
        return rmse, rel_err
    
    def pick_combos(self, items, k):
        return list(combinations(range(items), k))

    
    #def pick_combos(self, items, k, max_combos=30):
     #   all_combos = list(combinations(range(items), k))
      #  return all_combos if len(all_combos) <= max_combos else random.sample(all_combos, max_combos)

    # ============================================================
    # 1. IDEAL CONDITION
    # ============================================================
    def idealcondition(self, k_list=[3]):     # k_list=[2,3,4,5,6,7,8]
        rows = []

        for k in k_list:
            combos = self.pick_combos(items=20, k=k)

            for combo in combos:

                A_sub = self.A_full[:, combo]

                # ============ USE GIVEN CSV CONTRIBUTIONS ============
                x_true = self.x_true_full[list(combo)]
                # ======================================================

                b = A_sub @ x_true
                x_est, residual = nnls(A_sub, b)

                condA = self.condition_number(A_sub)
                rmse, rel_err = self.metrics(x_est, x_true, A_sub, b)

                rows.append({
                    "k": k,
                    "combination": ",".join(self.source_ids[list(combo)]),
                    "cond_A": condA,
                    "residual": residual,
                    "rmse": rmse,
                    "mean_rmse": rmse,   # <--- just for the sake of completeness
                    "rel_err": rel_err,
                    "mean_relerr": rel_err,  # <--- just for the sake of completeness
                    "true_x": x_true,         # <---- added
                    "x_est": x_est,
                    "mean_x_est": x_est   # <--- just for the sake of completeness
                })

        return pd.DataFrame(rows)


    # ============================================================
    # 2. NOISE TO OBSERVATION
    # ============================================================
    def noisetoobs(self, k_list=[3], mc_runs=100):  # k_list=[2,3,4,5,6,7,8]
        rows = []

        for k in k_list:
            combos = self.pick_combos(20, k)

            for combo in combos:

                A_sub = self.A_full[:, combo]

                # TRUE contributions from CSV
                x_true = self.x_true_full[list(combo)]

                b_clean = A_sub @ x_true

                x_list = []
                rmse_list = []
                err_list = []
                condA = self.condition_number(A_sub)

                for _ in range(mc_runs):

                    noise = np.random.uniform(-0.1, 0.1, size=b_clean.shape)
                    b_noisy = b_clean * (1 + noise)

                    x_est, _ = nnls(A_sub, b_noisy)
                    rmse, rel_err = self.metrics(x_est, x_true, A_sub, b_clean)

                    x_list.append(x_est)
                    rmse_list.append(rmse)
                    err_list.append(rel_err)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": ",".join(self.source_ids[list(combo)]),
                    "cond_A": condA,
                    "mean_rmse": np.mean(rmse_list),
                    "std_rmse": np.std(rmse_list),
                    "mean_relerr": np.mean(err_list),
                    "std_relerr": np.std(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "std_x_est": x_array.std(axis=0),
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12),
                    "true_x": x_true          # <---- added
                })


        return pd.DataFrame(rows)

    # ============================================================
    # 3. NOISE TO SIGNATURE
    # ============================================================
    def noisetosignature(self, k_list=[2,3,4,5,6,7,8,9,10,15,20], mc_runs=1000):
        rows = []

        for k in k_list:
            combos = self.pick_combos(20, k)

            for combo in combos:

                A_sub = self.A_full[:, combo]

                # Use provided contributions
                x_true = self.x_true_full[list(combo)]

                b_clean = A_sub @ x_true

                x_list = []
                rmse_list = []
                err_list = []

                for _ in range(mc_runs):

                    noiseA = np.random.uniform(-0.1, 0.1, size=A_sub.shape)
                    A_noisy = A_sub * (1 + noiseA)

                    x_est, _ = nnls(A_noisy, b_clean)
                    rmse, rel_err = self.metrics(x_est, x_true, A_noisy, b_clean)

                    x_list.append(x_est)
                    rmse_list.append(rmse)
                    err_list.append(rel_err)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": ",".join(self.source_ids[list(combo)]),
                    "cond_A": self.condition_number(A_sub),
                    "mean_rmse": np.mean(rmse_list),
                    "std_rmse": np.std(rmse_list),
                    "mean_relerr": np.mean(err_list),
                    "std_relerr": np.std(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "std_x_est": x_array.std(axis=0),
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12),
                    "true_x": x_true          # <---- added
                })

        return pd.DataFrame(rows)

    # ============================================================
    # 4. NOISE TO SIGNATURE AND OBSERVATION
    # ============================================================
    def noisetosignatureandobs(self, k_list=[2,3,4,5,6,7,8,9,10,15,20], mc_runs=100):
        rows = []

        for k in k_list:
            combos = self.pick_combos(20, k)

            for combo in combos:

                A_sub = self.A_full[:, combo]

                # Use GIVEN contributions
                x_true = self.x_true_full[list(combo)]

                b_clean = A_sub @ x_true

                x_list = []
                rmse_list = []
                err_list = []

                for _ in range(mc_runs):

                    noiseA = np.random.uniform(-0.1, 0.1, size=A_sub.shape)
                    A_noisy = A_sub * (1 + noiseA)

                    noiseB = np.random.uniform(-0.1, 0.1, size=b_clean.shape)
                    b_noisy = b_clean * (1 + noiseB)

                    x_est, _ = nnls(A_noisy, b_noisy)
                    rmse, rel_err = self.metrics(x_est, x_true, A_noisy, b_clean)

                    x_list.append(x_est)
                    rmse_list.append(rmse)
                    err_list.append(rel_err)

                x_array = np.array(x_list)

                rows.append({
                    "k": k,
                    "combination": ",".join(self.source_ids[list(combo)]),
                    "mean_rmse": np.mean(rmse_list),
                    "std_rmse": np.std(rmse_list),
                    "mean_relerr": np.mean(err_list),
                    "std_relerr": np.std(err_list),
                    "mean_x_est": x_array.mean(axis=0),
                    "std_x_est": x_array.std(axis=0),
                    "cv_x_est": x_array.std(axis=0) / (x_array.mean(axis=0) + 1e-12),
                    "true_x": x_true          # <---- added
                })

        return pd.DataFrame(rows)



# ============================================================
# RUNNING
# ============================================================

#solver for 20 positive signatures
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios.csv")
#solver for 1 positive signature
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_1outof20.csv")

#2-8 Sources Analysis
#2sources
solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_2sources.csv")
#4Sources
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_4sources.csv")
#8Sources
#solver = NNLSsolver("C:/Users/kanch/Research_models/data_2/Bifurcation_senarios/BifurcationSenarios_4sources.csv")


df_ideal = solver.idealcondition()
df_ideal["scenario"] = "Ideal Condition"

df_noise_obs = solver.noisetoobs()
df_noise_obs["scenario"] = "Noise to Observation"

df_noise_sig = solver.noisetosignature()
df_noise_sig["scenario"] = "Noise to Signature"

df_noise_both = solver.noisetosignatureandobs()
df_noise_both["scenario"] = "Noise to Signature & Observation"

# ============================================================
# Save output CSVs
# ============================================================
df_ideal.to_csv(os.path.join(OUTPUT_DIR, "ideal_results.csv"), index=False)
df_noise_obs.to_csv(os.path.join(OUTPUT_DIR, "noise_obs_results.csv"), index=False)
#df_noise_sig.to_csv(os.path.join(OUTPUT_DIR, "noise_sig_results.csv"), index=False)
#df_noise_both.to_csv(os.path.join(OUTPUT_DIR, "noise_both_results.csv"), index=False)

# ============================================================
# MERGE ALL SCENARIOS INTO ONE
# ============================================================

df_all = pd.concat([df_ideal, df_noise_obs, df_noise_sig, df_noise_both], ignore_index=True)

df_all = df_all.rename(columns={
    "mean_rmse": "rmse",
    "mean_relerr": "relerr"
})


print("Completed all analyses.")

print("Completed all analyses.")

#nnls_plots.plot_condition_number(df_ideal)
#nnls_plots.plot_error_vs_condition(df_ideal)
#nnls_plots.plot_contribution_violin(df_noise_obs, k_value=2)
#nnls_plots.plot_mean_contribution_heatmap(df_noise_obs, k_value=10)
#nnls_plots.plot_rmse_comparison(df_noise_obs, k_value=3)
#nnls_plots.plot_truth_vs_estimation(df_noise_obs, k_value=3, source_index=0)
#nnls_plots.plot_sources_vs_rmse(df_noise_obs, rmse_col="mean_rmse")
#nnls_plots.plot_sources_vs_relerr(df_noise_obs, relerr_col="mean_relerr")
#nnls_plots.plot_box_relerr(df_noise_obs, relerr_col="mean_relerr", title_suffix="(Noise to Observation)")


#==combined plots 
nnls_plots.plot_rmse_all_scenarios(df_all, rmse_col="rmse")
#nnls_plots.plot_relerr_all_scenarios(df_all, relerr_col="relerr")
#nnls_plots.plot_box_relerr(df_all, relerr_col="relerr", title_suffix="(All Scenarios)")






# ==============================
# Correlation Analysis Script
# ==============================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ------------------------------
# 1. Load the data
# ------------------------------
file_path = r"C:\Users\kanch\Research_models\Chamila_model\Questionnaire.csv"

df = pd.read_csv(file_path)

print("\n=== Original Data Preview ===")
print(df.head())
print("\n=== Data Types ===")
print(df.dtypes)

# ------------------------------
# 2. Remove non-numeric columns
# ------------------------------
# (e.g., household name, ID, etc.)
df_numeric = df.select_dtypes(include=[np.number])

print("\n=== Numeric Variables Used for Analysis ===")
print(df_numeric.columns.tolist())

# ------------------------------
# 3. Handle missing values
# ------------------------------
print("\n=== Missing Values (Before) ===")
print(df_numeric.isna().sum())

# Mean imputation (safe for correlation analysis)
df_numeric = df_numeric.fillna(df_numeric.mean())

print("\n=== Missing Values (After) ===")
print(df_numeric.isna().sum())

# ------------------------------
# 4. Correlation matrices
# ------------------------------
# Pearson (linear relationships)
pearson_corr = df_numeric.corr(method="pearson")

# Spearman (recommended for ordinal & binary variables)
spearman_corr = df_numeric.corr(method="spearman")

print("\n=== Pearson Correlation Matrix ===")
print(pearson_corr)

print("\n=== Spearman Correlation Matrix ===")
print(spearman_corr)

# ------------------------------
# 5. Heatmap visualization
# ------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True
)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()

# ------------------------------
# 6. Pairwise correlation + p-values
# ------------------------------
print("\n=== Pairwise Spearman Correlations with p-values ===")

variables = df_numeric.columns

for i in range(len(variables)):
    for j in range(i + 1, len(variables)):
        r, p = spearmanr(df_numeric[variables[i]], df_numeric[variables[j]])
        print(f"{variables[i]} vs {variables[j]}: r = {r:.3f}, p = {p:.4f}")

# ------------------------------
# 7. Save correlation results
# ------------------------------
pearson_corr.to_csv("pearson_correlation_matrix.csv")
spearman_corr.to_csv("spearman_correlation_matrix.csv")

print("\nCorrelation matrices saved to CSV files.")

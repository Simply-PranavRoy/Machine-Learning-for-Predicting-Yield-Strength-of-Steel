# Steel Yield Strength ML Project
# --------------------------------
# This script loads a steel dataset, analyzes it, trains ML models (Random Forest & XGBoost),
# evaluates their performance, visualizes results, and exports all relevant data and predictions.
# All steps are commented for clarity and reproducibility.

# --- Imports ---
# Data loading, manipulation, ML, plotting, and file operations
from matminer.datasets import load_dataset
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# --- Setup ---
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory for saving outputs

# --- Load Dataset ---
df = load_dataset("matbench_steels")  # Loads steel dataset from matminer

# --- Initial Data Exploration ---
print(df.head())  # Show first few rows
print("\nNote: 'Yield Strength' values are in MPa.")
print("\n--- Dataset Info ---")
print(df.info())  # Data types and non-null counts
print("\n--- Statistical Summary ---")
print(df.describe())  # Summary stats for numeric columns
print("\nNote: 'Yield Strength' values are in MPa.")
print("\n--- Missing Values ---")
print(df.isnull().sum())  # Check for missing data
print("\n--- Correlation Matrix ---")
numeric_df = df.select_dtypes(include='number')
print(numeric_df.corr())  # Correlation between numeric columns

# --- Yield Strength Distribution Plot ---
if 'yield strength' in df.columns:
    plots_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.hist(df['yield strength'].dropna(), bins=30)
    plt.xlabel('Yield Strength (MPa)')
    plt.ylabel('Frequency')
    plt.title('Yield Strength Distribution')
    plot_path = os.path.join(plots_dir, "yield_strength_distribution.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
    print("Plot saved and closed, continuing to ML section...")

# --- Parse Composition Strings ---
def parse_composition(comp_str):
    """Extracts element symbols and their fractions from a composition string."""
    matches = re.findall(r'([A-Z][a-z]?)([0-9.]+)', comp_str)
    return {el: float(frac) for el, frac in matches}

# --- Expand Dataset with Elemental Fractions ---
element_df = df['composition'].apply(parse_composition).apply(pd.Series).fillna(0)
df_expanded = pd.concat([df, element_df], axis=1)
df_expanded = df_expanded.drop(columns=['composition'])
df_expanded = df_expanded.rename(columns={'yield strength': 'Yield Strength (MPa)'})

# --- Prepare Features and Target ---
X = df_expanded.drop(columns=['Yield Strength (MPa)'])  # Features: element fractions
y = df_expanded['Yield Strength (MPa)']  # Target: yield strength

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Train Random Forest Model ---
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- Evaluate Random Forest ---
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
print(f"Random Forest MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

# --- Train XGBoost Model ---
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# --- Evaluate XGBoost ---
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb) ** 0.5
print(f"XGBoost MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

# --- Random Forest Prediction Intervals ---
# Use all trees to estimate mean and std for uncertainty
all_tree_preds = np.stack([tree.predict(X_test.values) for tree in rf.estimators_])
pred_mean = np.mean(all_tree_preds, axis=0)
pred_std = np.std(all_tree_preds, axis=0)
lower = pred_mean - 2 * pred_std  # 95% lower bound
upper = pred_mean + 2 * pred_std  # 95% upper bound

# --- Print ML Model Results ---
print("\n--- ML Model Results ---")
print("ML Model Results (Random Forest Method)")
print("All prediction intervals (Random Forest):")
for i in range(len(y_test)):
    print(f"Predicted: {pred_mean[i]:.2f}, Interval: [{lower[i]:.2f}, {upper[i]:.2f}], Actual: {y_test.iloc[i]:.2f}")

print("\nML Model Results (XGBoost Method)")
print("All predictions (XGBoost):")
for i in range(len(y_test)):
    print(f"Predicted: {y_pred_xgb[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")

# --- Visualization: Actual vs Predicted Plots ---
# Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_mean, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel('Actual Yield Strength (MPa)')
plt.ylabel('Predicted Yield Strength (MPa)')
plt.title('Actual vs Predicted Yield Strength (Random Forest)')
plt.legend()
plt.tight_layout()
rf_text = f"MAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}"
plt.gca().text(0.05, 0.95, rf_text, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
avp_plot_path = os.path.join(base_dir, "Plots", "rf_actual_vs_predicted.png")
plt.savefig(avp_plot_path)
print(f"Actual vs Predicted plot (RF)saved to {avp_plot_path}")
plt.close()

# XGBoost
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel('Actual Yield Strength (MPa)')
plt.ylabel('Predicted Yield Strength (MPa)')
plt.title('Actual vs Predicted Yield Strength (XGBoost)')
plt.legend()
plt.tight_layout()
xgb_text = f"MAE: {mae_xgb:.2f}\nRMSE: {rmse_xgb:.2f}"
plt.gca().text(0.05, 0.95, xgb_text, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
xgb_avp_plot_path = os.path.join(base_dir, "Plots", "xgb_actual_vs_predicted.png")
plt.savefig(xgb_avp_plot_path)
print(f"Actual vs Predicted plot (XGBoost) saved to {xgb_avp_plot_path}")
plt.close()

# Combined RF & XGBoost
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_mean, alpha=0.7, label='Random Forest')
plt.scatter(y_test, y_pred_xgb, alpha=0.7, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel('Actual Yield Strength (MPa)')
plt.ylabel('Predicted Yield Strength (MPa)')
plt.title('Actual vs Predicted Yield Strength: RF vs XGBoost')
plt.legend()
plt.tight_layout()
combined_text = f"Random Forest\nMAE: {mae_rf:.2f}\nRMSE: {rmse_rf:.2f}\n\nXGBoost\nMAE: {mae_xgb:.2f}\nRMSE: {rmse_xgb:.2f}"
plt.gca().text(0.05, 0.95, combined_text, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
combined_plot_path = os.path.join(base_dir, "Plots", "combined_actual_vs_predicted.png")
plt.savefig(combined_plot_path)
print(f"Combined Actual vs Predicted plot saved to {combined_plot_path}")
plt.close()

# --- Feature Importance Plots ---
# Random Forest
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.ylabel("Importance")
plt.tight_layout()
fi_plot_path = os.path.join(base_dir, "Plots", "rf_feature_importance.png")
plt.savefig(fi_plot_path)
print(f"RF feature importance plot saved to {fi_plot_path}")
plt.close()

# XGBoost
xgb_importances = xgb.feature_importances_
xgb_feature_names = X.columns
xgb_indices = np.argsort(xgb_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("XGBoost Feature Importances")
plt.bar(range(len(xgb_importances)), xgb_importances[xgb_indices], align="center")
plt.xticks(range(len(xgb_importances)), xgb_feature_names[xgb_indices], rotation=45, ha='right')
plt.ylabel("Importance")
plt.tight_layout()
xgb_fi_plot_path = os.path.join(base_dir, "Plots", "xgb_feature_importance.png")
plt.savefig(xgb_fi_plot_path)
print(f"XGBoost feature importance plot saved to {xgb_fi_plot_path}")
plt.close()

# --- Export Results to Excel ---
# Three sheets: strength/elements, RF predictions, XGB predictions
rf_results_df = pd.DataFrame({
    'Actual': y_test.values,
    'RF_Predicted': pred_mean,
    'RF_Lower': lower,
    'RF_Upper': upper
})
xgb_results_df = pd.DataFrame({
    'Actual': y_test.values,
    'XGB_Predicted': y_pred_xgb
})
excel_path = os.path.join(base_dir, "matbench_steels_expanded.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    df_expanded.to_excel(writer, sheet_name='Strength_Elements', index=False)
    rf_results_df.to_excel(writer, sheet_name='RF_Predictions', index=False)
    xgb_results_df.to_excel(writer, sheet_name='XGB_Predictions', index=False)
print(f"All results exported to {excel_path}")

# --- Pause for User Review ---
input("\nScript finished. Press Enter to close the window...")
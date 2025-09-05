# Steel Yield Strength Predictor

**Author:** Pranav Roy
**Created:** September 2025

© 2025 Pranav Roy. All rights reserved.

---

## Overview
This project provides a machine learning pipeline for predicting the yield strength of steel alloys using elemental composition data. It includes:
- Data loading and exploration
- Feature engineering
- Model training (Random Forest & XGBoost)
- Model evaluation and visualization
- Export of results and predictions
- Virtual environment setup instructions

---

## Features
- Loads the `matbench_steels` dataset using matminer
- Parses steel compositions into elemental fractions
- Trains and evaluates two ML models: Random Forest and XGBoost
- Prints MAE and RMSE for both models
- Visualizes yield strength distribution, actual vs predicted plots, and feature importances
- Exports all results to a single Excel workbook with three sheets:
  - Strength_Elements: yield strength and elemental fractions
  - RF_Predictions: actual, predicted, and interval results for Random Forest
  - XGB_Predictions: actual and predicted results for XGBoost
- Pauses at the end so users can review results before closing

---

## Getting Started

### 1. Python Virtual Environment Setup
See `Python Virtual Environment Setup.txt` for step-by-step instructions:
- Create and activate a virtual environment
- Install all required packages using `requirements.txt`

### 2. Running the Predictor
- Open a terminal in your project directory
- Activate your virtual environment
- Run the script:
  ```
  python "Yield Strenth Predictor - [Steel].py"
  ```
- The script will load data, train models, print results, generate plots, and export an Excel file
- At the end, press Enter to close the window

---

## Scientific Background

**Yield Strength:**
The stress at which a material begins to deform plastically. For steels, it is a key property for engineering and design.

**Machine Learning Models:**
- **Random Forest:** Ensemble of decision trees, provides prediction intervals for uncertainty
- **XGBoost:** Gradient boosting, often achieves high accuracy on tabular data

**Metrics:**
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error):** Square root of average squared difference

---

## Output Files
- All plots are saved in the `Plots` directory
- All results are exported to `matbench_steels_expanded.xlsx` in the project folder

---

## License
This project is provided for educational and research purposes.

---

**Feel free to modify and expand the predictor for your needs!**

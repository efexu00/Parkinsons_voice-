# Parkinsons_voice-
# 🧠 Predicting Parkinson’s Disease Progression from Voice Features

This project implements a machine learning pipeline to predict Parkinson’s Disease severity (UPDRS scores) from voice-derived features using open-access data.  

The pipeline is implemented in **Google Colab** for accessibility and reproducibility.  

---

## 🎯 Research Objectives
- 📊 **Dataset utilisation**: Use an open-access Parkinson’s voice dataset (Telemonitoring dataset) to support reproducibility.  
- 🧹 **Transparent preprocessing**: Implement reproducible cleaning (missing values, scaling, feature selection).  
- 🤖 **Comparative modelling**: Train & compare Linear Regression, Random Forest, and XGBoost for UPDRS prediction.  
- 🔍 **Explainability**: Apply SHAP and permutation importance to identify the most predictive voice features.  

---

## 📂 Folder Structure
/parkinsons_voice_msc/
│
├── data/
│ ├── raw/ # original dataset (from Kaggle or UCI)
│ ├── cleaned/ # processed data, scaled/imputed
│
├── models/ # trained models (.joblib)
├── images/ # plots (heatmaps, SHAP, feature importance)
├── reports/ # results tables, ranked features
└── notebooks/ # Google Colab notebook(s)

---

## ⚙️ Installation & Setup
1. Open **Google Colab**.  
2. Mount Google Drive:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
3. Place the raw dataset (parkinsons_updrs.data.csv)
   /parkinsons_voice_msc/data/raw/
   !pip install xgboost shap plotly
   🛠️ Pipeline Steps (Notebook Sections)

Data Loading & Exploration

Load dataset from /data/raw/.

Inspect columns, datatypes, summary statistics.

Generate correlation heatmap.

Data Preprocessing

Handle missing values (median imputation).

Standardise features (z-score).

Optional feature selection (variance threshold, high correlation filter).

Save cleaned dataset.

Baseline Model

Train Multiple Linear Regression.

Evaluate using RMSE, MAE, R².

Record cross-validation scores.

Advanced Models

Train Random Forest and XGBoost regressors.

Compare against baseline with same metrics.

Hyperparameter Tuning

GridSearchCV for Random Forest & XGBoost.

Store tuned models.

Save final performance table.

Explainability Analysis

Apply SHAP to best model.

Plot beeswarm & bar charts of feature importance.

Save ranked features list.

📈 Outputs

Cleaned datasets → /data/cleaned/

Trained models → /models/

Evaluation results → /reports/results_summary.csv

Explainability plots → /images/

🔁 Reproducibility Notes

All code is contained in a single Colab notebook (parkinsons_voice_pipeline.ipynb).

Random seeds are fixed (random_state=42) for reproducibility.

Outputs (models, metrics, plots) are saved to Drive for version control.

Repository can be mirrored on GitHub with notebook + requirements.

💡 Optional Enhancements

Interactive Plotly dashboards for exploring feature importance.

Additional regression models (e.g., LightGBM, CatBoost) for comparison.

Integration with GitHub Actions for automated evaluation on new data.


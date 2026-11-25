# Project Portfolio: Forest Cover Type Classification

**Proof of Work & Technical Showcase for Recruiters**  
*An End-to-End Machine Learning Pipeline: From Exploratory Data Analysis to Optimized Gradient Boosting and Model Interpretability (SHAP).*

---

## 📌 Executive Summary
- **Objective**: Classify forest cover types (7 categories) based on cartographic variables including elevation, aspect, soil type, and distance to hydrology/roads/wildfire points.
- **Top-Performing Model**: **Hyperparameter-Tuned LightGBM**
- **Primary Metrics achieved on Test Set**:
  - **Accuracy**: **86.90%**
  - **Macro-Averaged F1-Score**: **86.70%**
- **Key Technologies**: Python, Pandas, Scikit-Learn, LightGBM, XGBoost, CatBoost, SHAP, MLflow, Matplotlib, Seaborn.

---

## 🛠️ Software Engineering & ML Workflow

### 1. Exploratory Data Analysis & Quality Control
- Analyzed a dataset of **15,120 samples** across **56 initial features**.
- Performed sanity checks for null values (0 found) and duplicated rows (0 found).
- Inspected binary distributions of **Soil_Type** and **Wilderness_Area** features.
- **Data Cleaning**: Dropped constant/zero-variance features (`Soil_Type7` and `Soil_Type15`) to reduce dimensionality and speed up training.

### 2. Feature Engineering & Scaling Pipeline
- Divided features into distinct logical groups: Numerical, Soil One-Hot, and Wilderness One-Hot columns.
- Constructed a robust validation split using **Stratified 5-Fold Cross-Validation** to guarantee equal representation of all 7 target classes.
- Designed a reusable Scikit-Learn `ColumnTransformer` to apply `MinMaxScaler` only to numerical components while preserving binary indicators.

### 3. Rigorous Model Exploration (The Benchmark)
Evaluated multiple algorithms using robust cross-validation alongside systematic hyperparameter optimization via `RandomizedSearchCV`:

| Model | CV Macro F1 | Test Accuracy | Test Macro F1 | Key Attributes & Configuration |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Dummy)** | 3.56% | 14.26% | 3.56% | Most frequent classifier benchmark |
| **Logistic Regression** | 67.63% | 67.39% | 67.11% | Regularized, Max Iterations = 1000 |
| **Support Vector Classifier (SVM)** | 69.85% | 70.89% | 70.63% | Radial Basis Function (RBF) Kernel |
| **Decision Tree** | 77.91% | 78.33% | 78.22% | Standard Gini criterion baseline |
| **K-Nearest Neighbors (KNN)** | 77.48% | 78.53% | 78.09% | Distance-based, $k=5$ |
| **CatBoost** | 81.63% | 81.84% | 81.47% | Native gradient boosting with standard parameters |
| **XGBoost (Tuned)** | 86.03% | 86.27% | 86.05% | Opt. Params: Max Depth = 9, Estimators = 250 |
| **LightGBM (Tuned)** | **86.32%** | **86.90%** | **86.70%** | **Best Model** (Num Leaves = 70, Depth = 15, Estimators = 200) |

### 4. Experiment Tracking & MLOps
- Fully integrated **MLflow** for robust tracking of hyperparameters, validation scores, out-of-sample metrics, and training run duration for reproducibility and auditing.

### 5. Model Interpretability (SHAP Explanations)
Rather than deploying a "black box" solution, built-in transparency was established using **TreeSHAP** over 1,000 randomized test samples:
- **Elevation** emerged as the primary split variable (highest SHAP and Gini importances).
- Secondary critical factors include physical distance features (`Horizontal_Distance_To_Roadways` and `Horizontal_Distance_To_Fire_Points`), showing complex, non-linear physical constraints on tree species ecosystems.
```

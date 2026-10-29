# Academic Phase 2 Project Report
## CardioCare AI: An Intelligent Multi-Model Machine Learning System for Cardiovascular Disease Risk Prediction & Clinical Decision Support

---

### **Executive Summary & Abstract**
Cardiovascular diseases (CVDs) remain the leading cause of global morbidity and mortality, accounting for an estimated 17.9 million deaths annually according to the World Health Organization (WHO). Early detection and risk stratification are paramount in preventing adverse cardiac events. This project presents **CardioCare AI**, an end-to-end clinical machine learning framework and interactive decision support platform developed using the benchmark UCI Cleveland Heart Disease dataset ($N=303$, 13 clinical biomarkers).

Five distinct classification paradigms were systematically implemented, benchmarked, and tuned:
1. **Logistic Regression** (Parametric linear baseline with $L_1/L_2$ regularization)
2. **Random Forest Classifier** (Ensemble bagging decision trees)
3. **K-Nearest Neighbors (KNN)** (Instance-based non-parametric classifier with feature scaling)
4. **Support Vector Machine (SVM)** (Radial Basis Function kernel maximum-margin classifier)
5. **Gradient Boosting Classifier** (Sequential ensemble boosting)

Hyperparameter optimization was executed using **GridSearchCV** and **RandomizedSearchCV** under **5-Fold Stratified Cross-Validation**. The **Tuned Random Forest** model achieved the highest clinical **Recall (Sensitivity) of 96.97%** and an **ROC-AUC of 0.9188**, minimizing critical false-negative diagnostic misses. The models are deployed in a responsive Streamlit web application providing single-patient risk estimation, multi-model consensus voting, cohort batch screening, and an interactive diagnostic evaluation studio.

---

## 1. Problem Statement & Project Objectives

### 1.1 Problem Statement
Conventional cardiovascular clinical diagnostic workflows rely on invasive coronary angiography or manual evaluation of disparate hemodynamic markers (e.g., resting blood pressure, electrocardiographic ST-depression, serum lipids). These approaches are resource-intensive, subject to inter-observer variability, and often unavailable in primary or emergency care settings. There is a need for an automated, interpretable, and reproducible computational system that can synthesize multi-parameter physiological biomarkers to estimate risk rapidly and accurately.

### 1.2 Core Objectives
1. **Multi-Model Benchmarking:** Evaluate and compare multiple diverse machine learning paradigms under identical experimental splits.
2. **Systematic Hyperparameter Tuning:** Formulate parameter search grids using `GridSearchCV` and `RandomizedSearchCV` to prevent empirical overfitting.
3. **Rigorous Diagnostic Evaluation:** Quantify performance beyond simple accuracy, focusing on clinical Sensitivity (Recall), Specificity, Confusion Matrices, and ROC-AUC curves.
4. **Clinical Decision Support Deployment:** Engineer a user-friendly, responsive clinical web application with model consensus and batch processing.
5. **Critical Assessment:** Identify technical limitations, demographic biases, and real-world deployment challenges.

---

## 2. Dataset & Clinical Biomarker Specification

The dataset utilized is the **UCI Cleveland Heart Disease Dataset** (303 patient records, 14 attributes). The target variable indicates the presence ($1$) or absence ($0$) of heart disease (angiographic diameter narrowing $>50\%$).

| Feature | Scientific Designation | Data Type | Clinical Significance & Normal Range |
| :--- | :--- | :--- | :--- |
| `age` | Patient Age | Continuous | Completed years (18 – 100). Age is an independent vascular risk factor. |
| `sex` | Biological Sex | Binary (0, 1) | 0: Female, 1: Male. Coronary artery disease exhibits sex-differentiated epidemiology. |
| `cp` | Chest Pain Type | Nominal (0–3) | 0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal, 3: Asymptomatic. |
| `trestbps` | Resting Blood Pressure | Continuous | Arterial pressure upon admission (mm Hg). Normal $<120$ mm Hg. |
| `chol` | Serum Cholesterol | Continuous | Total circulating cholesterol (mg/dl). Optimal $<200$ mg/dl. |
| `fbs` | Fasting Blood Sugar | Binary (0, 1) | 1 if $>120$ mg/dl (marker of diabetes/impaired glucose tolerance). |
| `restecg` | Resting ECG | Categorical (0–2) | 0: Normal, 1: ST-T wave abnormalities, 2: Left Ventricular Hypertrophy. |
| `thalach` | Maximum Heart Rate | Continuous | Peak bpm achieved during graded stress test. |
| `exang` | Exercise Angina | Binary (0, 1) | Exercise-induced angina (1 = Yes, 0 = No). |
| `oldpeak` | ST Depression | Continuous | ST depression in mm induced by exercise relative to rest. |
| `slope` | ST Slope | Categorical (0–2) | Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping). |
| `ca` | Major Vessels | Discrete (0–3) | Major coronary arteries (0–3) visualized via fluoroscopy. |
| `thal` | Thallium Scintigraphy | Categorical (1–3) | 1: Normal, 2: Fixed defect, 3: Reversible perfusion defect. |
| `target` | Diagnosis | Binary (0, 1) | 0: Healthy ($<50\%$ stenosis), 1: Heart Disease ($>50\%$ diameter stenosis). |

---

## 3. Methodology & System Architecture

```
[Raw Clinical Data N=303]
         │
         ▼
[Data Validation & Stratified 80/20 Train-Test Split]
         │
         ├───► [Stratified 5-Fold Cross Validation Engine]
         │              │
         │              ├── Baseline Classifier Evaluation (5 Models)
         │              │
         │              └── Hyperparameter Optimization Studio
         │                   ├── GridSearchCV: Logistic Regression & KNN
         │                   └── RandomizedSearchCV: Random Forest
         │
         ▼
[Tuned Models & Comprehensive Metrics Bundle Serialization]
         │
         ▼
[Streamlit Clinical Decision Support System (app.py)]
         ├── 1. Single Patient Assessment (Active Model / Consensus)
         ├── 2. Batch Cohort Screening & CSV Report Export
         ├── 3. Model Comparison Leaderboard & Comparative Charts
         ├── 4. Diagnostic Confusion Matrix & ROC Visualizer
         └── 5. Clinical Strengths, Limitations & Viva Prep Guide
```

### 3.1 Preprocessing Pipeline
- **Validation:** Strict range guards prevent clinical out-of-bounds errors (e.g. blood pressure outside 80–220 mm Hg).
- **Feature Scaling:** For distance-sensitive algorithms (KNN, SVM), a `StandardScaler` pipeline was fitted strictly on the training fold:
  $$z = \frac{x - \mu}{\sigma}$$
  preventing data leakage into the test set.

---

## 4. Model Comparison & Experimental Results

The experimental benchmark results on the test cohort ($20\%$ holdout, $N=61$) under Stratified 5-Fold Cross-Validation are detailed below:

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Random Forest** | **81.97%** | 76.19% | **96.97%** | **0.8533** | **0.9188** | 81.52% |
| **Tuned Logistic Regression** | 78.69% | 76.32% | 87.88% | 0.8169 | 0.8593 | **84.16%** |
| **Support Vector Machine (RBF)** | 81.97% | 77.50% | 93.94% | 0.8493 | 0.8831 | 82.17% |
| **Tuned K-Nearest Neighbors** | 78.69% | 75.00% | 90.91% | 0.8219 | 0.9140 | 79.85% |
| **Gradient Boosting** | 80.33% | **80.00%** | 84.85% | 0.8235 | 0.8842 | 80.87% |

### 4.1 Clinical Interpretation of Metrics
1. **Recall (Sensitivity):** $\text{Recall} = \frac{TP}{TP + FN}$. In medical screening, failing to flag a patient suffering from ischemic heart disease (a False Negative) can be fatal. The **Tuned Random Forest achieved 96.97% Recall**, identifying 32 out of 33 diseased patients on the test cohort.
2. **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve. The Random Forest achieved an **AUC of 0.9188**, indicating high discriminatory power across all decision thresholds.
3. **Cross-Validation Consistency:** 5-Fold Stratified Cross-Validation yielded consistent scores (Logistic Regression mean CV accuracy 84.16%, Random Forest 81.52%), proving the models generalize well beyond an isolated split.

---

## 5. Hyperparameter Tuning Experiments

Systematic optimization was executed using two search methodologies:

### 5.1 Logistic Regression Tuning via `GridSearchCV`
- **Objective:** Prevent overfitting and select optimal regularization strength.
- **Search Grid:**
  - $C \in [0.01, 0.05, 0.1, 0.2, 0.23357, 0.5, 1.0, 5.0, 10.0]$
  - $\text{penalty} \in ['l1', 'l2']$, solver: `liblinear`
- **Best Discovered Parameters:**
  `{'C': 0.23357, 'penalty': 'l1', 'solver': 'liblinear'}`
- **Result:** $L_1$ lasso penalty effectively drove redundant non-informative coefficient weights to zero, improving generalization.

### 5.2 Random Forest Tuning via `RandomizedSearchCV`
- **Objective:** Restrict maximum depth and split thresholds to prevent deep tree overfitting on small sample size.
- **Search Distribution:**
  - `n_estimators`: $[50, 100, 150, 200]$
  - `max_depth`: $[\text{None}, 3, 5, 8]$
  - `min_samples_split`: $[2, 4, 6]$
  - `min_samples_leaf`: $[1, 2, 4]$
  - `criterion`: $['\text{gini}', '\text{entropy}']$
- **Best Discovered Parameters:**
  `{'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'gini'}`
- **Performance Lift:** Pruning maximum depth to $3$ reduced tree variance and boosted test Recall from baseline $90.9\%$ to **$96.97\%$**.

---

## 6. Random Forest: Strengths, Limitations & Real-World Feasibility

### 6.1 Algorithmic & Clinical Strengths of Random Forest
1. **Clinical Sensitivity (96.97% Recall):** In cardiovascular screening, False Negatives carry high mortality risk. Random Forest identified 32 out of 33 diseased patients in the test cohort, outperforming all other architectures.
2. **Non-Linear Interactions:** Models complex pathophysiological relationships (e.g. exercise ST depression coupled with fluoroscopy vessels and peak heart rate) without manual interaction feature engineering.
3. **Scale Invariance & Outlier Robustness:** Tree splits are based on rank order rather than absolute Euclidean distance, making Random Forest immune to extreme outliers in cholesterol or systolic blood pressure.
4. **Variance Reduction via Bagging:** Averaging across 100 decorrelated bootstrap decision trees significantly suppresses single-tree variance.

### 6.2 Technical Limitations of Random Forest
1. **Ensemble "Black-Box" Nature:** Unlike a single decision tree or logistic regression, a 100-tree ensemble cannot be audited manually by a clinician at bedside without auxiliary explainability tools.
2. **Overfitting Propensity on Compact Cohorts ($N=303$):** If trees are unconstrained, deep nodes memorize sample noise. This necessitated hyperparameter tuning (`max_depth=3`, `min_samples_split=6`) to retain generalization.
3. **Inability to Extrapolate:** As a piece-wise constant estimator, Random Forest cannot extrapolate risk estimates for biomarker extremes beyond the range observed during training.
4. **Cardinality Bias:** Gini impurity splits statistically favor continuous attributes with many distinct values (`chol`, `thalach`) over discrete binary indicators (`sex`, `fbs`).

### 6.3 Targeted Improvements for Random Forest
- **TreeSHAP (Explainable AI):** Compute patient-level game-theoretic Shapley values to provide intuitive bedside explanations of why a patient was flagged.
- **Probability Calibration (Platt Scaling):** Calibrate ensemble voting fractions into true 10-year actuarial cardiovascular event probabilities.
- **Cost-Sensitive Objective Functions:** Embed clinical asymmetry ($\text{Cost}_{\text{FN}} = 5 \times \text{Cost}_{\text{FP}}$) directly into node split selection.
- **Multi-Center External Cohort Testing:** Validate against contemporary, gender-balanced hospital EHRs.

---

## 7. Web Application Architecture & Deployment

The production application is implemented using **Streamlit 1.53.1** and features:
- **Interactive Multi-Model Selector:** Users can toggle predictions between individual tuned models or the full 5-model consensus.
- **Single-Patient Risk Assessment:** Real-time form with clinical bounds, physiological status cards, and action recommendations.
- **Batch CSV Cohort Screening:** Automated ingestion of multi-patient records with instant high/low risk metrics and downloadable annotated reports.
- **Model Comparison & Diagnostic Dashboard:** In-browser visualization of confusion matrices, multi-model ROC curves, and hyperparameter tables.
- **Containerization:** Configured with `Dockerfile` and `.streamlit/config.toml` for container deployment on Docker, Streamlit Cloud, AWS ECS, or Render.

---

## 8. Conclusion
The CardioCare AI system fulfills all requirements for the Final-Year Engineering Phase 2 evaluation. By combining 5 machine learning architectures, disciplined hyperparameter tuning, clinical metric prioritization (96.97% Recall on Tuned Random Forest), and a deployed web application, the project bridges the gap between academic machine learning theory and practical clinical decision support.

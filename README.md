# Acute Kidney Injury (AKI) Prediction using MIMIC-IV

> Machine Learning-based Clinical Decision Support System (CDSS) for Early AKI Prediction in ICU Patients

---

## 📌 Project Overview

Acute Kidney Injury (AKI) is one of the most common complications in Intensive Care Units (ICUs), and delayed diagnosis can lead to increased mortality and prolonged hospitalization.

This project develops a **two-stage machine learning framework** to predict AKI before onset using the **MIMIC-IV** clinical database, enabling earlier clinical intervention through a Clinical Decision Support System (CDSS).

---

## 🎯 Objectives

- Predict AKI occurrence before clinical diagnosis
- Classify AKI severity after prediction
- Compare multiple machine learning models
- Build an interpretable prediction pipeline for clinical decision support

---

## 👩‍💻 My Role

- Cohort Construction
- Feature Engineering
- Data Preprocessing
- Missing Value Processing
- Outlier Handling
- Dataset Splitting
- Machine Learning Modeling
- Hyperparameter Optimization
- Performance Evaluation
- SHAP-based Feature Importance Analysis

---

## 📊 Dataset

| Item | Description |
|------|-------------|
| Database | MIMIC-IV |
| Domain | Intensive Care Unit (ICU) |
| Observation Window | 48 Hours |
| Target | Acute Kidney Injury (AKI) |
| Label Definition | KDIGO 2012 Guideline |

---

## 🏗 Project Workflow

```
Cohort Selection
        ↓
Feature Extraction
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
EDA
        ↓
Train / Validation / Test Split
        ↓
AKI Label Generation
        ↓
Stage 1 Model
(AKI Prediction)
        ↓
Stage 2 Model
(AKI Stage Classification)
        ↓
Model Evaluation
        ↓
SHAP Interpretation
```

---

## 📁 Repository Structure

```
final_project_aki

├── 코호트
├── 피처
├── 전처리
├── 변수변환
├── EDA 및 변수변환
├── 데이터 분할
├── 레이블
├── 모델링
├── 최종성능검사
```

---

## 🤖 Models

### Stage 1 — AKI Prediction

- Logistic Regression
- Random Forest
- XGBoost

### Stage 2 — AKI Stage Classification

- LightGBM
- XGBoost
- CatBoost

---

## 📈 Evaluation Metrics

- AUROC
- AUPRC
- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix

---

## 🔍 Feature Engineering

Clinical variables include:

- Demographics
- Vital Signs
- Laboratory Tests
- Urine Output
- Vasopressor Usage

Feature engineering methods:

- Mean
- Maximum
- Minimum
- Sum
- Duration
- Missing Indicator

---

## 🛠 Tech Stack

### Programming

- Python
- SQL

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Visualization

- Matplotlib

---

## 📊 Explainable AI

Model interpretation was performed using **SHAP (SHapley Additive exPlanations)** to identify the most influential clinical variables contributing to AKI prediction.

---

## 💡 Expected Clinical Impact

- Early identification of high-risk AKI patients
- Support timely clinical intervention
- Improve ICU patient management
- Enhance clinical decision-making through AI-based CDSS

---

## 📌 Future Work

- External validation using multicenter datasets
- Real-time prediction pipeline
- Integration with hospital EMR systems
- Deep Learning-based time-series modeling

---

## 👤 Author

**Lee Chae Hui**

Big Data Major

Machine Learning • Healthcare AI • Clinical Decision Support System

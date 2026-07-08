# 🚑 ICU 급성 신손상(AKI) 조기 예측 시스템

> MIMIC-IV 데이터를 활용한 머신러닝 기반 AKI 예측 및 임상 의사결정 지원 시스템(CDSS)

---

# 📌 프로젝트 소개

급성 신손상(Acute Kidney Injury, AKI)은 중환자실(ICU) 환자에게 흔하게 발생하는 합병증으로, 조기 발견 여부가 환자의 예후에 큰 영향을 미칩니다.

본 프로젝트는 **MIMIC-IV 임상 데이터**를 활용하여 AKI를 조기에 예측하고, 발생 환자의 중증도를 분류하는 **2단계 머신러닝 모델**을 개발하였습니다.

---

# 🎯 프로젝트 목표

- ICU 환자의 AKI 조기 예측
- AKI Stage 분류
- 다양한 머신러닝 모델 성능 비교
- 임상 의사결정 지원(CDSS) 기반 마련

---

# 👩‍💻 담당 역할

- Cohort 구축
- Feature Engineering
- 데이터 전처리
- 결측치 처리
- 이상치 처리
- 데이터 분할
- 머신러닝 모델 개발
- 하이퍼파라미터 튜닝
- 모델 성능 평가
- SHAP 기반 변수 중요도 분석

---

# 📊 데이터

|항목|내용|
|---|---|
|데이터셋|MIMIC-IV|
|대상|ICU 환자|
|관찰 기간|48시간|
|예측 대상|AKI 발생 여부|
|Label 기준|KDIGO 2012|

---

# 🏗 프로젝트 진행 과정

```

Cohort 구축
↓
Feature 추출
↓
데이터 전처리
↓
Feature Engineering
↓
EDA
↓
Train / Validation / Test 분할
↓
AKI Label 생성
↓
1단계 모델(AKI 예측)
↓
2단계 모델(Stage 분류)
↓
최종 성능 평가
↓
SHAP 분석

```

---

# 📂 프로젝트 구조

```

📁 코호트
📁 피처
📁 전처리
📁 변수변환
📁 EDA 및 변수변환
📁 데이터 분할
📁 레이블
📁 모델링
📁 최종성능검사

```

---

# 🤖 사용 모델

## 1단계 (AKI 발생 예측)

- Logistic Regression
- Random Forest
- XGBoost

## 2단계 (AKI Stage 분류)

- LightGBM
- XGBoost
- CatBoost

---

# 📈 평가 지표

- AUROC
- AUPRC
- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Confusion Matrix

---

# 🛠 기술 스택

### Language

- Python
- SQL

### Library

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Visualization

- Matplotlib

---

# 🔍 주요 수행 내용

- ICU Cohort 구축
- 임상 변수 추출
- 결측치 및 이상치 처리
- Feature Engineering
- 머신러닝 모델 비교
- SHAP 기반 변수 중요도 분석
- 최종 모델 성능 평가

---

# 📌 기대 효과

- AKI 조기 발견 지원
- 의료진의 의사결정 지원
- 중환자 관리 효율 향상
- AI 기반 CDSS 활용 가능성 제시

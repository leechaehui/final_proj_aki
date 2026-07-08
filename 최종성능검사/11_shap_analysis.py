# ==========================================================
# 11_shap_analysis.py
#
# SHAP 분석
# 1단계: LR 모델 — Non-AKI vs AKI
# 2단계: LGBM 모델 — Stage1 vs Stage2+3
# ==========================================================

import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR    = CURRENT_DIR / "data"
MODEL_DIR   = CURRENT_DIR / "models"
OUT_DIR     = CURRENT_DIR / "outputs" / "shap"

os.makedirs(str(OUT_DIR / "lr"),   exist_ok=True)
os.makedirs(str(OUT_DIR / "lgbm"), exist_ok=True)

print("=" * 70)
print("SHAP 분석 시작")
print("=" * 70)

# ==========================================================
# 데이터 로딩
# ==========================================================

X_train_raw   = np.load(DATA_DIR / "X_train.npy")
X_test_raw    = np.load(DATA_DIR / "X_test.npy")
y_stage_train = np.load(DATA_DIR / "y_stage_train.npy")
y_stage_test  = np.load(DATA_DIR / "y_stage_test.npy")

# ==========================================================
# 모델 로딩
# ==========================================================

with open(MODEL_DIR / "stage1_LR_full.pkl", "rb") as f:
    lr_bundle = pickle.load(f)

with open(MODEL_DIR / "stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    lgbm_bundle = pickle.load(f)

lr_model     = lr_bundle["model"]
lgbm_model   = lgbm_bundle["model"]
feature_cols = lr_bundle["feature_cols"]

print(f"피처 수: {len(feature_cols)}개")
print(f"피처 목록: {feature_cols[:5]} ...")
print()

# DataFrame 변환 (컬럼명 경고 방지)
X_train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test_raw,  columns=feature_cols)

# ==========================================================
# 1단계: LR 모델 SHAP
# ==========================================================

print("=" * 70)
print("1단계 LR 모델 SHAP 분석")
print("=" * 70)

# LR은 Linear Explainer 사용
lr_explainer   = shap.LinearExplainer(lr_model, X_train_df)
lr_shap_values = lr_explainer.shap_values(X_test_df)

# SHAP values shape 확인
print(f"LR SHAP values shape: {lr_shap_values.shape}")
print()

# ----------------------------------------------------------
# LR 1-1. Summary Plot (Beeswarm)
# ----------------------------------------------------------

plt.figure()
shap.summary_plot(
    lr_shap_values,
    X_test_df,
    feature_names=feature_cols,
    max_display=15,
    show=False
)
plt.title("LR SHAP — 피처 중요도 (AKI 예측)", fontsize=13)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "lr" / "shap_summary_beeswarm.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: lr/shap_summary_beeswarm.png")

# ----------------------------------------------------------
# LR 1-2. Bar Plot (평균 절대값)
# ----------------------------------------------------------

plt.figure()
shap.summary_plot(
    lr_shap_values,
    X_test_df,
    feature_names=feature_cols,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("LR SHAP — Top 15 피처 (평균 중요도)", fontsize=13)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "lr" / "shap_summary_bar.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: lr/shap_summary_bar.png")

# ----------------------------------------------------------
# LR 1-3. Waterfall Plot (개별 환자 설명)
#         AKI로 예측된 환자 1명 선택
# ----------------------------------------------------------

lr_pred_proba = lr_model.predict_proba(X_test_df)[:, 1]
aki_indices   = np.where(lr_pred_proba >= lr_bundle["best_threshold"])[0]

if len(aki_indices) > 0:
    sample_idx = aki_indices[0]  # AKI 예측 첫 번째 환자
    explanation = shap.Explanation(
        values        = lr_shap_values[sample_idx],
        base_values   = lr_explainer.expected_value,
        data          = X_test_df.iloc[sample_idx].values,
        feature_names = feature_cols
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"LR SHAP — 환자 {sample_idx} 개별 설명 (AKI 예측)", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "lr" / "shap_waterfall_sample.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: lr/shap_waterfall_sample.png (환자 idx={sample_idx})")

# ----------------------------------------------------------
# LR 1-4. SHAP 값 CSV 저장
# ----------------------------------------------------------

lr_shap_df = pd.DataFrame(lr_shap_values, columns=feature_cols)
lr_shap_df.to_csv(str(OUT_DIR / "lr" / "shap_values.csv"), index=False)
print("저장: lr/shap_values.csv")

# ----------------------------------------------------------
# LR 1-5. 피처 중요도 순위 저장
# ----------------------------------------------------------

lr_importance = pd.DataFrame({
    "feature"          : feature_cols,
    "mean_abs_shap"    : np.abs(lr_shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

lr_importance["rank"] = lr_importance.index + 1
lr_importance.to_csv(str(OUT_DIR / "lr" / "feature_importance.csv"), index=False)

print()
print("LR SHAP Top 10 피처:")
print(lr_importance.head(10).to_string(index=False))

# ==========================================================
# 2단계: LGBM 모델 SHAP
# ==========================================================

print()
print("=" * 70)
print("2단계 LGBM 모델 SHAP 분석")
print("=" * 70)

# LGBM AKI 환자만 사용
aki_train_mask = y_stage_train > 0
aki_test_mask  = y_stage_test  > 0

X_train_aki = X_train_df[aki_train_mask].reset_index(drop=True)
X_test_aki  = X_test_df[aki_test_mask].reset_index(drop=True)

print(f"LGBM SHAP 대상: Test AKI 환자 {len(X_test_aki)}명")

# LGBM은 TreeExplainer 사용 (가장 정확하고 빠름)
lgbm_explainer   = shap.TreeExplainer(lgbm_model)
lgbm_shap_values = lgbm_explainer.shap_values(X_test_aki)

print(f"LGBM SHAP values shape: {lgbm_shap_values.shape}")
print()

# ----------------------------------------------------------
# LGBM 2-1. Summary Plot (Beeswarm)
# ----------------------------------------------------------

plt.figure()
shap.summary_plot(
    lgbm_shap_values,
    X_test_aki,
    feature_names=feature_cols,
    max_display=15,
    show=False
)
plt.title("LGBM SHAP — 피처 중요도 (Stage2+3 예측)", fontsize=13)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "lgbm" / "shap_summary_beeswarm.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: lgbm/shap_summary_beeswarm.png")

# ----------------------------------------------------------
# LGBM 2-2. Bar Plot
# ----------------------------------------------------------

plt.figure()
shap.summary_plot(
    lgbm_shap_values,
    X_test_aki,
    feature_names=feature_cols,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("LGBM SHAP — Top 15 피처 (평균 중요도)", fontsize=13)
plt.tight_layout()
plt.savefig(str(OUT_DIR / "lgbm" / "shap_summary_bar.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: lgbm/shap_summary_bar.png")

# ----------------------------------------------------------
# LGBM 2-3. Waterfall Plot
#           Stage2+3으로 예측된 환자 1명
# ----------------------------------------------------------

lgbm_pred_proba = lgbm_model.predict_proba(X_test_aki)[:, 1]
severe_indices  = np.where(
    lgbm_pred_proba >= lgbm_bundle["best_threshold"]
)[0]

if len(severe_indices) > 0:
    sample_idx = severe_indices[0]
    explanation = shap.Explanation(
        values        = lgbm_shap_values[sample_idx],
        base_values   = lgbm_explainer.expected_value,
        data          = X_test_aki.iloc[sample_idx].values,
        feature_names = feature_cols
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"LGBM SHAP — 환자 {sample_idx} 개별 설명 (Stage2+3 예측)", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "lgbm" / "shap_waterfall_sample.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: lgbm/shap_waterfall_sample.png (환자 idx={sample_idx})")

# ----------------------------------------------------------
# LGBM 2-4. Dependence Plot
#           상위 3개 피처의 영향 방향 시각화
# ----------------------------------------------------------

lgbm_importance = pd.DataFrame({
    "feature"       : feature_cols,
    "mean_abs_shap" : np.abs(lgbm_shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

top3_features = lgbm_importance["feature"].head(3).tolist()

for feat in top3_features:
    feat_idx = list(feature_cols).index(feat)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        lgbm_shap_values,
        X_test_aki,
        feature_names=feature_cols,
        show=False
    )
    plt.title(f"LGBM SHAP Dependence — {feat}", fontsize=13)
    plt.tight_layout()
    fname = feat.replace("/", "_")
    plt.savefig(str(OUT_DIR / "lgbm" / f"shap_dependence_{fname}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"저장: lgbm/shap_dependence_{fname}.png")

# ----------------------------------------------------------
# LGBM 2-5. SHAP 값 CSV 저장
# ----------------------------------------------------------

lgbm_shap_df = pd.DataFrame(lgbm_shap_values, columns=feature_cols)
lgbm_shap_df.to_csv(str(OUT_DIR / "lgbm" / "shap_values.csv"), index=False)
print("저장: lgbm/shap_values.csv")

# ----------------------------------------------------------
# LGBM 2-6. 피처 중요도 순위 저장
# ----------------------------------------------------------

lgbm_importance["rank"] = lgbm_importance.index + 1
lgbm_importance.to_csv(
    str(OUT_DIR / "lgbm" / "feature_importance.csv"), index=False
)

print()
print("LGBM SHAP Top 10 피처:")
print(lgbm_importance.head(10).to_string(index=False))

# ==========================================================
# 두 모델 비교 — 공통 중요 피처
# ==========================================================

print()
print("=" * 70)
print("두 모델 SHAP 피처 중요도 비교")
print("=" * 70)

compare_df = pd.merge(
    lr_importance[["feature", "mean_abs_shap", "rank"]].rename(
        columns={"mean_abs_shap":"lr_shap", "rank":"lr_rank"}),
    lgbm_importance[["feature", "mean_abs_shap", "rank"]].rename(
        columns={"mean_abs_shap":"lgbm_shap", "rank":"lgbm_rank"}),
    on="feature"
).sort_values("lr_rank")

compare_df.to_csv(str(OUT_DIR / "shap_comparison.csv"), index=False)

print(compare_df.head(15).to_string(index=False))

# ----------------------------------------------------------
# 비교 막대 그래프
# ----------------------------------------------------------

top15 = compare_df.head(15)
x     = np.arange(len(top15))
w     = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - w/2, top15["lr_shap"],   w,
       label="LR (AKI 예측)",      color="#185FA5", alpha=0.8)
ax.bar(x + w/2, top15["lgbm_shap"], w,
       label="LGBM (Stage2+3 예측)", color="#E24B4A", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(top15["feature"], rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("LR vs LGBM — 피처 중요도 비교 (SHAP)")
ax.legend()
plt.tight_layout()
plt.savefig(str(OUT_DIR / "shap_model_comparison.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print()
print("저장: shap_model_comparison.png")

# ==========================================================
# 완료
# ==========================================================

print()
print("=" * 70)
print("SHAP 분석 완료")
print("=" * 70)
print(f"저장 위치: {OUT_DIR}")
print()
print("생성 파일:")
print("  outputs/shap/lr/")
print("    - shap_summary_beeswarm.png")
print("    - shap_summary_bar.png")
print("    - shap_waterfall_sample.png")
print("    - shap_values.csv")
print("    - feature_importance.csv")
print("  outputs/shap/lgbm/")
print("    - shap_summary_beeswarm.png")
print("    - shap_summary_bar.png")
print("    - shap_waterfall_sample.png")
print("    - shap_dependence_[피처명].png  ×3")
print("    - shap_values.csv")
print("    - feature_importance.csv")
print("  outputs/shap/")
print("    - shap_comparison.csv")
print("    - shap_model_comparison.png")
# ==========================================================
# 12_final_report.py
#
# AKI CDSS 최종 성능 보고서 생성
# 1단계 LR + 2단계 LGBM + 통합 모델 결과 종합
# ==========================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve,
    f1_score, recall_score, precision_score
)
from sklearn.preprocessing import label_binarize

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR    = CURRENT_DIR / "data"
MODEL_DIR   = CURRENT_DIR / "models"
SHAP_DIR    = CURRENT_DIR / "outputs" / "shap"
OUT_DIR     = CURRENT_DIR / "outputs" / "final_report"

os.makedirs(str(OUT_DIR), exist_ok=True)

print("=" * 70)
print("AKI CDSS 최종 보고서 생성 시작")
print("=" * 70)

# ==========================================================
# 데이터 로딩
# ==========================================================

X_train_raw   = np.load(DATA_DIR / "X_train.npy")
X_valid_raw   = np.load(DATA_DIR / "X_valid.npy")
X_test_raw    = np.load(DATA_DIR / "X_test.npy")
y_train       = np.load(DATA_DIR / "y_train.npy")
y_valid       = np.load(DATA_DIR / "y_valid.npy")
y_test        = np.load(DATA_DIR / "y_test.npy")
y_stage_train = np.load(DATA_DIR / "y_stage_train.npy")
y_stage_valid = np.load(DATA_DIR / "y_stage_valid.npy")
y_stage_test  = np.load(DATA_DIR / "y_stage_test.npy")

# ==========================================================
# 모델 로딩
# ==========================================================

with open(MODEL_DIR / "stage1_LR_full.pkl", "rb") as f:
    lr_bundle = pickle.load(f)

with open(MODEL_DIR / "stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    lgbm_bundle = pickle.load(f)

lr_model     = lr_bundle["model"]
lr_threshold = lr_bundle["best_threshold"]
lgbm_model   = lgbm_bundle["model"]
lgbm_threshold = lgbm_bundle["best_threshold"]
feature_cols = lr_bundle["feature_cols"]

X_train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
X_valid_df = pd.DataFrame(X_valid_raw, columns=feature_cols)
X_test_df  = pd.DataFrame(X_test_raw,  columns=feature_cols)

print("데이터 및 모델 로딩 완료")
print(f"  Train: {len(X_train_raw):,}명")
print(f"  Valid: {len(X_valid_raw):,}명")
print(f"  Test : {len(X_test_raw):,}명")
print()

# ==========================================================
# 1단계 LR 성능 계산
# ==========================================================

print("=" * 70)
print("1단계 LR 모델 성능")
print("=" * 70)

# Valid
lr_proba_valid = lr_model.predict_proba(X_valid_raw)[:, 1]
lr_pred_valid  = (lr_proba_valid >= lr_threshold).astype(int)

# Test
lr_proba_test = lr_model.predict_proba(X_test_raw)[:, 1]
lr_pred_test  = (lr_proba_test >= lr_threshold).astype(int)

def calc_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUROC"      : roc_auc_score(y_true, y_proba),
        "AUPRC"      : average_precision_score(y_true, y_proba),
        "F1"         : f1_score(y_true, y_pred, zero_division=0),
        "Recall"     : recall_score(y_true, y_pred, zero_division=0),
        "Precision"  : precision_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

lr_valid_metrics = calc_metrics(y_valid, lr_pred_valid, lr_proba_valid)
lr_test_metrics  = calc_metrics(y_test,  lr_pred_test,  lr_proba_test)

for name, m in [("Valid", lr_valid_metrics), ("Test", lr_test_metrics)]:
    print(f"\n[{name}]")
    print(f"  AUROC      : {m['AUROC']:.4f}")
    print(f"  AUPRC      : {m['AUPRC']:.4f}")
    print(f"  F1         : {m['F1']:.4f}")
    print(f"  Recall     : {m['Recall']:.4f}")
    print(f"  Specificity: {m['Specificity']:.4f}")

# ==========================================================
# 2단계 LGBM 성능 계산 (AKI 환자만)
# ==========================================================

print()
print("=" * 70)
print("2단계 LGBM 모델 성능")
print("=" * 70)

aki_valid_mask = y_stage_valid > 0
aki_test_mask  = y_stage_test  > 0

X_valid_aki = X_valid_df[aki_valid_mask].reset_index(drop=True)
X_test_aki  = X_test_df[aki_test_mask].reset_index(drop=True)

y_stage_valid_aki = y_stage_valid[aki_valid_mask]
y_stage_test_aki  = y_stage_test[aki_test_mask]

y_lgbm_valid = (y_stage_valid_aki >= 2).astype(int)
y_lgbm_test  = (y_stage_test_aki  >= 2).astype(int)

lgbm_proba_valid = lgbm_model.predict_proba(X_valid_aki)[:, 1]
lgbm_pred_valid  = (lgbm_proba_valid >= lgbm_threshold).astype(int)

lgbm_proba_test = lgbm_model.predict_proba(X_test_aki)[:, 1]
lgbm_pred_test  = (lgbm_proba_test >= lgbm_threshold).astype(int)

lgbm_valid_metrics = calc_metrics(y_lgbm_valid, lgbm_pred_valid, lgbm_proba_valid)
lgbm_test_metrics  = calc_metrics(y_lgbm_test,  lgbm_pred_test,  lgbm_proba_test)

for name, m in [("Valid", lgbm_valid_metrics), ("Test", lgbm_test_metrics)]:
    print(f"\n[{name}]")
    print(f"  AUROC       : {m['AUROC']:.4f}")
    print(f"  AUPRC       : {m['AUPRC']:.4f}")
    print(f"  F1          : {m['F1']:.4f}")
    print(f"  S2+3 Recall : {m['Recall']:.4f}")
    print(f"  S1 Recall   : {m['Specificity']:.4f}")

# Stage별 세부 탐지율
for name, y_stage_aki, lgbm_pred in [
    ("Valid", y_stage_valid_aki, lgbm_pred_valid),
    ("Test",  y_stage_test_aki,  lgbm_pred_test),
]:
    print(f"\n  [{name}] Stage별 탐지율:")
    for sv, sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
        mask = y_stage_aki == sv
        if mask.sum() > 0:
            c = (lgbm_pred[mask]==0).sum() if sv==1 \
                else (lgbm_pred[mask]==1).sum()
            print(f"    {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")

# ==========================================================
# 통합 모델 성능 계산
# ==========================================================

print()
print("=" * 70)
print("통합 모델 성능 (Non-AKI / Stage1 / Stage2+3)")
print("=" * 70)

def get_final_pred(X, lr_m, lgbm_m, lr_t, lgbm_t, feature_cols):
    X_df = pd.DataFrame(X, columns=feature_cols)
    p_aki    = lr_m.predict_proba(X)[:, 1]
    p_severe = lgbm_m.predict_proba(X_df)[:, 1]
    p_nonaki  = 1.0 - p_aki
    p_stage1  = p_aki * (1.0 - p_severe)
    p_stage23 = p_aki * p_severe
    prob = np.column_stack([p_nonaki, p_stage1, p_stage23])
    pred = np.argmax(prob, axis=1)
    return pred, prob

# Test 통합 예측
final_pred_test, final_prob_test = get_final_pred(
    X_test_raw, lr_model, lgbm_model,
    lr_threshold, lgbm_threshold, feature_cols
)

# True label (0=Non-AKI, 1=Stage1, 2=Stage2+3)
y_final_test = np.zeros_like(y_stage_test)
y_final_test[y_stage_test == 1] = 1
y_final_test[(y_stage_test == 2) | (y_stage_test == 3)] = 2

# 클래스별 탐지율
class_names = ["Non-AKI", "Stage1", "Stage2+3"]
for i, name in enumerate(class_names):
    mask    = y_final_test == i
    correct = (final_pred_test[mask] == i).sum()
    rate    = correct / mask.sum()
    print(f"  {name}: {correct}/{mask.sum()} ({rate*100:.1f}%)")

# ==========================================================
# FIGURE 1 — 종합 성능 대시보드
# ==========================================================

print()
print("그래프 생성 중...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("AKI CDSS 최종 모델 성능 대시보드",
             fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.45, wspace=0.35)

# -- 1-1. LR ROC Curve --
ax1 = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, lr_proba_test)
ax1.plot(fpr, tpr, color="#185FA5", linewidth=2,
         label=f"AUROC={lr_test_metrics['AUROC']:.3f}")
ax1.plot([0,1],[0,1],"--",color="gray",alpha=0.5)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("1단계 LR — ROC Curve")
ax1.legend(fontsize=9)

# -- 1-2. LGBM ROC Curve --
ax2 = fig.add_subplot(gs[0, 1])
fpr2, tpr2, _ = roc_curve(y_lgbm_test, lgbm_proba_test)
ax2.plot(fpr2, tpr2, color="#E24B4A", linewidth=2,
         label=f"AUROC={lgbm_test_metrics['AUROC']:.3f}")
ax2.plot([0,1],[0,1],"--",color="gray",alpha=0.5)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("2단계 LGBM — ROC Curve")
ax2.legend(fontsize=9)

# -- 1-3. 통합 모델 탐지율 --
ax3 = fig.add_subplot(gs[0, 2])
rates = []
for i in range(3):
    mask    = y_final_test == i
    correct = (final_pred_test[mask] == i).sum()
    rates.append(correct / mask.sum())
bars = ax3.bar(class_names, rates,
               color=["#185FA5","#E24B4A","#E67E22"])
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Detection Rate")
ax3.set_title("통합 모델 — 클래스별 탐지율")
for bar, val in zip(bars, rates):
    ax3.text(bar.get_x()+bar.get_width()/2,
             val+0.02, f"{val:.3f}", ha='center', fontsize=10)

# -- 2-1. 통합 모델 Confusion Matrix --
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_final_test, final_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=ax4)
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
ax4.set_title("통합 모델 — Confusion Matrix")

# -- 2-2. Normalized Confusion Matrix --
ax5 = fig.add_subplot(gs[1, 1])
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None]
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=ax5)
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")
ax5.set_title("Normalized Confusion Matrix")

# -- 2-3. 성능 지표 비교 표 --
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
metrics_data = [
    ["지표", "1단계 LR", "2단계 LGBM"],
    ["AUROC",      f"{lr_test_metrics['AUROC']:.4f}",
                   f"{lgbm_test_metrics['AUROC']:.4f}"],
    ["AUPRC",      f"{lr_test_metrics['AUPRC']:.4f}",
                   f"{lgbm_test_metrics['AUPRC']:.4f}"],
    ["F1",         f"{lr_test_metrics['F1']:.4f}",
                   f"{lgbm_test_metrics['F1']:.4f}"],
    ["Recall",     f"{lr_test_metrics['Recall']:.4f}",
                   f"{lgbm_test_metrics['Recall']:.4f}"],
    ["Specificity",f"{lr_test_metrics['Specificity']:.4f}",
                   f"{lgbm_test_metrics['Specificity']:.4f}"],
    ["Threshold",  f"{lr_threshold:.2f}",
                   f"{lgbm_threshold:.2f}"],
]
tbl = ax6.table(cellText=metrics_data[1:],
                colLabels=metrics_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#185FA5")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EEF4FB")
ax6.set_title("모델별 성능 지표 (Test)", fontweight='bold')

# -- 3-1. LR SHAP Top 10 --
ax7 = fig.add_subplot(gs[2, 0])
try:
    lr_imp = pd.read_csv(SHAP_DIR / "lr" / "feature_importance.csv")
    top10  = lr_imp.head(10)
    ax7.barh(top10["feature"][::-1], top10["mean_abs_shap"][::-1],
             color="#185FA5", alpha=0.8)
    ax7.set_xlabel("Mean |SHAP|")
    ax7.set_title("1단계 LR — Top 10 피처 중요도")
except:
    ax7.text(0.5, 0.5, "SHAP 파일 없음\n11_shap_analysis.py 먼저 실행",
             ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title("1단계 LR — SHAP (미실행)")

# -- 3-2. LGBM SHAP Top 10 --
ax8 = fig.add_subplot(gs[2, 1])
try:
    lgbm_imp = pd.read_csv(SHAP_DIR / "lgbm" / "feature_importance.csv")
    top10_l  = lgbm_imp.head(10)
    ax8.barh(top10_l["feature"][::-1], top10_l["mean_abs_shap"][::-1],
             color="#E24B4A", alpha=0.8)
    ax8.set_xlabel("Mean |SHAP|")
    ax8.set_title("2단계 LGBM — Top 10 피처 중요도")
except:
    ax8.text(0.5, 0.5, "SHAP 파일 없음\n11_shap_analysis.py 먼저 실행",
             ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title("2단계 LGBM — SHAP (미실행)")

# -- 3-3. 데이터 구성 파이 차트 --
ax9 = fig.add_subplot(gs[2, 2])
counts = [
    (y_final_test == 0).sum(),
    (y_final_test == 1).sum(),
    (y_final_test == 2).sum(),
]
wedges, texts, autotexts = ax9.pie(
    counts,
    labels=class_names,
    autopct='%1.1f%%',
    colors=["#185FA5","#E24B4A","#E67E22"],
    startangle=90
)
for at in autotexts:
    at.set_fontsize(9)
ax9.set_title("Test Set 클래스 분포")

plt.savefig(str(OUT_DIR / "final_dashboard.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: final_dashboard.png")

# ==========================================================
# FIGURE 2 — Precision-Recall 비교
# ==========================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Precision-Recall Curve 비교", fontsize=14, fontweight='bold')

# LR PR Curve
precision, recall, _ = precision_recall_curve(y_test, lr_proba_test)
ap = average_precision_score(y_test, lr_proba_test)
axes2[0].plot(recall, precision, color="#185FA5", linewidth=2,
              label=f"AUPRC={ap:.3f}")
axes2[0].set_xlabel("Recall"); axes2[0].set_ylabel("Precision")
axes2[0].set_title("1단계 LR — PR Curve")
axes2[0].legend()

# LGBM PR Curve
precision2, recall2, _ = precision_recall_curve(y_lgbm_test, lgbm_proba_test)
ap2 = average_precision_score(y_lgbm_test, lgbm_proba_test)
axes2[1].plot(recall2, precision2, color="#E24B4A", linewidth=2,
              label=f"AUPRC={ap2:.3f}")
axes2[1].set_xlabel("Recall"); axes2[1].set_ylabel("Precision")
axes2[1].set_title("2단계 LGBM — PR Curve")
axes2[1].legend()

plt.tight_layout()
plt.savefig(str(OUT_DIR / "pr_curve_comparison.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: pr_curve_comparison.png")

# ==========================================================
# FIGURE 3 — 확률 분포
# ==========================================================

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("예측 확률 분포", fontsize=14, fontweight='bold')

# LR 확률 분포
axes3[0].hist(lr_proba_test[y_test==0], bins=50, alpha=0.6,
              color="#185FA5", label="Non-AKI 실제")
axes3[0].hist(lr_proba_test[y_test==1], bins=50, alpha=0.6,
              color="#E24B4A", label="AKI 실제")
axes3[0].axvline(lr_threshold, color="black", linestyle="--",
                 label=f"Threshold={lr_threshold:.2f}")
axes3[0].set_xlabel("P(AKI)")
axes3[0].set_ylabel("Count")
axes3[0].set_title("1단계 LR — 확률 분포")
axes3[0].legend()

# LGBM 확률 분포
axes3[1].hist(lgbm_proba_test[y_lgbm_test==0], bins=50, alpha=0.6,
              color="#185FA5", label="Stage1 실제")
axes3[1].hist(lgbm_proba_test[y_lgbm_test==1], bins=50, alpha=0.6,
              color="#E24B4A", label="Stage2+3 실제")
axes3[1].axvline(lgbm_threshold, color="black", linestyle="--",
                 label=f"Threshold={lgbm_threshold:.2f}")
axes3[1].set_xlabel("P(Stage2+3)")
axes3[1].set_ylabel("Count")
axes3[1].set_title("2단계 LGBM — 확률 분포")
axes3[1].legend()

plt.tight_layout()
plt.savefig(str(OUT_DIR / "probability_distribution.png"),
            dpi=300, bbox_inches='tight')
plt.close()
print("저장: probability_distribution.png")

# ==========================================================
# CSV 결과 저장
# ==========================================================

# 최종 성능 요약
summary = pd.DataFrame([
    {
        "모델"      : "1단계 LR (AKI 여부)",
        "데이터"    : "Test 전체",
        "AUROC"     : f"{lr_test_metrics['AUROC']:.4f}",
        "AUPRC"     : f"{lr_test_metrics['AUPRC']:.4f}",
        "F1"        : f"{lr_test_metrics['F1']:.4f}",
        "Recall"    : f"{lr_test_metrics['Recall']:.4f}",
        "Specificity": f"{lr_test_metrics['Specificity']:.4f}",
        "Threshold" : f"{lr_threshold:.2f}",
    },
    {
        "모델"      : "2단계 LGBM (Stage 분류)",
        "데이터"    : "Test AKI 환자",
        "AUROC"     : f"{lgbm_test_metrics['AUROC']:.4f}",
        "AUPRC"     : f"{lgbm_test_metrics['AUPRC']:.4f}",
        "F1"        : f"{lgbm_test_metrics['F1']:.4f}",
        "Recall"    : f"{lgbm_test_metrics['Recall']:.4f}",
        "Specificity": f"{lgbm_test_metrics['Specificity']:.4f}",
        "Threshold" : f"{lgbm_threshold:.2f}",
    },
])
summary.to_csv(str(OUT_DIR / "performance_summary.csv"),
               index=False, encoding="utf-8-sig")
print("저장: performance_summary.csv")

# 통합 예측 결과
result_df = pd.DataFrame({
    "y_true_stage"  : y_stage_test,
    "y_true_final"  : y_final_test,
    "y_pred_final"  : final_pred_test,
    "P_NonAKI"      : final_prob_test[:, 0],
    "P_Stage1"      : final_prob_test[:, 1],
    "P_Stage2+3"    : final_prob_test[:, 2],
})
result_df["Label"] = result_df["y_pred_final"].map(
    {0:"Non-AKI", 1:"AKI Stage1", 2:"AKI Stage2+3"}
)
result_df["Correct"] = (
    result_df["y_true_final"] == result_df["y_pred_final"]
).astype(int)

result_df.to_csv(str(OUT_DIR / "final_predictions.csv"),
                 index=False, encoding="utf-8-sig")
print("저장: final_predictions.csv")

# ==========================================================
# 최종 요약 출력
# ==========================================================

print()
print("=" * 70)
print("AKI CDSS 최종 성능 요약 (Test Set)")
print("=" * 70)
print()
print("[ 1단계 — LR: AKI 발생 여부 예측 ]")
print(f"  AUROC      : {lr_test_metrics['AUROC']:.4f}")
print(f"  AUPRC      : {lr_test_metrics['AUPRC']:.4f}")
print(f"  F1         : {lr_test_metrics['F1']:.4f}")
print(f"  Sensitivity: {lr_test_metrics['Recall']:.4f}  ← AKI 탐지율")
print(f"  Specificity: {lr_test_metrics['Specificity']:.4f}  ← 정상 구분율")
print(f"  Threshold  : {lr_threshold:.2f}")
print()
print("[ 2단계 — LGBM: AKI Stage 분류 ]")
print(f"  AUROC        : {lgbm_test_metrics['AUROC']:.4f}")
print(f"  AUPRC        : {lgbm_test_metrics['AUPRC']:.4f}")
print(f"  F1           : {lgbm_test_metrics['F1']:.4f}")
print(f"  S2+3 Recall  : {lgbm_test_metrics['Recall']:.4f}  ← 중증 탐지율")
print(f"  S1 Recall    : {lgbm_test_metrics['Specificity']:.4f}  ← 경증 탐지율")
print(f"  Threshold    : {lgbm_threshold:.2f}")
print()
print("[ 통합 모델 — 클래스별 탐지율 ]")
for i, name in enumerate(class_names):
    mask    = y_final_test == i
    correct = (final_pred_test[mask] == i).sum()
    rate    = correct / mask.sum()
    print(f"  {name}: {correct:,}/{mask.sum():,} ({rate*100:.1f}%)")
print()
print(f"저장 위치: {OUT_DIR}")
print()
print("생성 파일:")
print("  - final_dashboard.png      ← 종합 대시보드")
print("  - pr_curve_comparison.png  ← PR Curve 비교")
print("  - probability_distribution.png ← 확률 분포")
print("  - performance_summary.csv  ← 성능 요약표")
print("  - final_predictions.csv    ← 전체 예측 결과")
print()
print("=" * 70)
print("12_final_report.py 완료")
print("=" * 70)
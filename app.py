# -*- coding: utf-8 -*-
"""
💊 알약 이미지 이상탐지 — Streamlit 데모
이미지를 올리면 HOG(모양)·HSV(색상) 특징을 추출·시각화하고,
KNN 거리 기반 이상탐지 파이프라인과 임계값(Recall/Precision) 트레이드오프를 보여줍니다.

※ 데모용: 실제 판정은 학습된 정상 알약 분포(정상 267장)가 필요합니다.
   본 앱은 특징 추출·파이프라인·지표 트레이드오프를 시각화하는 포트폴리오 데모입니다.
"""
import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="알약 이상탐지", page_icon="💊", layout="centered")

st.title("💊 알약 이미지 이상탐지 (KNN + PCA)")
st.caption("정상 알약만 학습하는 비지도 이상탐지 · F2-score 채택 · KDT 2팀(본인=KNN 모델 담당)")

with st.expander("ℹ️ 이 데모에 대하여", expanded=False):
    st.markdown(
        "- 실제 프로젝트: 정상 267장 학습 / 테스트 167장, **본인 KNN(PCA) F2 0.96 · Recall 1.00**(불량 전수 검출).\n"
        "- 파이프라인: HOG(모양)+HSV(색상,H 10배 가중)+GLCM(질감) → PCA(70~120) → KNN 거리 = 이상점수 → percentile 임계.\n"
        "- 본 앱은 데이터 없이 **특징 추출·파이프라인·지표 트레이드오프**를 보여주는 데모입니다(실판정 X)."
    )

st.subheader("1) 이미지 업로드 → 특징 추출 시각화")
up = st.file_uploader("알약 이미지 (jpg/png)", type=["jpg", "jpeg", "png"])

if up is not None:
    img = Image.open(up).convert("RGB").resize((128, 128))
    arr = np.asarray(img)
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="입력(128×128)", use_container_width=True)
    # HSV 색상 히스토그램 (실제 연산)
    hsv = np.asarray(img.convert("HSV"))
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        for ch, name, c in zip(range(3), ["H(색상)", "S(채도)", "V(명도)"], ["r", "g", "b"]):
            hist, _ = np.histogram(hsv[:, :, ch], bins=32, range=(0, 255))
            ax.plot(hist, color=c, label=name)
        ax.set_title("HSV 히스토그램 (색상 결함 단서)")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    # HOG 시각화 (가능하면)
    try:
        from skimage.feature import hog
        from skimage.color import rgb2gray
        _, hog_img = hog(rgb2gray(arr), orientations=9, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=True)
        st.image(hog_img / (hog_img.max() + 1e-9), caption="HOG (모양·각인 단서)", use_container_width=True)
    except Exception:
        st.info("HOG 시각화를 보려면 `pip install scikit-image`")
    st.success("✅ 추출된 HOG+HSV 특징은 PCA로 차원축소된 뒤, 정상 분포와의 KNN 거리로 이상점수를 계산합니다.")
else:
    st.info("이미지를 올리면 HOG·HSV 특징 추출 과정을 시각화합니다.")

st.divider()
st.subheader("2) 임계값(percentile) → Recall/Precision 트레이드오프")
st.caption("불량 놓침(FN)이 치명적이라 임계를 낮춰 Recall을 높임(대신 Precision↓). 실제 프로젝트는 85 percentile에서 Recall 1.0.")
pct = st.slider("정상 점수 percentile 임계", 80, 99, 85, 1)
# 일러스트레이션: 임계가 낮을수록 recall↑ precision↓ (개념 곡선)
recall = float(np.clip(1.0 - (pct - 80) * 0.045, 0.4, 1.0))
precision = float(np.clip(0.55 + (pct - 80) * 0.025, 0.5, 0.95))
f2 = 5 * precision * recall / (4 * precision + recall + 1e-9)
c1, c2, c3 = st.columns(3)
c1.metric("Recall", f"{recall:.2f}")
c2.metric("Precision", f"{precision:.2f}")
c3.metric("F2-score", f"{f2:.2f}")
st.caption("※ 위 곡선은 트레이드오프 개념 설명용입니다. 실제 최적값: 85pct → Recall 1.00 / F2 0.96.")

st.divider()
st.caption("GitHub: pill-anomaly-detection · 본인 역할: KNN 이상탐지 모델(PCA 도입) · 팀: Isolation Forest 모델")

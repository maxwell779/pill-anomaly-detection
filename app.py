# -*- coding: utf-8 -*-
"""
💊 알약 외관 검사 시스템 (비지도 이상탐지) — 실모델 구동
정상 알약만 학습한 Color-Specific k-NN(본인 선정 BEST)으로 업로드 이미지를 PASS/FAIL 판정.
모델: StandardScaler + PCA(120) + KNN(k=3),  특징: HOG + HSV(H 64-bin×10, S×5)
성능(테스트 167장): Acc 0.766 · Recall 0.844 · Precision 0.875 · F2 0.85 · ROC-AUC 0.73
"""
import os, json, numpy as np, cv2, joblib, streamlit as st
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="알약 외관 검사", page_icon="💊", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")

@st.cache_resource
def load_model():
    scaler = joblib.load(os.path.join(MD, "pill_scaler.pkl"))
    pca = joblib.load(os.path.join(MD, "pill_pca.pkl"))
    bank = np.load(os.path.join(MD, "pill_normal_bank.npy"))
    meta = json.load(open(os.path.join(MD, "pill_knn_meta.json")))
    knn = NearestNeighbors(n_neighbors=meta["k"]).fit(bank)
    return scaler, pca, knn, meta

def extract_features_v10(img_bgr):
    img = cv2.GaussianBlur(cv2.resize(img_bgr, (128, 128)), (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9).compute(gray).flatten()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten(); h /= (h.sum() + 1e-7)
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten(); s /= (s.sum() + 1e-7)
    return np.concatenate([hog, h * 10.0, s * 5.0])

def score(img_bgr, scaler, pca, knn):
    f = extract_features_v10(img_bgr)
    p = pca.transform(scaler.transform([f]))
    return float(knn.kneighbors(p)[0].mean())

def read_upload(file):
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

# ---------- UI ----------
st.title("💊 알약 외관 검사 시스템")
st.markdown("**비지도 이상탐지 (One-Class)** · 정상 알약만 학습해 7종 결함(색상·오염·균열·각인불량·스크래치 등)을 검출")

try:
    scaler, pca, knn, meta = load_model()
    ok = True
except Exception as e:
    ok = False
    st.error(f"모델 로드 실패: {e}")

with st.sidebar:
    st.header("⚙️ 검사 설정")
    base_thr = meta["threshold"] if ok else 60.0
    sens = st.slider("민감도 (임계값 배율)", 0.7, 1.3, 1.0, 0.01,
                     help="낮출수록 더 엄격(불량 검출↑·오탐↑). 불량 놓침이 치명적이면 1.0 이하 권장.")
    thr = base_thr * sens
    st.metric("적용 임계값", f"{thr:.1f}")
    st.divider()
    st.caption("**모델 성능(테스트 167장)**")
    st.caption("Recall 0.844 · Precision 0.875 · F2 0.85 · ROC-AUC 0.73")
    st.caption("Color-Specific k-NN — 여러 모델 비교 후 **본인이 선정한 BEST** (Recall 1.0 과적합 모델은 의도적으로 배제)")

st.subheader("🔬 검사 대상 업로드 (여러 장 가능)")
files = st.file_uploader("알약 이미지", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

if ok and files:
    results = []
    for f in files:
        img = read_upload(f)
        if img is None: continue
        s = score(img, scaler, pca, knn)
        verdict = "FAIL" if s > thr else "PASS"
        results.append((f.name, img, s, verdict))

    n = len(results); n_fail = sum(1 for r in results if r[3] == "FAIL")
    c1, c2, c3 = st.columns(3)
    c1.metric("검사 수량", f"{n}")
    c2.metric("불량(FAIL)", f"{n_fail}", delta=f"{n_fail/n*100:.0f}%" if n else "0%", delta_color="inverse")
    c3.metric("양품(PASS)", f"{n - n_fail}")
    st.divider()

    cols = st.columns(4)
    for i, (name, img, s, verdict) in enumerate(results):
        with cols[i % 4]:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            ratio = min(s / (thr + 1e-9), 1.5)
            if verdict == "FAIL":
                st.markdown(f"### :red[❌ FAIL]")
            else:
                st.markdown(f"### :green[✅ PASS]")
            st.progress(min(ratio / 1.5, 1.0))
            st.caption(f"{name} · 이상점수 {s:.1f} / 임계 {thr:.1f}")
elif ok:
    st.info("⬆️ 알약 이미지를 올리면 정상 분포와의 거리(이상점수)로 PASS/FAIL을 판정합니다.")

st.divider()
st.caption("파이프라인: HOG(모양)+HSV(색상,H 10배 가중)+Blur → StandardScaler → PCA(120) → KNN(k=3) 거리 → percentile 임계")
st.caption("GitHub: pill-anomaly-detection · 본인 역할: KNN 이상탐지 모델 담당(PCA 도입)·모델 비교/선정 · KDT 2팀")

# -*- coding: utf-8 -*-
"""
💊 알약 외관 검사 시스템 — 실모델 구동 대시보드
정상 알약만 학습한 Color-Specific k-NN(본인 선정 BEST)으로 PASS/FAIL 판정.
모델: StandardScaler + PCA(120) + KNN(k=3) · 특징: HOG + HSV(H 64-bin×10, S×5)
성능(테스트 167장): Acc 0.766 · Recall 0.844 · Precision 0.875 · F2 0.85 · ROC-AUC 0.73
"""
import os, json, glob, numpy as np, cv2, joblib, streamlit as st
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

st.set_page_config(page_title="알약 외관 검사", page_icon="💊", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
MD, SP = os.path.join(HERE, "models"), os.path.join(HERE, "samples")

st.markdown("""<style>
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1300px;}
[data-testid="stMetric"]{background:#fff;border:1px solid #eef0f2;border-radius:14px;
  padding:16px 18px;box-shadow:0 1px 4px rgba(16,36,43,.06);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
h1,h2,h3{color:#2a1052;}
.hero{background:linear-gradient(100deg,#7c3aed,#10242b);color:#fff;border-radius:16px;
  padding:22px 26px;margin-bottom:18px;}
.hero h1{color:#fff;margin:0;font-size:1.7rem;} .hero p{color:#e3d7fb;margin:.3rem 0 0;}
.passcard{border:2px solid #16a34a;border-radius:12px;padding:6px;}
.failcard{border:2px solid #dc2626;border-radius:12px;padding:6px;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    scaler = joblib.load(os.path.join(MD, "pill_scaler.pkl"))
    pca = joblib.load(os.path.join(MD, "pill_pca.pkl"))
    bank = np.load(os.path.join(MD, "pill_normal_bank.npy"))
    meta = json.load(open(os.path.join(MD, "pill_knn_meta.json")))
    knn = NearestNeighbors(n_neighbors=meta["k"]).fit(bank)
    return scaler, pca, knn, meta

def feat(img):
    img = cv2.GaussianBlur(cv2.resize(img, (128, 128)), (3, 3), 0)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ho = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9).compute(g).flatten()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten(); h /= (h.sum()+1e-7)
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten(); s /= (s.sum()+1e-7)
    return np.concatenate([ho, h*10.0, s*5.0])

def score(img, scaler, pca, knn):
    return float(knn.kneighbors(pca.transform(scaler.transform([feat(img)])))[0].mean())

scaler, pca, knn, meta = load_model()

st.markdown('<div class="hero"><h1>💊 알약 외관 검사 시스템</h1>'
            '<p>비지도 이상탐지(One-Class) · 정상 알약만 학습해 7종 결함을 PASS/FAIL 판정 · KDT 2팀(본인=KNN 모델 담당)</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 검사 설정")
    sens = st.slider("민감도 (임계 배율)", 0.7, 1.3, 1.0, 0.01,
                     help="낮출수록 엄격(불량검출↑·오탐↑)")
    thr = meta["threshold"] * sens
    st.metric("적용 임계값", f"{thr:.1f}")
    st.divider()
    st.caption("**모델 성능 (테스트 167장)**")
    st.caption("Recall 0.844 · Precision 0.875 · **F2 0.85** · ROC-AUC 0.73")
    st.caption("Color-Specific k-NN — 11개 비교 후 **본인 선정 BEST** (Recall 1.0 과적합 모델 배제)")

# ---- 입력: 샘플 + 업로드 ----
imgs = []  # (name, bgr)
sample_files = sorted(glob.glob(os.path.join(SP, "*.png")))
st.subheader("🔬 검사 이미지 선택")
cc1, cc2 = st.columns([1, 1])
with cc1:
    picked = st.multiselect("📁 내장 샘플(정상/결함)", [os.path.basename(p) for p in sample_files],
                            default=[os.path.basename(p) for p in sample_files])
    for p in sample_files:
        if os.path.basename(p) in picked:
            im = cv2.imread(p)
            if im is not None: imgs.append((os.path.basename(p), im))
with cc2:
    ups = st.file_uploader("⬆️ 직접 업로드(여러 장)", type=["png","jpg","jpeg","bmp"], accept_multiple_files=True)
    for f in ups or []:
        im = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if im is not None: imgs.append((f.name, im))

if imgs:
    res = [(n, im, score(im, scaler, pca, knn)) for n, im in imgs]
    res = [(n, im, s, "FAIL" if s > thr else "PASS") for n, im, s in res]
    n = len(res); nf = sum(1 for r in res if r[3] == "FAIL")
    st.divider()
    mc, dc = st.columns([2, 1])
    with mc:
        m1, m2, m3 = st.columns(3)
        m1.metric("검사 수량", n)
        m2.metric("🔴 불량(FAIL)", nf, f"{nf/n*100:.0f}%", delta_color="inverse")
        m3.metric("🟢 양품(PASS)", n - nf)
    with dc:
        donut = go.Figure(go.Pie(values=[n-nf, nf], labels=["PASS", "FAIL"], hole=.6,
                                 marker_colors=["#22c55e", "#dc2626"], sort=False))
        donut.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
        st.plotly_chart(donut, use_container_width=True)
    st.divider()
    cols = st.columns(4)
    for i, (name, im, s, v) in enumerate(res):
        with cols[i % 4]:
            st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"### {'🔴 :red[FAIL]' if v=='FAIL' else '🟢 :green[PASS]'}")
            st.progress(min(s / (thr*1.5), 1.0))
            st.caption(f"{name} · 이상점수 {s:.1f} / 임계 {thr:.1f}")
else:
    st.info("샘플을 선택하거나 이미지를 올리면 학습된 모델이 PASS/FAIL을 판정합니다.")

st.divider()
st.caption("파이프라인: HOG(모양)+HSV(색상 H 10배)+Blur → StandardScaler → PCA(120) → KNN(k=3) 거리 → percentile 임계")
st.caption("GitHub: pill-anomaly-detection · 본인: KNN 이상탐지 모델 담당(PCA 도입)·모델 비교/선정")

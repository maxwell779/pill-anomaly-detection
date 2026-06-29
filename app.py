# -*- coding: utf-8 -*-
"""
💊 알약 외관 검사 시스템 — 실모델 구동 대시보드
Color-Specific k-NN(본인 선정 BEST)으로 PASS/FAIL 판정. 검사 라인처럼 샘플이 자동 순환.
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
html, body, [class*="css"], .stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"]{
  font-family:'Pretendard','Malgun Gothic','Apple SD Gothic Neo',sans-serif;}
/* rerun 시 요소 재렌더로 인한 흔들림(FOUT 리플로우) 억제 */
[data-testid="stElementContainer"], .element-container{animation:none !important;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
.block-container{padding-top:1.3rem;max-width:1300px;}
[data-testid="stMetric"]{background:#fff;border:1px solid #ece8f5;border-radius:16px;
  padding:16px 18px;box-shadow:0 2px 10px rgba(42,16,82,.05);transition:transform .15s, box-shadow .15s;}
[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(124,58,237,.14);}
[data-testid="stMetricLabel"] p{color:#64748b;font-weight:600;}
[data-testid="stMetricValue"]{color:#2a1052;font-weight:800;}
h1,h2,h3{color:#2a1052;letter-spacing:-.3px;}
.hero{background:linear-gradient(110deg,#7c3aed 0%,#10242b 100%);color:#fff;border-radius:18px;
  padding:24px 30px;margin-bottom:18px;box-shadow:0 12px 32px rgba(124,58,237,.22);}
.hero h1{color:#fff;margin:0;font-size:1.85rem;font-weight:800;letter-spacing:-.5px;} .hero p{color:#e3d7fb;margin:.4rem 0 0;font-size:.97rem;}
.hero .chip{display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.28);
  color:#f3ecff;border-radius:20px;padding:4px 13px;font-size:.8rem;font-weight:600;margin:11px 6px 0 0;}
.stamp{font-size:2rem;font-weight:900;text-align:center;border-radius:12px;padding:10px;margin-top:8px;}
.s-pass{background:#dcfce7;color:#15803d;border:3px solid #22c55e;}
.s-fail{background:#fee2e2;color:#b91c1c;border:3px solid #ef4444;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    scaler = joblib.load(os.path.join(MD, "pill_scaler.pkl"))
    pca = joblib.load(os.path.join(MD, "pill_pca.pkl"))
    bank = np.load(os.path.join(MD, "pill_normal_bank.npy"))
    meta = json.load(open(os.path.join(MD, "pill_knn_meta.json")))
    knn = NearestNeighbors(n_neighbors=meta["k"]).fit(bank)
    return scaler, pca, knn, meta

def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

def feat(img):
    img = cv2.GaussianBlur(cv2.resize(img, (128, 128)), (3, 3), 0)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ho = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9).compute(g).flatten()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten(); h /= (h.sum() + 1e-7)
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten(); s /= (s.sum() + 1e-7)
    return np.concatenate([ho, h * 10.0, s * 5.0])

@st.cache_data(show_spinner=False)
def score_path(path):
    scaler, pca, knn, meta = load_model()
    img = imread_u(path)
    return float(knn.kneighbors(pca.transform(scaler.transform([feat(img)])))[0].mean())

def score_arr(img):
    scaler, pca, knn, meta = load_model()
    return float(knn.kneighbors(pca.transform(scaler.transform([feat(img)])))[0].mean())

scaler, pca, knn, meta = load_model()

st.markdown('<div class="hero"><h1>💊 알약 외관 검사 시스템</h1>'
            '<p>비지도 이상탐지(One-Class) · 정상 알약만 학습해 7종 결함을 PASS/FAIL 판정 · KDT 팀(본인=KNN 모델 담당)</p>'
            '<div><span class="chip">💊 7종 결함 탐지</span><span class="chip">🎯 F2 0.85 · Recall 0.84</span>'
            '<span class="chip">🔬 KNN+PCA · 11종 비교 BEST</span></div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 검사 설정")
    sens = st.slider("민감도 (임계 배율)", 0.7, 1.3, 1.0, 0.01, help="낮출수록 엄격")
    thr = meta["threshold"] * sens
    speed = st.slider("검사 라인 주기(초)", 1.0, 5.0, 2.5, 0.5)
    st.metric("적용 임계값", f"{thr:.1f}")
    st.divider()
    st.caption("**모델 성능 (테스트 167장)**")
    st.caption("Recall 0.844 · Precision 0.875 · **F2 0.85** · ROC-AUC 0.73")
    st.caption("Color-Specific k-NN — 11개 비교 후 **본인 선정 BEST**")

samples = sorted(glob.glob(os.path.join(SP, "*.png")))

# ---- 전체 샘플 통계 (항상 표시) ----
results = [(os.path.basename(p), p, score_path(p)) for p in samples]
n = len(results); nf = sum(1 for _, _, s in results if s > thr)
k1, k2, k3, k4 = st.columns([1, 1, 1, 1.2])
k1.metric("검사 수량", n)
k2.metric("🔴 불량(FAIL)", nf, f"{nf/max(n,1)*100:.0f}%", delta_color="inverse")
k3.metric("🟢 양품(PASS)", n - nf)
with k4:
    donut = go.Figure(go.Pie(values=[n - nf, nf], labels=["PASS", "FAIL"], hole=.62,
                             marker_colors=["#22c55e", "#dc2626"], sort=False))
    donut.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=True, legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(donut, use_container_width=True)

st.divider()
st.subheader("🏭 검사 라인 (자동 순환)")

if samples:
    if "pl_i" not in st.session_state: st.session_state.pl_i = 0

    @st.fragment(run_every=f"{speed}s")
    def line():
        i = st.session_state.pl_i % len(samples)
        name, p, sc = results[i]
        verdict = "FAIL" if sc > thr else "PASS"
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.image(cv2.cvtColor(imread_u(p), cv2.COLOR_BGR2RGB), use_container_width=True)
            cls = "s-fail" if verdict == "FAIL" else "s-pass"
            mark = "❌ FAIL" if verdict == "FAIL" else "✅ PASS"
            st.markdown(f'<div class="stamp {cls}">{mark}</div>', unsafe_allow_html=True)
            st.caption(f"검사 #{i+1}/{len(samples)} · {name} · 이상점수 {sc:.1f} / 임계 {thr:.1f}")
        with c2:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=sc, number={'suffix': ""},
                title={'text': "이상 점수"},
                gauge={'axis': {'range': [0, max(thr * 1.8, sc * 1.2)]},
                       'bar': {'color': "#dc2626" if verdict == "FAIL" else "#22c55e"},
                       'threshold': {'line': {'color': "black", 'width': 3}, 'value': thr},
                       'steps': [{'range': [0, thr], 'color': '#dcfce7'},
                                 {'range': [thr, max(thr * 1.8, sc * 1.2)], 'color': '#fee2e2'}]}))
            g.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(g, use_container_width=True)
        st.session_state.pl_i = i + 1
    line()

with st.expander("⬆️ 직접 업로드해서 검사 (여러 장)"):
    ups = st.file_uploader("알약 이미지", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
    if ups:
        cols = st.columns(4)
        for j, f in enumerate(ups):
            im = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            sc = score_arr(im); v = "FAIL" if sc > thr else "PASS"
            with cols[j % 4]:
                st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown(f"### {'🔴 :red[FAIL]' if v=='FAIL' else '🟢 :green[PASS]'}")
                st.caption(f"{f.name} · {sc:.1f}/{thr:.1f}")

st.divider()
st.caption("파이프라인: HOG+HSV(H 10배)+Blur → StandardScaler → PCA(120) → KNN(k=3) 거리 → percentile 임계")
st.caption("GitHub: pill-anomaly-detection · 본인: KNN 이상탐지 모델 담당(PCA 도입)·모델 비교/선정")

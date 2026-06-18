# 💊 알약 이미지 이상탐지 (KNN + PCA, 비지도)

> 정상 알약만 학습해(비지도 One-Class) 7종 결함을 잡는 이미지 이상탐지. 불량을 놓치면 치명적이라 Accuracy 대신 **Recall 가중 F2-score**를 핵심 지표로 채택.

| 항목 | 내용 |
|---|---|
| 기간 | 2026.03.17 ~ 2026.03.23 |
| 팀 | 2팀 (KDT 팀 프로젝트) |
| **나의 역할** | **KNN 이상탐지 모델 담당 (PCA 차원축소 도입)** · OneClassSVM 실험 |

> ℹ️ KDT 부트캠프 팀 프로젝트입니다. 본 저장소는 **본인의 KNN 모델** 작업이며, 팀원은 Isolation Forest 모델을 담당했습니다.

---

## 🎯 문제 정의
알약 결함은 종류가 많고(색상·오염·균열·각인불량·스크래치 등 **7유형**) 불량 샘플이 적다 → **정상만 학습하는 비지도(OCC)** 접근. **불량 놓침(FN)이 치명적**이라 Recall에 2배 가중한 F2-score로 평가.
- 데이터: 정상 **267장** 학습 / 테스트 **167장**(정상 26 + 불량 141).

## 🛠 기술 스택
`Python` · `scikit-learn` · `KNN` · `PCA` · `OneClassSVM` · `HOG / HSV / GLCM` · `OpenCV`

## 🔧 핵심 구현
1. **특징 추출** — HOG(모양·각인) + HSV 색상 히스토그램(**H 채널 10배 가중**) + GLCM(질감).
2. **전처리** — Gaussian Blur(노이즈 제거) + StandardScaler(스케일 정합).
3. **PCA 차원축소** — 고차원(700+) → 70~120차원으로 거리 계산 안정화.
4. **KNN 이상탐지** — 정상만 학습 후 k-최근접 거리 평균 = 이상점수(k=1~5), 정상 점수 percentile 임계(95→85).
5. **지표** — F2-score(β=2)로 모델 선택.

## 🔧 트러블슈팅
| 문제 | 해결 |
|---|---|
| 색상 결함 미검출(HOG는 모양만) | HSV(H 10배)+GLCM 추가 |
| 차원의 저주(거리 흔들림) | PCA 차원축소 |
| HOG+Color 단순결합 역효과(F2 0.42) | Blur + 가중 결합으로 재구성 → **F2 0.96** |
| Precision–Recall 트레이드오프 | 불량 놓침 방지 우선 → 85 percentile → **Recall 1.0** (Precision ≈0.84) |

## 📈 결과
- HOG-only **F2 0.62** → 최적화 후 **F2 0.96 / Recall 1.00** (불량 141장 전수 검출).
- 팀의 Isolation Forest 모델과 **동일 F2 기준으로 공정 비교**.
- ⚠️ Recall 1.0은 임계를 공격적으로 잡은 결과로 Precision은 ≈0.84 (FN 회피가 우선인 맥락에선 타당).

## 🖥 데모 (Streamlit)
이미지를 올리면 **HOG·HSV 특징 추출을 시각화**하고, KNN 임계값(percentile)에 따른 **Recall/Precision 트레이드오프**를 보여주는 데모.
```bash
pip install -r requirements.txt
streamlit run app.py
```
> 실제 정상/이상 판정은 학습된 정상 분포(267장)가 필요해, 본 데모는 특징추출·파이프라인 시각화 중심입니다.

## 📁 구조
```
app.py                   # Streamlit 데모(특징 시각화 + 지표 트레이드오프)
pill_knn_anomaly.ipynb   # 본인 작업: 특징추출·PCA·KNN·F2 평가 전 과정
requirements.txt
```
> 알약 이미지 데이터셋·모델(.pkl)은 용량 관계로 제외. 과정·결과는 노트북에 포함.

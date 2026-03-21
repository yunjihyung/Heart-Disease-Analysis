import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="심장질환 위험도 예측 시스템",
    page_icon="💓",
    layout="wide"
)

# ===============================
# 모델 로드
# ===============================
@st.cache_resource
def load_models():
    bundle = joblib.load("final_models.pkl")
    return (
        bundle["perf_model"],
        bundle["threshold"],
        bundle["explain_model"],
        bundle["cal_model"],
        bundle["features"]
    )

perf_model, threshold, explain_model, cal_model, features = load_models()

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Pretendard", "Noto Sans KR", sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.hero-box {
    background: linear-gradient(135deg, #e8f3ff 0%, #f5fbff 100%);
    border: 1px solid #d6e9ff;
    border-radius: 20px;
    padding: 28px 28px 20px 28px;
    margin-bottom: 24px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
}
.card {
    background: white;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    border: 1px solid #edf1f7;
    margin-bottom: 18px;
}
.risk-high {
    background: linear-gradient(135deg, #fff0f0 0%, #fff7f7 100%);
    border-left: 8px solid #e74c3c;
}
.risk-low {
    background: linear-gradient(135deg, #ecfff8 0%, #f8fffc 100%);
    border-left: 8px solid #16a085;
}
.score-box {
    background: #f8fbff;
    border: 1px solid #dcecff;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
}
.mini-badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
}
.badge-risk {
    background: #ffe4e1;
    color: #c0392b;
}
.badge-safe {
    background: #ddfff4;
    color: #117a65;
}
.small-note {
    color: #5f6b7a;
    font-size: 0.92rem;
    line-height: 1.6;
}
.var-up {
    color: #c0392b;
    font-weight: 600;
}
.var-down {
    color: #117a65;
    font-weight: 600;
}
.var-neutral {
    color: #7f8c8d;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 상단 소개
# ===============================
st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:6px;">💓 심장질환 위험도 예측 시스템</h1>
    <p style="font-size:1.05rem; color:#425466; margin-bottom:10px;">
        건강 관련 정보를 입력하면 <b>위험/안전 분류</b>, <b>위험도 점수</b>, 그리고 <b>계수 기반 설명</b>을 제공합니다.
    </p>
    <p class="small-note">
        분류 결과는 성능 최적화용 로지스틱 회귀 모델과 임계값(threshold)을 기반으로 계산하고,
        위험도 점수는 별도의 보정(calibration)된 확률 모델을 이용합니다.
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# 입력 UI
# ===============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 건강 정보 입력")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("성별", ["Female", "Male"])
    age = st.selectbox("연령대", [
        '18-24','25-29','30-34','35-39','40-44',
        '45-49','50-54','55-59','60-64','65-69',
        '70-74','75-79','80 or older'
    ])
    height = st.number_input("키(cm)", min_value=100, max_value=220, value=170)
    weight = st.number_input("몸무게(kg)", min_value=30, max_value=200, value=65)

    race = st.selectbox("인종", [
        'White', 'Black', 'Asian',
        'American Indian/Alaskan Native', 'Other', 'Hispanic'
    ])

    genhealth = st.selectbox("일반적 건강 상태", [
        'Poor', 'Fair', 'Good', 'Very good', 'Excellent'
    ])

    asthma = st.selectbox("천식 여부", ["Yes", "No"])
    kidney = st.selectbox("신장질환 여부", ["Yes", "No"])
    skin_cancer = st.selectbox("피부암 진단 여부", ["Yes", "No"])

with col2:
    smoking = st.selectbox("흡연 여부", ["Yes", "No"])
    alcohol = st.selectbox("음주 여부", ["Yes", "No"])
    stroke = st.selectbox("뇌졸중 진단 여부", ["Yes", "No"])
    diffwalking = st.selectbox("보행 어려움 여부", ["Yes", "No"])

    diabetic = st.selectbox("당뇨 여부", [
        'Yes', 'No'
    ])

    physical_activity = st.selectbox("규칙적 신체활동 여부", ["Yes", "No"])

    mental_health = st.slider("정신 건강 문제 일수 (최근 30일 중)", 0, 30, 3)
    sleep_time = st.slider("평균 수면 시간(시간)", 0, 24, 7)
    # 혹시 이전 학습 모델에 PhysicalHealth가 있을 수 있으므로 남겨둠
    physical_health = st.slider("신체 건강 문제 일수 (최근 30일 중)", 0, 30, 3)

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# 전처리 함수
# ===============================
def build_input_df(features):
    d = {"Yes": 1, "No": 0}
    d_sex = {"Female": 0, "Male": 1}
    age_order = [
        '18-24','25-29','30-34','35-39','40-44',
        '45-49','50-54','55-59','60-64','65-69',
        '70-74','75-79','80 or older'
    ]
    # 기존 코드 흐름 유지: Poor=0 ... Excellent=4
    health_order = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']

    bmi_raw = weight / ((height / 100) ** 2) 

    row = {col: 0 for col in features}

    # 공통 입력값 채우기 (모델 feature에 있을 때만)
    mapping = {
        "BMI": bmi_raw,
        "Smoking": d[smoking],
        "AlcoholDrinking": d[alcohol],
        "Stroke": d[stroke],
        "MentalHealth": mental_health,
        "PhysicalHealth": physical_health,
        "SleepTime": sleep_time,
        "DiffWalking": d[diffwalking],
        "Sex": d_sex[sex],
        "AgeCategory": age_order.index(age),
        "Diabetic": d[diabetic],
        "PhysicalActivity": d[physical_activity],
        "GenHealth": health_order.index(genhealth),
        "Asthma": d[asthma],
        "KidneyDisease": d[kidney],
        "SkinCancer": d[skin_cancer],
        
    }

    for k, v in mapping.items():
        if k in row:
            row[k] = v

    # race dummy 처리: 저장된 feature에 맞춰 유연하게 주입
    race_to_feature = {
        "Asian": "Race_Asian",
        "Black": "Race_Black",
        "Hispanic": "Race_Hispanic",
        "Other": "Race_Other",
        "White": "Race_White",
        "American Indian/Alaskan Native": "Race_American Indian/Alaskan Native"
    }

    chosen_race_feature = race_to_feature.get(race)
    if chosen_race_feature in row:
        row[chosen_race_feature] = 1

    return pd.DataFrame([row]), bmi_raw


# ===============================
# 설명용 텍스트 헬퍼
# ===============================
feature_kor = {
    "BMI": "BMI",
    "Smoking": "흡연",
    "AlcoholDrinking": "음주",
    "Stroke": "뇌졸중 병력",
    "MentalHealth": "정신 건강 문제 일수",
    "PhysicalHealth": "신체 건강 문제 일수",
    "SleepTime": "수면 시간",
    "DiffWalking": "보행 어려움",
    "Sex": "성별(남성 기준)",
    "AgeCategory": "연령대",
    "Diabetic": "당뇨 여부",
    "PhysicalActivity": "규칙적 신체활동",
    "GenHealth": "주관적 건강 상태",
    "Asthma": "천식",
    "KidneyDisease": "신장질환",
    "SkinCancer": "피부암 병력",
    "Race_Asian": "아시아계",
    "Race_Black": "흑인",
    "Race_Hispanic": "히스패닉",
    "Race_Other": "기타 인종",
    "Race_White": "백인",
    "Race_American Indian/Alaskan Native": "미국 원주민/알래스카 원주민"
}

def feature_display_name(col):
    return feature_kor.get(col, col)

def build_contribution_df(df_input):
    coef = explain_model.coef_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "value": df_input.iloc[0][features].values.astype(float),
        "coef": coef
    })
    coef_df["contribution"] = coef_df["value"] * coef_df["coef"]
    coef_df["abs_contribution"] = np.abs(coef_df["contribution"])
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df["feature_kor"] = coef_df["feature"].map(feature_display_name)

    return coef_df

def summarize_contributions(coef_df, top_n=5,threshold=0.05):
    # 1. 영향 있는 변수만 필터
    important_df = coef_df[
        coef_df["abs_contribution"] >= threshold
    ].copy()

    # 2. 위험 증가
    pos_df = important_df[important_df["contribution"] > 0] \
        .sort_values("contribution", ascending=False)

    # 3. 위험 감소
    neg_df = important_df[important_df["contribution"] < 0] \
        .sort_values("contribution", ascending=True)

    # 4. 상위만 자르기
    pos_df = pos_df.head(top_n)
    neg_df = neg_df.head(top_n)

    # 영향이 거의 없는 변수
    neutral_df = coef_df.sort_values(
        "abs_contribution", ascending=True
    ).head(top_n)

    return pos_df, neg_df, neutral_df

def risk_band(prob):
    if prob < 0.15:
        return "낮음"
    elif prob < 0.30:
        return "보통"
    else:
        return "높음"


# ===============================
# 예측 버튼
# ===============================
if st.button("🚀 예측하기", use_container_width=True):
    try:
        df_input, bmi_raw = build_input_df(features)

        # 성능용: raw prob + threshold
        perf_prob = perf_model.predict_proba(df_input)[:, 1][0]
        pred_label = int(perf_prob >= threshold)

        # 확률용: calibrated prob
        cal_prob = cal_model.predict_proba(df_input)[:, 1][0]

        # 설명용
        coef_df = build_contribution_df(df_input)
        pos_df, neg_df, neutral_df = summarize_contributions(coef_df, top_n=5)

        # 결과 섹션
        left, right = st.columns([1.2, 1])

        with left:
            box_class = "risk-high" if pred_label == 1 else "risk-low"
            badge = "badge-risk" if pred_label == 1 else "badge-safe"
            label_text = "위험" if pred_label == 1 else "안전"

            st.markdown(f"""
            <div class="card {box_class}">
                <span class="mini-badge {badge}">분류 결과</span>
                <h2 style="margin-top:10px; margin-bottom:10px;">
                    {'⚠️ 심장질환 위험군' if pred_label == 1 else '✅ 상대적 저위험군'}
                </h2>
                <p style="font-size:1.02rem; line-height:1.7;">
                    성능 최적화용 로지스틱 회귀 모델과 임계값 <b>{threshold:.2f}</b>을 기준으로
                    현재 입력은 <b>{label_text}</b>으로 분류되었습니다.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="score-box">
                <h4 style="margin-bottom:6px;">📈 위험도 점수</h4>
                <div style="font-size:2.1rem; font-weight:700; color:#2c3e50;">
                    {cal_prob*100:.1f}%
                </div>
                <div style="margin-top:8px; font-size:1rem; color:#5f6b7a;">
                    보정된(calibrated) 확률 기준
                </div>
                <div style="margin-top:10px; font-weight:600;">
                    위험도 수준: {risk_band(cal_prob)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 기본 정보 요약
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧾 입력 요약")
        summary_cols = st.columns(4)
        summary_cols[0].metric("BMI", f"{bmi_raw:.1f}")
        summary_cols[1].metric("연령대", age)
        summary_cols[2].metric("수면시간", f"{sleep_time}시간")
        summary_cols[3].metric("정신건강 문제일수", f"{mental_health}일")

        st.markdown(f"""
        <p class="small-note">
            주의: 위험도 점수는 참고용 예측값이며, 실제 진단을 대체하지 않습니다.
            증상이 있거나 걱정되는 경우 반드시 의료진과 상담하세요.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 계수 기반 설명
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 계수 기반 설명")

        st.markdown("""
        <p class="small-note">
            아래 설명은 설명용 로지스틱 회귀 모델의 계수와 현재 입력값을 바탕으로,
            이번 예측에서 상대적으로 영향을 크게 준 변수와 거의 영향을 주지 않은 변수를 정리한 것입니다.
        </p>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 🔺 위험을 높인 변수")
            if len(pos_df) == 0:
                st.info("이번 입력에서는 뚜렷한 위험 증가 요인이 크지 않았습니다.")
            else:
                for _, row in pos_df.iterrows():
                    st.markdown(
                        f"- <span class='var-up'>{row['feature_kor']}</span> "
                        f"(기여도: {row['contribution']:.3f})",
                        unsafe_allow_html=True
                    )

        with c2:
            st.markdown("### 🔻 위험을 낮춘 변수")
            if len(neg_df) == 0:
                st.info("이번 입력에서는 뚜렷한 보호 방향 요인이 크지 않았습니다.")
            else:
                for _, row in neg_df.iterrows():
                    st.markdown(
                        f"- <span class='var-down'>{row['feature_kor']}</span> "
                        f"(기여도: {row['contribution']:.3f})",
                        unsafe_allow_html=True
                    )

        with c3:
            st.markdown("### ⚪ 영향이 거의 없던 변수")
            if len(neutral_df) == 0:
                st.info("영향이 작은 변수를 계산할 수 없습니다.")
            else:
                for _, row in neutral_df.iterrows():
                    st.markdown(
                        f"- <span class='var-neutral'>{row['feature_kor']}</span> "
                        f"(기여도: {row['contribution']:.3f})",
                        unsafe_allow_html=True
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        # 상세 표
        with st.expander("변수별 상세 기여도 보기"):
            show_df = coef_df[["feature_kor", "value", "coef", "odds_ratio", "contribution", "abs_contribution"]].copy()
            show_df.columns = ["변수", "입력값", "계수", "오즈비", "기여도", "절대기여도"]
            st.dataframe(
                show_df.sort_values("절대기여도", ascending=False),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
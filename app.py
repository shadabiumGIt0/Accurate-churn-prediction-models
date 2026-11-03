import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------
# PAGE CONFIG & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Bank Subscriber Attrition Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .metric-box { padding: 1rem; background-color: #F3F4F6; border-left: 5px solid #2563EB; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-header'>Bank Subscriber Retention Architecture</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Operational Risk Engine & What-If Churn Profiler</div>", unsafe_allow_html=True)

RISK_THRESHOLD = 0.45
GEOGRAPHIES = ["France", "Germany", "Spain"]
GENDERS = ["Female", "Male"]
NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Balance_to_Salary_Ratio", "Product_Density",
    "Engagement_Score", "Age_Tenure_Interaction",
]

# ----------------------------------------------------
# 1. DATA, FEATURE PIPELINE & MODEL
# ----------------------------------------------------
@st.cache_data
def get_raw_training_context() -> pd.DataFrame:
    raw_data = """Year,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
2025,619,France,Female,42,2,0,1,1,1,101348.88,1
2025,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
2025,502,France,Female,42,8,159660.8,3,1,0,113931.57,1
2025,699,France,Female,39,1,0,2,0,0,93826.63,0
2025,850,Spain,Female,43,2,125510.82,1,1,1,79084.1,0
2025,645,Spain,Male,44,8,113755.78,2,1,0,149756.71,1
2025,822,France,Male,50,7,0,2,1,1,10062.8,0
2025,376,Germany,Female,29,4,115046.74,4,1,0,119346.88,1
2025,501,France,Male,44,4,142051.07,2,0,1,74940.5,0
2025,684,France,Male,27,2,134603.88,1,1,1,71725.73,0
2025,756,Germany,Male,36,2,136815.64,1,1,1,170041.95,0
2025,591,Spain,Female,39,3,0,3,1,0,140469.38,1
2025,553,Germany,Male,41,9,110112.54,2,0,0,81898.81,0
2025,475,France,Female,45,0,134264.04,1,1,0,27822.99,1
2025,850,France,Male,36,7,0,1,1,1,40812.9,0
2025,465,France,Female,51,8,122522.32,1,0,0,181297.65,1
2025,834,France,Female,49,2,131394.56,1,0,0,194365.76,1
2025,637,Germany,Female,39,9,137843.8,1,1,1,117622.8,1
2025,721,Germany,Male,28,9,154475.54,2,0,1,101300.94,1
2025,584,Germany,Female,30,8,112013.81,1,1,0,177772.03,1"""
    return pd.read_csv(io.StringIO(raw_data))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and one-hot encode with a fixed category set."""
    out = df.copy()
    out["Balance_to_Salary_Ratio"] = out["Balance"] / (out["EstimatedSalary"] + 1e-5)
    out["Product_Density"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    out["Engagement_Score"] = out["HasCrCard"].astype(int) + out["IsActiveMember"].astype(int) * 2
    out["Age_Tenure_Interaction"] = out["Age"] * (out["Tenure"] + 1)

    # Fixed categories guarantee identical dummy columns for any input size
    out["Geography"] = pd.Categorical(out["Geography"], categories=GEOGRAPHIES)
    out["Gender"] = pd.Categorical(out["Gender"], categories=GENDERS)
    out = pd.get_dummies(out, columns=["Geography", "Gender"], drop_first=False, dtype=int)
    return out


@st.cache_resource
def train_engine():
    """Fit scaler and model once on the training context."""
    df = get_raw_training_context()
    y = df["Exited"]
    X = build_features(df.drop(columns=["Year", "Exited"]))
    feature_cols = list(X.columns)

    scaler = StandardScaler()
    X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"
    )
    model.fit(X[feature_cols], y)
    return model, scaler, feature_cols


engine, scaler, FEATURE_COLS = train_engine()


def transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(raw_df)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    return X[FEATURE_COLS]


# ----------------------------------------------------
# 2. SIDEBAR INPUTS
# ----------------------------------------------------
st.sidebar.header("Customer Attribute Controls")

input_geo = st.sidebar.selectbox("Geographic Jurisdiction", GEOGRAPHIES)
input_gender = st.sidebar.selectbox("Gender Designation", GENDERS)
input_age = st.sidebar.slider("Customer Age", 18, 90, 38)
input_score = st.sidebar.slider("Credit Rating Score", 350, 850, 620)
input_tenure = st.sidebar.slider("Account Relationship Seniority (Tenure Years)", 0, 10, 4)
input_balance = st.sidebar.number_input("Account Balance Capital (€)", min_value=0.0, value=75000.0, step=5000.0)
input_products = st.sidebar.selectbox("Active Bank Products Enrolled", [1, 2, 3, 4], index=0)
input_salary = st.sidebar.number_input("Estimated Annual Compensation (€)", min_value=1000.0, value=85000.0, step=5000.0)

st.sidebar.subheader("Engagement Modifiers")
input_crcard = st.sidebar.checkbox("Possesses Active Credit Card Product", value=True)
input_active = st.sidebar.checkbox("Marked as Domestically Active Member", value=True)

raw_vector = pd.DataFrame([{
    "CreditScore": input_score,
    "Age": input_age,
    "Tenure": input_tenure,
    "Balance": input_balance,
    "NumOfProducts": input_products,
    "HasCrCard": int(input_crcard),
    "IsActiveMember": int(input_active),
    "EstimatedSalary": input_salary,
    "Geography": input_geo,
    "Gender": input_gender,
}])

risk_probability = float(engine.predict_proba(transform(raw_vector))[0][1])
is_high_risk = risk_probability > RISK_THRESHOLD
risk_flag = "HIGH RISK ALERT" if is_high_risk else "STABLE (LOW RISK)"

# ----------------------------------------------------
# 3. RISK GAUGE & SCORECARDS
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Inference Result")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability (%)", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "#EF4444" if is_high_risk else "#10B981"},
            "bgcolor": "white",
            "steps": [
                {"range": [0, RISK_THRESHOLD * 100], "color": "#E5E7EB"},
                {"range": [RISK_THRESHOLD * 100, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": RISK_THRESHOLD * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if is_high_risk:
        st.error(f"Status Flags: {risk_flag}")
    else:
        st.success(f"Status Flags: {risk_flag}")

with col2:
    st.subheader("Account Summary Vectors")

    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown(
            f"""
            <div class='metric-box'>
                <small>BALANCE-SALARY CORRELATION</small><br>
                <b style='font-size: 1.4rem;'>{(input_balance / (input_salary + 1e-5)):.2f}x Ratio</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class='metric-box'>
                <small>PRODUCT DENSITY LEVEL</small><br>
                <b style='font-size: 1.4rem;'>{(input_products / (input_tenure + 1)):.2f} Index</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_c2:
        st.markdown(
            f"""
            <div class='metric-box'>
                <small>ENGAGEMENT LEVEL COEFFICIENT</small><br>
                <b style='font-size: 1.4rem;'>{int(input_crcard) + int(input_active) * 2} / 3 Score</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class='metric-box'>
                <small>RISK MITIGATION STATUS</small><br>
                <b style='font-size: 1.1rem; color:{"#EF4444" if is_high_risk else "#10B981"};'>
                    {"PRIORITY INTERVENTION" if is_high_risk else "STANDARD STANDBY"}
                </b>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------
# 4. GLOBAL MODEL INSIGHTS
# ----------------------------------------------------
st.write("---")
st.subheader("Global Pipeline Explanations & Parameter Sensitivity")

tab1, tab2 = st.tabs(["Global Feature Importance Mapping", "Strategic Simulation Matrix"])

with tab1:
    imp_viz_df = pd.DataFrame({
        "Structural Feature Variable": FEATURE_COLS,
        "Relative Model Importance": engine.feature_importances_,
    }).sort_values(by="Relative Model Importance", ascending=True)

    fig_imp = px.bar(
        imp_viz_df,
        x="Relative Model Importance",
        y="Structural Feature Variable",
        orientation="h",
        title="Random Forest Node Split Impurity Importances (Global System View)",
        color="Relative Model Importance",
        color_continuous_scale="blues",
    )
    fig_imp.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.markdown("##### Churn Probability Response Curve across Age Variants")

    sample_ages = np.linspace(18, 85, 25)
    sim_df = pd.concat([raw_vector] * len(sample_ages), ignore_index=True)
    sim_df["Age"] = sample_ages
    simulated_risks = engine.predict_proba(transform(sim_df))[:, 1] * 100

    curve_data = pd.DataFrame({
        "Target Simulated Age": sample_ages,
        "Calculated Risk Vector (%)": simulated_risks,
    })

    fig_curve = px.line(
        curve_data,
        x="Target Simulated Age",
        y="Calculated Risk Vector (%)",
        title="Sensitivity Trace Line: Risk Trajectory as Age Adjusts (Holding Other Attributes Static)",
        markers=True,
        template="plotly_white",
    )
    fig_curve.update_traces(line=dict(color="#EF4444", width=3), marker=dict(size=7, color="#1E3A8A"))
    fig_curve.update_layout(
        height=380,
        yaxis_range=[0, 105],
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

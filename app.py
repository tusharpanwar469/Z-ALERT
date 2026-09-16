import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Z-ALERT | Survival Intelligence",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 255, 170, 0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(255, 60, 60, 0.08),
                transparent 25%
            ),
            #070b10;
        color: #f2f5f7;
    }

    /* Hide default Streamlit header */
    header {
        visibility: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #090e14;
        border-right: 1px solid rgba(0, 255, 170, 0.15);
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 4px;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 15px;
        color: #8b9aa8;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }

    /* Status badge */
    .status {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 20px;
        background: rgba(0, 255, 170, 0.10);
        border: 1px solid rgba(0, 255, 170, 0.35);
        color: #00ffaa;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* Cards */
    .info-card {
        background: rgba(15, 23, 32, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .card-label {
        color: #7f8c99;
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .card-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .danger {
        color: #ff5252;
    }

    .safe {
        color: #00ffaa;
    }

    .warning {
        color: #ffc857;
    }

    /* Section heading */
    .section-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Prediction box */
    .prediction-box {
        padding: 30px;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(0, 255, 170, 0.08),
                rgba(15, 23, 32, 0.95)
            );
        border: 1px solid rgba(0, 255, 170, 0.25);
        text-align: center;
    }

    .prediction-number {
        font-size: 64px;
        font-weight: 900;
        line-height: 1;
        margin: 15px 0;
    }

    .prediction-label {
        color: #9aa7b3;
        font-size: 13px;
        letter-spacing: 2px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/survival_data.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/survival_model.pkl")


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error("Unable to load the dataset or trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# FEATURE LIST
# ============================================================

features = [
    "age",
    "health_score",
    "injury_severity",
    "infection_exposure",
    "food_days",
    "water_days",
    "shelter_quality",
    "medical_supplies",
    "weapon_availability",
    "group_size",
    "distance_to_safe_zone",
    "mobility_score",
    "days_since_outbreak"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0;">
            <div style="font-size:45px;">☣️</div>
            <h2 style="margin:0;">Z-ALERT</h2>
            <p style="color:#7f8c99;">SURVIVAL INTELLIGENCE</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "COMMAND MODULE",
        [
            "Command Center",
            "Survival Predictor",
            "Survival Analytics",
            "3D Survivor Analysis"
        ]
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.success("MODEL ONLINE")
    st.success("DATASET ONLINE")

    st.caption("Offline simulation mode")
    st.caption("No external API connection")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">☣️ Z-ALERT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">SURVIVAL PREDICTION & INTELLIGENCE SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<span class="status">● SYSTEM OPERATIONAL</span>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        '<div class="section-title">COMMAND CENTER</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)
    survivors = int(df["survived"].sum())
    non_survivors = total_records - survivors
    survival_rate = survivors / total_records * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">SIMULATED SUBJECTS</div>
                <div class="card-value">{total_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">SURVIVORS</div>
                <div class="card-value safe">{survivors:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">NON-SURVIVORS</div>
                <div class="card-value danger">{non_survivors:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">SIMULATED SURVIVAL RATE</div>
                <div class="card-value warning">{survival_rate:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:

        st.markdown(
            '<div class="section-title">SURVIVAL DISTRIBUTION</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Status": ["Survived", "Did Not Survive"],
            "Count": [survivors, non_survivors]
        })

        fig = px.pie(
            chart_data,
            names="Status",
            values="Count",
            hole=0.65
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=True
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with col_right:

        st.markdown(
            '<div class="section-title">KEY SURVIVAL FACTORS</div>',
            unsafe_allow_html=True
        )

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=True
        )

        fig2 = px.bar(
            importance.tail(8),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=430
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )

    st.info(
        "Z-ALERT is a fictional educational machine-learning simulation. "
        "The dataset and predictions do not represent real-world biological "
        "or disaster forecasting."
    )


# ============================================================
# SURVIVAL PREDICTOR
# ============================================================

elif page == "Survival Predictor":

    st.markdown(
        '<div class="section-title">SURVIVAL PREDICTOR</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Configure the subject's condition and run the survival simulation."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 👤 SUBJECT CONDITION")

        age = st.slider(
            "Age",
            16,
            80,
            30
        )

        health_score = st.slider(
            "Health Score",
            20,
            100,
            75
        )

        injury_severity = st.slider(
            "Injury Severity",
            0,
            10,
            2
        )

        infection_exposure = st.slider(
            "Infection Exposure",
            0,
            100,
            20
        )

        mobility_score = st.slider(
            "Mobility Score",
            20,
            100,
            75
        )

        group_size = st.slider(
            "Group Size",
            1,
            10,
            4
        )

    with col2:

        st.markdown("### 🏕️ SURVIVAL RESOURCES")

        food_days = st.slider(
            "Food Supply (days)",
            0,
            30,
            15
        )

        water_days = st.slider(
            "Water Supply (days)",
            0,
            30,
            15
        )

        shelter_quality = st.slider(
            "Shelter Quality",
            0,
            100,
            70
        )

        medical_supplies = st.slider(
            "Medical Supplies",
            0,
            100,
            60
        )

        weapon_availability = st.slider(
            "Defense Resources",
            0,
            100,
            50
        )

        distance_to_safe_zone = st.slider(
            "Distance to Safe Zone",
            1,
            100,
            30
        )

        days_since_outbreak = st.slider(
            "Days Since Outbreak",
            1,
            60,
            10
        )

    st.divider()

    if st.button(
        "⚡ ANALYZE SURVIVAL",
        type="primary"
    ):

        input_data = pd.DataFrame([{
            "age": age,
            "health_score": health_score,
            "injury_severity": injury_severity,
            "infection_exposure": infection_exposure,
            "food_days": food_days,
            "water_days": water_days,
            "shelter_quality": shelter_quality,
            "medical_supplies": medical_supplies,
            "weapon_availability": weapon_availability,
            "group_size": group_size,
            "distance_to_safe_zone": distance_to_safe_zone,
            "mobility_score": mobility_score,
            "days_since_outbreak": days_since_outbreak
        }])

        probability = model.predict_proba(input_data)[0][1]

        prediction = model.predict(input_data)[0]

        survival_percentage = probability * 100

        # ==========================================
        # SURVIVAL OUTCOME
        # ==========================================

        st.markdown("## 🎯 SURVIVAL ANALYSIS RESULT")

        result_col, gauge_col = st.columns([1, 1.5])

        with result_col:

            if survival_percentage >= 70:
                risk_level = "LOW RISK"
                risk_icon = "🟢"
                outcome = "HIGH SURVIVAL POTENTIAL"

            elif survival_percentage >= 40:
                risk_level = "MODERATE RISK"
                risk_icon = "🟡"
                outcome = "SURVIVAL CONDITIONS UNCERTAIN"

            else:
                risk_level = "HIGH RISK"
                risk_icon = "🔴"
                outcome = "LOW SURVIVAL POTENTIAL"

            st.metric(
                "SURVIVAL PROBABILITY",
                f"{survival_percentage:.1f}%"
            )

            st.markdown(
                f"### {risk_icon} {risk_level}"
            )

            st.write(outcome)

            if prediction == 1:
                st.success(
                    "✓ MODEL OUTCOME: MORE LIKELY TO SURVIVE"
                )
            else:
                st.error(
                    "⚠ MODEL OUTCOME: HIGHER SURVIVAL RISK"
                )

        with gauge_col:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=survival_percentage,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 42
                        }
                    },
                    title={
                        "text": "SURVIVAL PROBABILITY"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "thickness": 0.25
                        },
                        "steps": [
                            {
                                "range": [0, 40]
                            },
                            {
                                "range": [40, 70]
                            },
                            {
                                "range": [70, 100]
                            }
                        ],
                        "threshold": {
                            "line": {
                                "width": 5
                            },
                            "thickness": 0.8,
                            "value": survival_percentage
                        }
                    }
                )
            )

            gauge.update_layout(
                height=350,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(
                gauge,
                width="stretch"
            )

        st.markdown("---")

        st.markdown("### 🧬 SURVIVAL FACTOR STATUS")

        factor1, factor2, factor3, factor4 = st.columns(4)

        with factor1:
            st.metric(
                "❤️ HEALTH",
                f"{health_score}/100"
            )

        with factor2:
            st.metric(
                "🍖 FOOD",
                f"{food_days} days"
            )

        with factor3:
            st.metric(
                "💧 WATER",
                f"{water_days} days"
            )

        with factor4:
            st.metric(
                "🏃 MOBILITY",
                f"{mobility_score}/100"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        

        st.markdown("### 📋 INPUT SUMMARY")

        summary = pd.DataFrame({
            "Factor": [
                "Health",
                "Food",
                "Water",
                "Shelter",
                "Medical",
                "Mobility"
            ],
            "Value": [
                health_score,
                food_days,
                water_days,
                shelter_quality,
                medical_supplies,
                mobility_score
            ]
        })

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True
        )


# ============================================================
# SURVIVAL ANALYTICS
# ============================================================

elif page == "Survival Analytics":

    st.markdown(
        '<div class="section-title">SURVIVAL ANALYTICS</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="age",
            color="survived",
            nbins=25,
            title="Age Distribution"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with col2:

        fig = px.scatter(
            df.sample(min(1200, len(df)), random_state=42),
            x="health_score",
            y="food_days",
            color="survived",
            size="water_days",
            hover_data=[
                "age",
                "shelter_quality",
                "mobility_score"
            ],
            title="Health vs Food vs Water"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.markdown("### 📊 RESOURCE IMPACT")

    resource_columns = [
        "food_days",
        "water_days",
        "shelter_quality",
        "medical_supplies",
        "mobility_score"
    ]

    resource_data = (
        df.groupby("survived")[resource_columns]
        .mean()
        .T
    )

    resource_data.columns = [
        "Did Not Survive",
        "Survived"
    ]

    st.dataframe(
        resource_data.round(2),
        width="stretch"
    )

    st.markdown("### 🔎 DATASET EXPLORER")

    st.dataframe(
        df.head(100),
        width="stretch",
        height=400
    )


# ============================================================
# 3D SURVIVOR ANALYSIS
# ============================================================

elif page == "3D Survivor Analysis":

    st.markdown(
        '<div class="section-title">3D SURVIVOR ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Rotate, zoom and explore the simulated survivor population."
    )

    plot_df = df.sample(
        min(1500, len(df)),
        random_state=42
    ).copy()

    plot_df["Status"] = plot_df["survived"].map({
        0: "Did Not Survive",
        1: "Survived"
    })

    fig = px.scatter_3d(
        plot_df,
        x="health_score",
        y="food_days",
        z="water_days",
        color="Status",
        size="shelter_quality",
        hover_data=[
            "age",
            "injury_severity",
            "mobility_score",
            "distance_to_safe_zone"
        ],
        title="3D Survival Landscape"
    )

    fig.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis_title="Health Score",
            yaxis_title="Food Supply",
            zaxis_title="Water Supply"
        ),
        font=dict(color="white")
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.info(
        "Each point represents a simulated subject. "
        "Point size represents shelter quality."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#596773;
        padding:20px;
        font-size:12px;
    ">
        Z-ALERT • FICTIONAL SURVIVAL INTELLIGENCE SYSTEM •
        MACHINE LEARNING DEMONSTRATION
    </div>
    """,
    unsafe_allow_html=True
)
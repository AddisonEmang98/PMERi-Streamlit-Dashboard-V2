import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# =====================================
# RISK THRESHOLD FUNCTION
# =====================================
def get_pmeri_category(score):
    if score < 0.45:
        return "Low Risk"
    elif score < 0.60:
        return "Moderate Risk"
    else:
        return "High Risk"


# =====================================
# NORMALIZATION FUNCTION
# =====================================
def normalize(value, vmin, vmax):
    norm = (value - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, norm))


# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="PMERi DSS",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ PMERi Dashboard")
st.caption("Predictive Model for Environmental Risk Index")


# =====================================
# LOAD REQUIRED FILES
# =====================================
REGRESSOR_FILE = "PMERi_RF_Regressor.pkl"
METRICS_FILE = "model_metrics.pkl"
DATA_FILE = "Corrected_PMERi_Data.csv"
LOG_FILE = "predictions_log.csv"

missing_files = []

for file in [REGRESSOR_FILE, DATA_FILE]:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error(f"Missing required file(s): {', '.join(missing_files)}")
    st.stop()

reg_model = joblib.load(REGRESSOR_FILE)
df = pd.read_csv(DATA_FILE)

metrics = None
if os.path.exists(METRICS_FILE):
    metrics = joblib.load(METRICS_FILE)


# =====================================
# MODEL PERFORMANCE SECTION
# =====================================
st.header("Model Performance Matrix")

if metrics is not None:
    st.markdown("""
        <style>
        .thesis-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 30px;
            font-size: 16px;
        }

        .thesis-table th {
            background-color: #0f172a;
            color: #f8fafc;
            font-weight: bold;
            padding: 12px;
            text-align: center;
            border: 2px solid #3b82f6;
        }

        .thesis-table td {
            padding: 12px;
            border: 2px solid #3b82f6;
            color: inherit;
            text-align: center;
            vertical-align: middle;
        }

        .thesis-table tr:nth-child(even) {
            background-color: rgba(59, 130, 246, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    mse_value = metrics["regressor_rmse"] ** 2

    html_metrics_table = f"""
    <table class="thesis-table">
        <thead>
            <tr>
                <th>Model Architecture Type</th>
                <th>Evaluation Parameter</th>
                <th>Statistical Output Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="6"><b>Regression Framework</b><br>Continuous PMERi Score Predictor</td>
                <td>R² Score</td>
                <td>{metrics["regressor_r2"]:.3f}</td>
            </tr>
            <tr>
                <td>Mean Squared Error</td>
                <td>{mse_value:.5f}</td>
            </tr>
            <tr>
                <td>Mean Absolute Error</td>
                <td>{metrics["regressor_mae"]:.4f}</td>
            </tr>
            <tr>
                <td>Root Mean Squared Error</td>
                <td>{metrics["regressor_rmse"]:.4f}</td>
            </tr>
            <tr>
                <td>Cross-Validation Mean R²</td>
                <td>{metrics["regressor_cv_mean"]:.3f}</td>
            </tr>
            <tr>
                <td>Cross-Validation Standard Deviation</td>
                <td>{metrics["regressor_cv_std"]:.3f}</td>
            </tr>
        </tbody>
    </table>
    """

    st.markdown(html_metrics_table, unsafe_allow_html=True)

    # =====================================
    # CV BREAKDOWN TABLE
    # =====================================
    st.subheader("5-Fold Cross-Validation Breakdown")

    regressor_cv_folds = metrics.get("regressor_cv_folds", [])

    if len(regressor_cv_folds) == 5:
        html_cv_table = f"""
        <table class="thesis-table">
            <thead>
                <tr>
                    <th>Fold</th>
                    <th>Regressor R²</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Fold 1</td>
                    <td>{regressor_cv_folds[0]:.3f}</td>
                </tr>
                <tr>
                    <td>Fold 2</td>
                    <td>{regressor_cv_folds[1]:.3f}</td>
                </tr>
                <tr>
                    <td>Fold 3</td>
                    <td>{regressor_cv_folds[2]:.3f}</td>
                </tr>
                <tr>
                    <td>Fold 4</td>
                    <td>{regressor_cv_folds[3]:.3f}</td>
                </tr>
                <tr>
                    <td>Fold 5</td>
                    <td>{regressor_cv_folds[4]:.3f}</td>
                </tr>
            </tbody>
        </table>
        """

        st.markdown(html_cv_table, unsafe_allow_html=True)

# =====================================
# INPUT SECTION
# =====================================
st.header("Real-Time Environmental Input")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 300.0, 80.0)
    co2 = st.number_input("CO₂ (ppm)", 0.0, 5000.0, 600.0)
    hcho = st.number_input("HCHO (ppm)", 0.0, 0.3, 0.05)

with col2:
    temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    noise = st.number_input("Noise (dBA)", 0.0, 120.0, 70.0)
    lighting = st.number_input("Lighting (lux)", 0.0, 1000.0, 300.0)


# =====================================
# PREDICTION SECTION
# =====================================
if st.button("Run PMERi Analysis"):

    aqi = (
        normalize(pm25, 15, 75) +
        normalize(pm10, 15, 70) +
        normalize(co2, 0, 1000) +
        normalize(hcho, 0, 0.1)
    ) / 4

    input_data = pd.DataFrame([{
        "Air Quality Index (AQI)": aqi,
        "Temperature (C)": normalize(temp, 23, 32.5),
        "Humidity (%)": normalize(humidity, 40, 70),
        "Noise (dBA)": normalize(noise, 0, 85),
        "Lighting (LUX)": normalize(lighting, 200, 750)
    }])

    # =====================================
    # REGRESSOR PREDICTION
    # =====================================
    pmeri_score = reg_model.predict(input_data)[0]
    pmeri_score = max(0.0, min(1.0, pmeri_score))
    pmeri_category = get_pmeri_category(pmeri_score)

    # =====================================
    # ALERT SYSTEM
    # =====================================
    if pmeri_category == "High Risk":
        st.error("🚨 HIGH RISK ALERT DETECTED — IMMEDIATE ACTION REQUIRED")

        st.markdown(
            """
            <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
            </audio>
            """,
            unsafe_allow_html=True
        )

    elif pmeri_category == "Moderate Risk":
        st.warning("⚠️ MODERATE RISK — MONITOR CONDITIONS CLOSELY")

    else:
        st.success("🟢 SAFE CONDITIONS — NO IMMEDIATE RISK")

    # =====================================
    # LIVE RISK MONITORING
    # =====================================
    st.header("Live Risk Monitoring")

    colA, colB, colC = st.columns(3)

    colA.metric("Predicted PMERi Score", f"{pmeri_score:.3f}")
    colB.metric("PMERi Category", pmeri_category)
    colC.metric("Model Type", "Random Forest Regressor")

    # =====================================
    # RISK GAUGE
    # =====================================
    st.subheader("Risk Gauge")

    gauge_value = int(pmeri_score * 100)
    st.progress(gauge_value)

    if pmeri_category == "Low Risk":
        st.success(f"🟢 LOW RISK ({gauge_value:.1f}%)")
        st.write("Safe operating conditions. No immediate intervention required.")

    elif pmeri_category == "Moderate Risk":
        st.warning(f"🟡 MODERATE RISK ({gauge_value:.1f}%)")
        st.write("Monitor environmental conditions. Preventive action is recommended.")

    else:
        st.error(f"🔴 HIGH RISK ({gauge_value:.1f}%)")
        st.write("Critical environmental condition detected. Immediate action is required.")

    # =====================================
    # NORMALIZED INPUT TABLE
    # =====================================
    st.subheader("Normalized Input Values")

    st.dataframe(
        input_data.T.rename(columns={0: "Normalized Value"}).style.format("{:.3f}"),
        use_container_width=True
    )

    # =====================================
    # REGRESSOR FEATURE IMPORTANCE
    # =====================================
    st.subheader("Regressor Feature Importance")

    reg_importance = pd.Series(
        reg_model.feature_importances_,
        index=input_data.columns
    ).sort_values(ascending=False)

    st.bar_chart(reg_importance)

    # =====================================
    # RISK DISTRIBUTION
    # =====================================
    st.markdown("---")
    st.subheader("Dataset Risk Distribution")

    risk_counts = df["Risk Label"].value_counts()

    chart_df = pd.DataFrame([
        {"Category": "Low Risk", "Count": int(risk_counts.get(0, 0))},
        {"Category": "Moderate Risk", "Count": int(risk_counts.get(1, 0))},
        {"Category": "High Risk", "Count": int(risk_counts.get(2, 0))}
    ])

    fig_pie = px.pie(
        chart_df,
        values="Count",
        names="Category",
        color="Category",
        color_discrete_map={
            "Low Risk": "#2ecc71",
            "Moderate Risk": "#f1c40f",
            "High Risk": "#e74c3c"
        },
        category_orders={
            "Category": ["Low Risk", "Moderate Risk", "High Risk"]
        },
        hole=0.35,
        title="Baseline Risk Label Distribution"
    )

    fig_pie.update_traces(
        textinfo="percent+value",
        textfont_size=12
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # =====================================
    # CORRELATION ANALYSIS
    # =====================================
    st.header("Correlation Analysis")

    env_variables = [
        "Air Quality Index (AQI)",
        "Temperature (C)",
        "Humidity (%)",
        "Noise (dBA)",
        "Lighting (LUX)"
    ]

    st.subheader("A. Environmental Correlation Matrix")

    env_corr = df[env_variables].corr()

    fig_corr = px.imshow(
        env_corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Between Environmental Stressors"
    )

    fig_corr.update_layout(
        height=650,
        xaxis_title="Environmental Variables",
        yaxis_title="Environmental Variables"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("B. Pearson Correlation vs PMERi Score")

    pmeri_corr = (
        df[
            env_variables + ["PMERi Score"]
        ]
        .corr()["PMERi Score"]
        .drop("PMERi Score")
        .sort_values(ascending=False)
    )

    corr_df = pd.DataFrame({
        "Environmental Variable": pmeri_corr.index,
        "Correlation": pmeri_corr.values
    })

    fig_pmeri_corr = px.bar(
        corr_df,
        x="Environmental Variable",
        y="Correlation",
        text="Correlation",
        title="Pearson Correlation Between Environmental Variables and PMERi Score"
    )

    fig_pmeri_corr.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_pmeri_corr.update_layout(
        height=500,
        yaxis_title="Pearson Correlation Coefficient",
        xaxis_title="Environmental Variable"
    )

    st.plotly_chart(fig_pmeri_corr, use_container_width=True)

    # =====================================
    # LOGGING
    # =====================================
    log = pd.DataFrame([{
        "Time": datetime.now(),
        "PM2.5": pm25,
        "PM10": pm10,
        "CO2": co2,
        "HCHO": hcho,
        "Temperature": temp,
        "Humidity": humidity,
        "Noise": noise,
        "Lighting": lighting,
        "AQI": aqi,
        "Predicted PMERi Score": pmeri_score,
        "PMERi Category": pmeri_category,
        "Model Type": "Random Forest Regressor"
    }])

    if os.path.exists(LOG_FILE):
        log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log.to_csv(LOG_FILE, index=False)

    st.success("Prediction logged successfully.")


# =====================================
# ACADEMIC FOOTER
# =====================================
st.markdown("---")
st.write(
    "Developed as part of the requirement for Bachelor of Mechanical Engineering (Honours) at Universiti Malaysia Sarawak (UNIMAS)."
)
st.write("Supervisor: Ts. Mohd. Azrin bin Mohd. Said")
st.write("FYP Student: Addison Ding Emang (83044)")
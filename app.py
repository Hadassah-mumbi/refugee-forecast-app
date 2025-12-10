import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="Refugee Movement Forecast", layout="wide")
st.title("🌍 Refugee Movement Forecasting Dashboard")

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# ============================
# SIDEBAR INPUTS
# ============================
st.sidebar.header("🔧 Prediction Parameters")

countries = ["Kenya","Uganda","Tanzania","Ethiopia","South Sudan","Eritrea","Djibouti","Somalia","Burundi","Rwanda","Sudan"]
origin_country = st.sidebar.selectbox("Country of Origin", countries)
asylum_country = st.sidebar.selectbox("Host Country", countries)

year_range = st.sidebar.slider("Prediction Year Range", 2025, 2035, (2026, 2030))
weather = st.sidebar.selectbox("Weather Condition", ["Normal","Drought","Flooding"])
conflict_types = st.sidebar.multiselect("Conflict Types", ["Civil War","Political Violence","Terrorism","Ethnic Clashes"])
intensity = st.sidebar.slider("Conflict Intensity (1–10)", 1, 10)

predict_btn = st.sidebar.button("🔮 Run Prediction")

# ============================
# CONVERT INPUTS
# ============================
weather_map = {"Normal":0.3, "Drought":0.7, "Flooding":0.8}
weather_value = weather_map[weather]
conflict_count = max(1, len(conflict_types))  # At least 1 type selected

# ============================
# PREDICTION
# ============================
if predict_btn:
    total_prediction = 0
    yearly_results = []

    for year in range(year_range[0], year_range[1]+1):
        features = np.array([[conflict_count, intensity, weather_value, year]])
        pred = model.predict(features)[0]
        pred = max(0, int(pred))
        yearly_results.append(pred)
        total_prediction += pred

    # ============================
    # RESULTS DISPLAY
    # ============================
    st.subheader("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Origin Country", origin_country)
    with col2:
        st.metric("Host Country", asylum_country)
    with col3:
        st.metric("Total Expected Refugees", f"{total_prediction:,}")

    # ============================
    # CAMPS & COST
    # ============================
    camp_capacity = 20000
    camps_needed = (total_prediction // camp_capacity) + (1 if total_prediction % camp_capacity > 0 else 0)
    cost_per_refugee = 350
    total_cost = total_prediction * cost_per_refugee

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Estimated Camps Needed", f"{camps_needed}")
    with col5:
        st.metric("Estimated Hosting Cost (USD)", f"${total_cost:,.0f}")

    # ============================
    # YEARLY TREND GRAPH
    # ============================
    trend_df = pd.DataFrame({"Year":list(range(year_range[0], year_range[1]+1)),
                             "Predicted Refugees": yearly_results})
    fig = px.line(trend_df, x="Year", y="Predicted Refugees", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # MIGRATION MAP (SIMULATED)
    # ============================
    st.subheader("🗺 Migration Flow Visualization")
    country_coords = {
        "Kenya":(0.0236,37.9062),
        "Uganda":(1.3733,32.2903),
        "Tanzania":(-6.3690,34.8888),
        "Ethiopia":(9.1450,40.4897),
        "South Sudan":(7.8698,29.6668),
        "Eritrea":(15.1794,39.7823),
        "Djibouti":(11.8251,42.5903),
        "Somalia":(5.1521,46.1996),
        "Burundi":(-3.3731,29.9189),
        "Rwanda":(-1.9403,29.8739),
        "Sudan":(12.8628,30.2176)
    }

    map_df = pd.DataFrame({
        "country":[origin_country, asylum_country],
        "lat":[country_coords[origin_country][0], country_coords[asylum_country][0]],
        "lon":[country_coords[origin_country][1], country_coords[asylum_country][1]],
        "type":["Origin","Destination"]
    })

    fig2 = px.scatter_geo(map_df, lat="lat", lon="lon", color="type",
                          hover_name="country", size=[10,20],
                          projection="natural earth")
    st.plotly_chart(fig2, use_container_width=True)

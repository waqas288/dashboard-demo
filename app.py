# =========================================
# Schistosomiasis Risk Forecasting Dashboard
# Interactive MVP with per-subcounty variability
# =========================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Schisto Risk Dashboard", layout="wide")
st.title("Schistosomiasis Risk Forecasting Dashboard")
st.caption("3-Month Ahead Climate-Based Risk Classification (Synthetic MVP Demo)")

# ----------------------------
# 1. SYNTHETIC DATA GENERATION
# ----------------------------
np.random.seed(42)

subcounties = [f"SubCounty_{i}" for i in range(1, 16)]
years = np.arange(2012, 2023)

rows = []

# Assign base climate differences per subcounty
subcounty_offsets = {sc: {"rainfall_base": np.random.uniform(80,160),
                          "temp_base": np.random.uniform(23,30)}
                     for sc in subcounties}

for sc in subcounties:
    r_base = subcounty_offsets[sc]["rainfall_base"]
    t_base = subcounty_offsets[sc]["temp_base"]
    for year in years:
        for month in range(1,13):
            # Seasonal effect + subcounty base + small noise
            rainfall = np.random.normal(r_base + 40*np.sin(month/12*2*np.pi), 15)
            temp = np.random.normal(t_base + 3*np.sin(month/12*2*np.pi), 1.2)
            rows.append({"subcounty": sc,
                         "year": year,
                         "month": month,
                         "rainfall": max(rainfall,0),
                         "temperature": temp})

climate_df = pd.DataFrame(rows)

# 3-month rolling averages
climate_df["rainfall_3mo"] = climate_df.groupby("subcounty")["rainfall"].rolling(3,min_periods=1).mean().reset_index(drop=True)
climate_df["temp_3mo"] = climate_df.groupby("subcounty")["temperature"].rolling(3,min_periods=1).mean().reset_index(drop=True)

# Yearly aggregation
yearly = climate_df.groupby(["subcounty","year"]).agg({
    "rainfall_3mo":"mean",
    "temp_3mo":"mean"
}).reset_index()

# ----------------------------
# 2. PREVALENCE AND RISK
# ----------------------------
# Add subcounty-specific prevalence bias
subcounty_prev_bias = {sc: np.random.uniform(-0.05,0.05) for sc in subcounties}

yearly["prevalence"] = (
    0.15
    + 0.002*yearly["rainfall_3mo"]
    + 0.025*yearly["temp_3mo"]
    + yearly["subcounty"].map(subcounty_prev_bias)
    + np.random.normal(0,0.03,len(yearly))
).clip(0,1)

# Percentile-based risk classification (guarantees multiple classes)
low_thr = yearly["prevalence"].quantile(0.33)
high_thr = yearly["prevalence"].quantile(0.66)

def classify(p):
    if p <= low_thr: return "Low"
    elif p <= high_thr: return "Medium"
    else: return "High"

yearly["risk"] = yearly["prevalence"].apply(classify)
risk_map = {"Low":0,"Medium":1,"High":2}
yearly["risk_encoded"] = yearly["risk"].map(risk_map)

# ----------------------------
# 3. MODEL TRAINING
# ----------------------------
features = ["rainfall_3mo","temp_3mo"]
X = yearly[features]
y = yearly["risk_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# ----------------------------
# 4. SIDEBAR FILTER
# ----------------------------
st.sidebar.header("Filters")
selected_sc = st.sidebar.selectbox("Select Subcounty", subcounties)
sc_data = yearly[yearly["subcounty"]==selected_sc].sort_values("year")

# ----------------------------
# 5. TOP KPI CARDS
# ----------------------------
latest = sc_data.iloc[-1]
input_array = np.array([[latest["rainfall_3mo"], latest["temp_3mo"]]])
pred = model.predict(scaler.transform(input_array))[0]
inverse_map = {0:"Low",1:"Medium",2:"High"}
predicted_risk = inverse_map[pred]

col1,col2,col3 = st.columns(3)
col1.metric("Latest Prevalence", f"{latest['prevalence']:.2f}")
col2.metric("Current Risk Class", latest["risk"])
col3.metric("3-Month Forecast", predicted_risk)

# ----------------------------
# 6. SECOND ROW: MULTIPLE GRAPHS SIDE-BY-SIDE
# ----------------------------
col4,col5,col6 = st.columns(3)

with col4:
    fig1 = px.line(sc_data,x="year",y="prevalence",markers=True,title="Prevalence Trend")
    st.plotly_chart(fig1,use_container_width=True)

with col5:
    fig2 = px.bar(sc_data,x="year",y="prevalence",color="risk",title="Yearly Risk Distribution")
    st.plotly_chart(fig2,use_container_width=True)

with col6:
    fig3 = px.line(sc_data,x="year",y="rainfall_3mo",title="3-Month Avg Rainfall")
    fig4 = px.line(sc_data,x="year",y="temp_3mo",title="3-Month Avg Temperature")
    st.plotly_chart(fig3,use_container_width=True)
    st.plotly_chart(fig4,use_container_width=True)

# ----------------------------
# 7. THIRD ROW: FEATURE IMPORTANCE + SUMMARY TABLE
# ----------------------------
col7,col8 = st.columns([1,1])

with col7:
    importance_df = pd.DataFrame({"Feature":features,"Importance":model.feature_importances_})
    fig5 = px.bar(importance_df,x="Feature",y="Importance",title="Model Feature Importance")
    st.plotly_chart(fig5,use_container_width=True)

with col8:
    st.subheader("Latest Year Risk Across All Subcounties")
    latest_year = yearly["year"].max()
    summary = yearly[yearly["year"]==latest_year][["subcounty","risk"]]
    st.dataframe(summary,use_container_width=True)

# ----------------------------
# 8. DOWNLOAD BUTTON
# ----------------------------
csv = yearly.to_csv(index=False).encode("utf-8")
st.download_button("Download Full Dataset",csv,"schisto_data.csv","text/csv")
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import BytesIO

st.set_page_config(page_title="Insurance Risk Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #f9f9f9; color: #333; font-family: 'Segoe UI', sans-serif;}
        .block-container {padding: 2rem;}
    </style>
""", unsafe_allow_html=True)

# --- Dataset Selector ---
st.sidebar.header("📂 Select Dataset")
data_choice = st.sidebar.selectbox("Choose Data Source:", ["Synthetic (500)", "Kaggle Real"])
if data_choice == "Synthetic (500)":
    df = pd.read_csv("synthetic_policy_data.csv")
else:
    df = pd.read_csv("kaggle_cleaned.csv")

# --- Dynamic Filters ---
st.sidebar.header("🧭 Filter Policies")
region_filter = st.sidebar.multiselect("Filter by Region", options=df["Region"].unique(), default=df["Region"].unique())
type_filter = st.sidebar.multiselect("Filter by Policy Type", options=df["Type"].unique(), default=df["Type"].unique())
df = df[(df["Region"].isin(region_filter)) & (df["Type"].isin(type_filter))]

st.title("📊 Insurance Portfolio Risk Management Dashboard")
st.caption(f"📌 Displaying **{len(df):,}** filtered policies.")

# --- Risk Score ---
if "RiskScore" not in df.columns:
    df["RiskScore"] = df["Coverage"] / df["Coverage"].max() * 100

# --- KPIs ---
region_risk = df.groupby("Region")["Coverage"].sum().reset_index()
alert_threshold = 0.3
total_coverage = df["Coverage"].sum()
region_risk["RiskPercent"] = region_risk["Coverage"] / total_coverage

col1, col2, col3 = st.columns(3)
col1.metric("Total Policies", len(df))
col2.metric("Total Coverage ₹", f"{total_coverage:,.0f}")
col3.metric("High Risk Alerts", (region_risk["RiskPercent"] > alert_threshold).sum())

# --- Region Bar Chart ---
st.subheader("📍 Risk Exposure by Region")
st.plotly_chart(px.bar(region_risk, x="Region", y="Coverage", text_auto=True, color_discrete_sequence=['#636EFA']))

# --- High Risk Alert ---
st.subheader("⚠️ Alert: High Concentration Regions")
for _, row in region_risk.iterrows():
    if row["RiskPercent"] > alert_threshold:
        st.error(f"Region **{row['Region']}** holds **{row['RiskPercent']*100:.1f}%** of total risk!")

# --- Correlation Matrix ---
st.subheader("📉 Correlation Matrix")
fig, ax = plt.subplots()
sns.heatmap(df[["Coverage", "Premium", "RiskScore"]].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --- Stress Simulation ---
st.subheader("🔥 Stress Testing Simulation")
selected_region = st.selectbox("Select Region to Simulate Catastrophe:", df["Region"].unique())
loss_percent = st.slider("Simulated Loss in Selected Region (%)", 0, 100, 50)
region_coverage = df[df["Region"] == selected_region]["Coverage"].sum()
estimated_loss = (loss_percent / 100) * region_coverage
portfolio_loss_percent = (estimated_loss / total_coverage) * 100
st.write(f"💥 Estimated Loss in **{selected_region}**: ₹{estimated_loss:,.0f}")
st.write(f"📉 This is **{portfolio_loss_percent:.2f}%** of the total portfolio exposure.")

# --- KMeans Clustering ---
st.subheader("🤖 Risk Segmentation using KMeans Clustering")
features = df[["Coverage", "Premium", "RiskScore"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df["RiskCluster"] = kmeans.fit_predict(features)
risk_order = df.groupby("RiskCluster")["RiskScore"].mean().sort_values().index
risk_labels = {cluster: label for cluster, label in zip(risk_order, ["Low Risk", "Medium Risk", "High Risk"])}
df["RiskLevel"] = df["RiskCluster"].map(risk_labels)
st.bar_chart(df["RiskLevel"].value_counts())

# --- Claim Risk Predictor ---
st.subheader("🔮 Claim Risk Predictor (ML Model)")
if "Claimed" in df.columns:
    df_enc = df.copy()
    df_enc["RegionEncoded"] = LabelEncoder().fit_transform(df["Region"])
    df_enc["TypeEncoded"] = LabelEncoder().fit_transform(df["Type"])
    X = df_enc[["Coverage", "Premium", "RiskScore", "RegionEncoded", "TypeEncoded"]]
    y = df_enc["Claimed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    df_enc["ClaimProbability"] = model.predict_proba(X)[:, 1] * 100
    st.metric("🎯 Model Accuracy", f"{accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

    # --- Claim Distribution ---
    st.subheader("🔍 Claim Distribution")
    st.markdown("✅ **'Claimed' Value Counts:**")
    claim_counts = df_enc["Claimed"].value_counts().reset_index()
    claim_counts.columns = ["Claimed", "Count"]
    st.dataframe(claim_counts)

    st.dataframe(df_enc[df_enc["ClaimProbability"] > 70][["PolicyID", "Region", "RiskScore", "ClaimProbability"]])
else:
    st.warning("🟡 'Claimed' column not found. ML prediction not available.")

# --- Export ---
st.subheader("📥 Export Data")
if st.button("Download as Excel"):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button("📥 Click to Download Excel", data=buffer.getvalue(), file_name="dashboard_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Raw Data ---
st.subheader("🗂️ Full Policy Data Table")
st.dataframe(df)

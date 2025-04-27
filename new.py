import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Read data from CSV file
df = pd.read_csv("synthetic_sensor_data.csv")

# Ensure you have the correct number of rows for n_samples
n_samples = len(df)

# Train-test split
X = df.drop(columns=['failure'])
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# KPIs
accuracy = accuracy_score(y_test, y_pred)
failure_rate = df['failure'].mean() * 100
feature_importance = model.feature_importances_
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Sidebar Input
st.sidebar.header("🔍 Predict Failure (Try Inputs)")
input_temp = st.sidebar.slider("Temperature", 50, 200, 70)
input_vib = st.sidebar.slider("Vibration", 1.0, 20.0, 5.0)
input_pres = st.sidebar.slider("Pressure", 60, 200, 100)
input_hum = st.sidebar.slider("Humidity", 10, 150, 40)

# Live Prediction
user_input = np.array([[input_temp, input_vib, input_pres, input_hum]])
prediction = model.predict(user_input)[0]
prediction_prob = model.predict_proba(user_input)[0][1]

# Live Sensor Status Indicator
def sensor_status(val, threshold):
    return " 🟢 OK" if val < threshold else " 🔴 High"

st.sidebar.markdown(f"**Temperature Status:** {sensor_status(input_temp, 90)}")
st.sidebar.markdown(f"**Vibration Status:** {sensor_status(input_vib, 7)}")
st.sidebar.markdown(f"**Pressure Status:** {sensor_status(input_pres, 120)}")
st.sidebar.markdown(f"**Humidity Status:** {sensor_status(input_hum, 60)}")

# Dashboard Title
st.title("🛠️ Predictive Maintenance KPI Dashboard")

# Top Metrics
col1, col2, col3 = st.columns(3)
col1.metric("🔍 Model Accuracy", f"{accuracy:.2%}")
col2.metric("⚠️ Failure Rate", f"{failure_rate:.2f}%")
col3.metric("🤖 Live Failure Prediction", "Failure" if prediction == 1 else "No Failure")

# Prediction Probability
st.sidebar.markdown(f"### ⚙️ Failure Probability: `{prediction_prob:.2%}`")

# Feature Importance
st.subheader("📊 Feature Importance")
fig_imp, ax_imp = plt.subplots()
sns.barplot(x=feature_importance, y=X.columns, ax=ax_imp)
st.pyplot(fig_imp)

# Confusion Matrix
with st.expander("🧾 Confusion Matrix & Report"):
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("Classification Report")
    st.json(report)

# Time Series View
with st.expander("📈 Simulated Sensor Monitoring (Time Series)"):
    df_time = df.copy()
    df_time['time'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    df_plot = df_time.set_index("time")
    st.line_chart(df_plot[['temperature', 'vibration', 'pressure', 'humidity']])

# Failure Trends Over Time
st.subheader("📅 Failure Trends Over Time")
fail_trend = df_time.set_index('time').resample('D').sum()['failure']
st.line_chart(fail_trend)

# Sensor Correlation Heatmap
st.subheader("🔗 Sensor Correlation Heatmap")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

# Data Distributions
st.subheader("📦 Sensor Data Distributions")
fig_dist, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(X.columns):
    sns.histplot(df[col], bins=30, ax=axes[i // 2, i % 2], kde=True)
    axes[i // 2, i % 2].set_title(f"{col} Distribution")
st.pyplot(fig_dist)

# Failure Count
st.subheader("💥 Failure Count")
fig_fail, ax_fail = plt.subplots()
sns.countplot(x=df['failure'], ax=ax_fail)
ax_fail.set_xticklabels(['No Failure', 'Failure'])
st.pyplot(fig_fail)

# Download Options
with st.expander("📁 Download Data & Report"):
    csv = df.to_csv(index=False)
    st.download_button("Download Dataset", csv, "sensor_data.csv", "text/csv")

    report_text = classification_report(y_test, y_pred)
    st.download_button("Download Report", report_text, "classification_report.txt", "text/plain")

# Raw Data Table
with st.expander("🧾 Raw Sensor Data Preview"):
    st.dataframe(df.head())


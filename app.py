import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")


# =========================================================
# CSS Styling (Professional UI)
# =========================================================
st.markdown("""
<style>
.big-title {
    background: linear-gradient(90deg,#1f4e79,#2e86c1);
    padding:22px;
    border-radius:15px;
    color:white;
    text-align:center;
    font-size:36px;
    font-weight:bold;
}
.card {
    background:#f5f6fa;
    padding:25px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.08);
}
.metric {
    font-size:32px;
    font-weight:bold;
    color:#1f4e79;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="big-title">💳 Credit Risk Prediction Dashboard</div>', unsafe_allow_html=True)
st.write("### Machine Learning Model using your Credit Dataset")


# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_csv("credit_risk_dataset.csv")


# =========================================================
# DATASET OVERVIEW
# =========================================================
with st.expander("📂 Click to View Dataset Preview"):
    st.dataframe(data.head(), use_container_width=True)
    st.write("Dataset Shape:", data.shape)

with st.expander("📊 Click to View Dataset Summary"):
    st.dataframe(data.describe(), use_container_width=True)


# =========================================================
# PREPROCESS
# =========================================================
def preprocess(df):
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders


df, encoders = preprocess(data)


# =========================================================
# 🔥 SAFE TARGET FIX (IMPORTANT)
# ensures ALWAYS binary
# =========================================================
target = df.iloc[:, -1]

if target.nunique() > 2:
    target = (target > target.median()).astype(int)

df.iloc[:, -1] = target


# =========================================================
# TRAIN MODEL
# =========================================================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds, labels=[0, 1])

low_risk = (preds == 0).sum()
high_risk = (preds == 1).sum()


# =========================================================
# PERFORMANCE CARDS
# =========================================================
st.subheader("📈 Model Performance Overview")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f'<div class="card">Accuracy<br><div class="metric">{accuracy:.4f}</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card">Low Risk (0)<br><div class="metric">{low_risk}</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card">High Risk (1)<br><div class="metric">{high_risk}</div></div>', unsafe_allow_html=True)


# =========================================================
# CONFUSION MATRIX HEATMAP
# =========================================================
st.subheader("🔍 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(cm)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Pred 0","Pred 1"])
ax.set_yticklabels(["Actual 0","Actual 1"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

st.pyplot(fig)


# =========================================================
# FEATURE IMPORTANCE (Top 10)
# =========================================================
st.subheader("⭐ Top 10 Important Features")

importances = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(10)

fig2, ax2 = plt.subplots()
ax2.barh(imp_df["Feature"], imp_df["Importance"])
ax2.invert_yaxis()

st.pyplot(fig2)


# =========================================================
# SIDEBAR PREDICTION
# =========================================================
st.sidebar.header("✍ Enter Customer Details")

inputs = {}

for col in data.columns[:-1]:
    if data[col].dtype == "object":
        inputs[col] = st.sidebar.selectbox(col, data[col].unique())
    else:
        inputs[col] = st.sidebar.number_input(
            col,
            float(data[col].min()),
            float(data[col].max()),
            float(data[col].mean())
        )

input_df = pd.DataFrame([inputs])


# encode
test = input_df.copy()
for col in test.columns:
    if col in encoders:
        test[col] = encoders[col].transform(test[col])


if st.sidebar.button("Predict Credit Risk"):
    pred = model.predict(test)[0]
    prob = model.predict_proba(test)[0][1]

    if pred == 1:
        st.sidebar.error(f"⚠ High Risk ({prob*100:.2f}%)")
    else:
        st.sidebar.success(f"✅ Low Risk ({(1-prob)*100:.2f}%)")
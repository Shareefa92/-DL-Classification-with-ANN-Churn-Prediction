import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="Churn Prediction", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(64, activation="relu", input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model, scaler, X.columns.tolist()

# --- Header ---
st.title("🏦 Customer Churn Prediction")
st.markdown("**By Sharifah Al-Dalbahi** | Deep Learning (ANN) Model")
st.markdown("---")

model, scaler, feature_names = load_model()

# --- Input ---
st.subheader("📋 Customer Information")
col1, col2 = st.columns(2)

with col1:
    credit_score     = st.slider("Credit Score", 300, 850, 600)
    age              = st.slider("Age", 18, 92, 35)
    tenure           = st.slider("Tenure (years)", 0, 10, 5)
    balance          = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
    num_products     = st.slider("Number of Products", 1, 4, 1)

with col2:
    geography        = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender           = st.selectbox("Gender", ["Male", "Female"])
    has_cr_card      = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active        = st.selectbox("Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)

st.markdown("---")

# --- Predict ---
if st.button("🔍 Predict Churn Risk", use_container_width=True):
    input_data = pd.DataFrame([[
        credit_score,
        1 if gender == "Male" else 0,
        age, tenure, balance, num_products,
        1 if has_cr_card == "Yes" else 0,
        1 if is_active == "Yes" else 0,
        estimated_salary,
        1 if geography == "Germany" else 0,
        1 if geography == "Spain" else 0
    ]], columns=feature_names)

    prediction = model.predict(scaler.transform(input_data))[0][0]

    st.markdown("### 🎯 Prediction Result")
    if prediction > 0.5:
        st.error(f"🔴 **High Risk of Churn: {prediction:.1%}**")
        st.markdown("⚠️ This customer is likely to leave. Consider proactive retention actions.")
    else:
        st.success(f"🟢 **Low Risk of Churn: {prediction:.1%}**")
        st.markdown("✅ This customer is likely to stay.")

    st.progress(float(prediction))
    st.caption(f"Churn probability: {prediction:.4f}")

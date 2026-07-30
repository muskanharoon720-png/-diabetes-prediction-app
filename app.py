"""
app.py
======
Intelligent AI-Powered Diabetes Prediction & Analytics System
Streamlit Dashboard - single file app.

Folder expectations (place these next to app.py):
    dataset/diabetes_health_indicators_50k.csv
    saved_models/Logistic_Regression.pkl
    saved_models/Decision_Tree.pkl
    saved_models/Random_Forest.pkl
    saved_models/KNN.pkl
    saved_models/SVM.pkl
    saved_models/XGBoost.pkl
    saved_models/Gradient_Boosting.pkl
    saved_models/scaler.pkl
    saved_models/ANN_model.h5
    saved_models/LSTM_model.h5
    model_comparison.csv   (from Colab Cell 52)

Run with:
    streamlit run app.py
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
)

# --------------------------------------------------------------------------- #
# PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
)

import glob

def _find_dataset_path():
    # Look in dataset/ first, then root, matching any csv that starts with
    # "diabetes_health_indicators" (handles renamed/duplicate-suffixed files)
    candidates = (
        glob.glob("dataset/diabetes_health_indicators*.csv")
        + glob.glob("diabetes_health_indicators*.csv")
    )
    return candidates[0] if candidates else "dataset/diabetes_health_indicators_50k.csv"

DATA_PATH = _find_dataset_path()
MODELS_DIR = "saved_models" if os.path.isdir("saved_models") else "."
COMPARISON_CSV = "model_comparison.csv"

FEATURE_COLS_DEFAULT = [
    "HighBP", "HighChol", "BMI", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "GenHlth", "DiffWalk", "Age", "Income",
]  # fallback if feature_cols not saved separately - update to match your
   # actual result["feature_cols"] from preprocessing if different


# --------------------------------------------------------------------------- #
# CACHED LOADERS
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_ml_models():
    models = {}
    if not os.path.isdir(MODELS_DIR):
        return models
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".pkl") and fname != "scaler.pkl":
            name = fname.replace(".pkl", "").replace("_", " ")
            try:
                models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
            except Exception:
                pass
    return models


@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_dl_models():
    dl_models = {}
    try:
        from tensorflow.keras.models import load_model
        ann_path = os.path.join(MODELS_DIR, "ANN_model.h5")
        lstm_path = os.path.join(MODELS_DIR, "LSTM_model.h5")
        if os.path.exists(ann_path):
            dl_models["ANN"] = load_model(ann_path)
        if os.path.exists(lstm_path):
            dl_models["LSTM"] = load_model(lstm_path)
    except Exception as e:
        st.session_state["dl_load_error"] = str(e)
    return dl_models


@st.cache_data
def load_comparison_table():
    if os.path.exists(COMPARISON_CSV):
        return pd.read_csv(COMPARISON_CSV, index_col=0)
    return None


# --------------------------------------------------------------------------- #
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Viewer",
        "📈 EDA Dashboard",
        "🤖 Model Comparison",
        "🔮 Prediction",
        "💬 AI Assistant",
    ],
)

df = load_dataset()
ml_models = load_ml_models()
scaler = load_scaler()
comparison_df = load_comparison_table()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {'✅ Loaded' if df is not None else '❌ Not found'}\n\n"
    f"ML Models: {len(ml_models)} loaded\n\n"
    f"Scaler: {'✅' if scaler is not None else '❌'}"
)

# --------------------------------------------------------------------------- #
# HOME PAGE
# --------------------------------------------------------------------------- #
if page == "🏠 Home":
    st.title("🩺 Intelligent AI-Powered Diabetes Prediction & Analytics System")
    st.markdown(
        """
        Welcome! This application uses **Machine Learning** and **Deep Learning**
        to predict diabetes risk from health indicator data, with full
        analytics, model comparison, and an AI assistant.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    if df is not None:
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", f"{df.shape[1] - 1}")
        col3.metric("Diabetes Cases", f"{int(df['Diabetes_binary'].sum()):,}")
        col4.metric("Diabetes Rate", f"{df['Diabetes_binary'].mean()*100:.1f}%")
    else:
        st.warning("Dataset not found. Place the CSV inside the `dataset/` folder.")

    st.markdown("### Project Highlights")
    st.markdown(
        """
        - ✅ 50,000+ records, 21 health indicator features
        - ✅ Full preprocessing pipeline (missing values, outliers, encoding, scaling)
        - ✅ 7 Machine Learning models + 2 Deep Learning models
        - ✅ Complete evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
        - ✅ Interactive EDA and real-time prediction
        - ✅ AI Assistant for natural-language questions about the data/model
        """
    )

# --------------------------------------------------------------------------- #
# DATASET VIEWER
# --------------------------------------------------------------------------- #
elif page == "📊 Dataset Viewer":
    st.title("📊 Dataset Viewer")
    if df is None:
        st.error("Dataset not found. Place the CSV inside the `dataset/` folder.")
    else:
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        tab1, tab2, tab3 = st.tabs(["Preview", "Data Types", "Summary Stats"])
        with tab1:
            n_rows = st.slider("Rows to preview", 5, 100, 10)
            st.dataframe(df.head(n_rows), use_container_width=True)
        with tab2:
            st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)
        with tab3:
            st.dataframe(df.describe(), use_container_width=True)

        st.download_button(
            "⬇️ Download full dataset (CSV)",
            df.to_csv(index=False).encode(),
            "diabetes_health_indicators_50k.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- #
# EDA DASHBOARD
# --------------------------------------------------------------------------- #
elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")
    if df is None:
        st.error("Dataset not found.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Target Distribution")
            fig = px.pie(df, names=df["Diabetes_binary"].map({0: "No Diabetes", 1: "Diabetes"}),
                         title="Diabetes vs No Diabetes")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("BMI Distribution")
            fig = px.histogram(df, x="BMI", color="Diabetes_binary", nbins=40,
                                title="BMI Distribution by Diabetes Status")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Diabetes Rate by Age Group")
        age_diab = df.groupby("Age")["Diabetes_binary"].mean().reset_index()
        fig = px.bar(age_diab, x="Age", y="Diabetes_binary",
                     title="Diabetes Rate by Age Group")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Correlation with Target")
        corr_target = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary").sort_values()
        fig = px.bar(corr_target, orientation="h", title="Correlation with Diabetes_binary")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# MODEL COMPARISON
# --------------------------------------------------------------------------- #
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison & Performance")

    if comparison_df is None:
        st.warning(
            "`model_comparison.csv` not found. Place it next to app.py "
            "(generated in Colab Cell 52)."
        )
    else:
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        metric = st.selectbox(
            "Compare models by:",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Training Time"],
        )
        fig = px.bar(
            comparison_df.sort_values(metric, ascending=False),
            y=metric, title=f"Model Comparison - {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

        best_model = comparison_df["Accuracy"].idxmax()
        st.success(f"🏆 Best performing model: **{best_model}** "
                   f"(Accuracy: {comparison_df.loc[best_model, 'Accuracy']:.4f})")

# --------------------------------------------------------------------------- #
# PREDICTION PAGE
# --------------------------------------------------------------------------- #
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Risk Prediction")

    if not ml_models:
        st.error(
            "No trained models found. Place `.pkl` files inside "
            "`saved_models/` (from Colab Cell 56)."
        )
    else:
        model_name = st.selectbox("Choose a model", list(ml_models.keys()))
        model = ml_models[model_name]

        tab1, tab2 = st.tabs(["🧍 Single Prediction (Real-time)", "📁 Batch Prediction (CSV Upload)"])

        # ---- Single real-time prediction ----
        with tab1:
            st.write("Enter patient health indicators:")
            c1, c2, c3 = st.columns(3)
            with c1:
                high_bp = st.selectbox("High Blood Pressure", [0, 1])
                high_chol = st.selectbox("High Cholesterol", [0, 1])
                bmi = st.number_input("BMI", 10.0, 70.0, 27.0)
                heart_disease = st.selectbox("Heart Disease / Attack history", [0, 1])
            with c2:
                phys_activity = st.selectbox("Physical Activity", [0, 1])
                fruits = st.selectbox("Eats Fruits regularly", [0, 1])
                veggies = st.selectbox("Eats Veggies regularly", [0, 1])
                gen_hlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
            with c3:
                diff_walk = st.selectbox("Difficulty Walking", [0, 1])
                age = st.slider("Age Group (1=18-24 ... 13=80+)", 1, 13, 8)
                income = st.slider("Income Level (1=lowest, 8=highest)", 1, 8, 5)

            if st.button("🔍 Predict"):
                input_dict = {
                    "HighBP": high_bp, "HighChol": high_chol, "BMI": bmi,
                    "HeartDiseaseorAttack": heart_disease, "PhysActivity": phys_activity,
                    "Fruits": fruits, "Veggies": veggies, "GenHlth": gen_hlth,
                    "DiffWalk": diff_walk, "Age": age, "Income": income,
                }
                input_df = pd.DataFrame([input_dict])

                # align columns to what the model was trained on
                try:
                    expected_cols = list(model.feature_names_in_)
                    for col in expected_cols:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[expected_cols]
                except AttributeError:
                    pass

                with st.spinner("Predicting..."):
                    time.sleep(0.4)
                    pred = model.predict(input_df)[0]
                    prob = (model.predict_proba(input_df)[0][1]
                            if hasattr(model, "predict_proba") else None)

                if pred == 1:
                    st.error(f"⚠️ High risk of Diabetes"
                              + (f" (probability: {prob*100:.1f}%)" if prob is not None else ""))
                else:
                    st.success(f"✅ Low risk of Diabetes"
                               + (f" (probability: {prob*100:.1f}%)" if prob is not None else ""))

        # ---- Batch CSV prediction ----
        with tab2:
            uploaded = st.file_uploader("Upload a CSV file with patient records", type="csv")
            if uploaded is not None:
                batch_df = pd.read_csv(uploaded)
                st.write("Preview:", batch_df.head())

                if st.button("Run Batch Prediction"):
                    try:
                        expected_cols = list(model.feature_names_in_)
                        pred_input = batch_df.copy()
                        for col in expected_cols:
                            if col not in pred_input.columns:
                                pred_input[col] = 0
                        pred_input = pred_input[expected_cols]
                    except AttributeError:
                        pred_input = batch_df

                    with st.spinner("Running predictions..."):
                        preds = model.predict(pred_input)
                        batch_df["Prediction"] = preds
                        batch_df["Prediction_Label"] = batch_df["Prediction"].map(
                            {0: "No Diabetes", 1: "Diabetes"}
                        )

                    st.success(f"Predicted {len(batch_df)} records.")
                    st.dataframe(batch_df, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        batch_df.to_csv(index=False).encode(),
                        "prediction_results.csv",
                        "text/csv",
                    )

# --------------------------------------------------------------------------- #
# AI ASSISTANT
# --------------------------------------------------------------------------- #
elif page == "💬 AI Assistant":
    st.title("💬 AI Assistant")
    st.markdown(
        "Ask questions about the dataset, models, or diabetes risk factors "
        "in plain English."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def ai_response(query: str) -> str:
        """Simple rule-based AI assistant using dataset/model context.
        Swap this out for a real LLM API call (OpenAI/Gemini) if you have
        an API key available — see README for the integration snippet."""
        q = query.lower()

        if df is None:
            return "Dataset isn't loaded, so I can't answer data questions right now."

        if "how many" in q and "diabetes" in q:
            n = int(df["Diabetes_binary"].sum())
            return f"There are {n:,} diabetic records out of {len(df):,} total ({n/len(df)*100:.1f}%)."
        if "best model" in q or "which model" in q:
            if comparison_df is not None:
                best = comparison_df["Accuracy"].idxmax()
                return (f"The best performing model is **{best}** with an accuracy of "
                        f"{comparison_df.loc[best,'Accuracy']:.4f}.")
            return "Run the model comparison step first to determine the best model."
        if "bmi" in q:
            avg_bmi_diab = df[df["Diabetes_binary"] == 1]["BMI"].mean()
            avg_bmi_no = df[df["Diabetes_binary"] == 0]["BMI"].mean()
            return (f"Average BMI for diabetic patients is {avg_bmi_diab:.1f}, "
                    f"vs {avg_bmi_no:.1f} for non-diabetic patients.")
        if "risk factor" in q or "important feature" in q:
            corr = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary")
            top3 = corr.abs().sort_values(ascending=False).head(3).index.tolist()
            return f"The top risk factors in this dataset are: {', '.join(top3)}."
        if "rows" in q or "size" in q or "shape" in q:
            return f"The dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."

        return ("I can answer questions about dataset size, diabetes rate, BMI patterns, "
                "top risk factors, and the best-performing model. Try asking one of those!")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Ask something about the data or models...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        reply = ai_response(user_q)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

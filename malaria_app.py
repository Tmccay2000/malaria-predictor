"""
Advanced Streamlit app for Malaria Incidence prediction:
- Real-time predictions using saved models (joblib)
- Prediction logging
- Upload evaluation dataset to compute metrics for all available models
- Ensemble predictions (average of available models)
- Visualizations: trend, predicted vs actual, residuals, feature importance (if available)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------
# Config & Globals
# ---------------------------
st.set_page_config(page_title="Malaria Predictor (Advanced)", layout="wide")

MODEL_FILES = {
    "Random Forest": "final_rf_model.pkl",
    "Gradient Boosting": "final_gb_model.pkl",
    "XGBoost": "final_xgb_model.pkl",
    "Linear Regression": "final_lr_model.pkl",
    "Decision Tree": "final_dt_model.pkl",
    "SVR": "final_svr_model.pkl",
    "HistGradientBoosting": "final_hgb_model.pkl",
    "Ridge": "final_ridge_model.pkl",
    "Lasso": "final_lasso_model.pkl"
}

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")
os.makedirs(LOG_DIR, exist_ok=True)

# Expected feature order used by models (must match training)
EXPECTED_FEATURES = [
    "RH2M_mean", "RH2M_lag_1", "RH2M_lag_2", "RH2M_lag_3",
    "Malaria_lag_1", "Malaria_lag_2", "Malaria_lag_3", "Season_Wet"
]

# mapping for friendly labels (optional)
FRIENDLY = {
    "RH2M_mean": "Avg Humidity",
    "RH2M_lag_1": "Humidity Lag1",
    "RH2M_lag_2": "Humidity Lag2",
    "RH2M_lag_3": "Humidity Lag3",
    "Malaria_lag_1": "Malaria Lag1",
    "Malaria_lag_2": "Malaria Lag2",
    "Malaria_lag_3": "Malaria Lag3",
    "Season_Wet": "Wet Season (1=yes)"
}

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_resource
def load_models(model_files):
    """Load available models from disk. Returns dict name -> model."""
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                mdl = joblib.load(path)
                models[name] = mdl
            except Exception as e:
                st.warning(f"Could not load {name} from {path}: {e}")
    return models

def log_prediction(entry: dict):
    """Append a prediction record to CSV (creates file with header if missing)."""
    df = pd.DataFrame([entry])
    header = not os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=header, index=False)

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

def safe_predict(models, X):
    """Return dict model_name -> predictions array for X. Models may differ."""
    preds = {}
    for name, mdl in models.items():
        try:
            preds[name] = mdl.predict(X)
        except Exception as e:
            st.warning(f"Prediction failed for {name}: {e}")
    return preds

# ---------------------------
# Load models once
# ---------------------------
models = load_models(MODEL_FILES)
available_model_names = list(models.keys())

# ---------------------------
# Sidebar: Navigation & Settings
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Evaluate Models", "Analytics & Logs", "Manage Models"])

st.sidebar.markdown("---")
st.sidebar.write("Available models:")
if available_model_names:
    for m in available_model_names:
        st.sidebar.write(f"- {m}")
else:
    st.sidebar.info("No saved models found. Place .pkl files in the app directory.")

# ---------------------------
# Page: HOME
# ---------------------------
if page == "Home":
    st.title("Malaria Incidence Predictor — Advanced")
    st.markdown("""
    This app provides:
    - Real-time predictions (single inputs or batch CSV)
    - Evaluate any saved models on uploaded test CSV
    - Analytics dashboard from prediction logs
    - Ensemble predictions (average of available models)
    """)
    st.info("Tip: Save models to the app folder named like `final_rf_model.pkl`, `final_xgb_model.pkl`, etc.")

# ---------------------------
# Page: Predict
# ---------------------------
elif page == "Predict":
    st.header("Real-time Prediction")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Enter features (single prediction)")
        # Default values can be adjusted
        RH2M_mean = st.number_input("RH2M_mean (Avg Humidity)", value=60.0, format="%.2f")
        RH2M_lag_1 = st.number_input("RH2M_lag_1", value=58.0, format="%.2f")
        RH2M_lag_2 = st.number_input("RH2M_lag_2", value=57.0, format="%.2f")
        RH2M_lag_3 = st.number_input("RH2M_lag_3", value=55.0, format="%.2f")
        Malaria_lag_1 = st.number_input("Malaria_lag_1", value=20.0, format="%.2f")
        Malaria_lag_2 = st.number_input("Malaria_lag_2", value=18.0, format="%.2f")
        Malaria_lag_3 = st.number_input("Malaria_lag_3", value=17.0, format="%.2f")
        season = st.selectbox("Season", ["Dry", "Wet"])
        Season_Wet = 1 if season == "Wet" else 0

        single_input = np.array([[RH2M_mean, RH2M_lag_1, RH2M_lag_2, RH2M_lag_3,
                                  Malaria_lag_1, Malaria_lag_2, Malaria_lag_3, Season_Wet]])
        use_ensemble = st.checkbox("Use ensemble (average of available models)", value=True)

        if st.button("Predict now"):
            if not models:
                st.error("No models available to predict. Please add model files and reload.")
            else:
                preds = safe_predict(models, single_input)
                st.subheader("Predictions")
                # Show model-wise predictions
                for name, p in preds.items():
                    st.write(f"- **{name}**: {p[0]:.3f} per 1000")
                # ensemble
                if use_ensemble and preds:
                    ensemble_val = np.mean([v[0] for v in preds.values()])
                    st.success(f"🔀 Ensemble (mean) prediction: {ensemble_val:.3f} per 1000")
                # log the prediction
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "RH2M_mean": RH2M_mean,
                    "RH2M_lag_1": RH2M_lag_1,
                    "RH2M_lag_2": RH2M_lag_2,
                    "RH2M_lag_3": RH2M_lag_3,
                    "Malaria_lag_1": Malaria_lag_1,
                    "Malaria_lag_2": Malaria_lag_2,
                    "Malaria_lag_3": Malaria_lag_3,
                    "Season_Wet": Season_Wet,
                    "ensemble": float(np.mean(list(preds.values())) if preds else np.nan)
                }
                # add model-wise predictions to log if desired
                for mname, mval in preds.items():
                    log_entry[f"pred_{mname}"] = float(mval[0])
                log_prediction(log_entry)

    with col2:
        st.subheader("Batch prediction (CSV)")
        st.markdown("Upload a CSV with columns matching the expected features:")
        st.write(EXPECTED_FEATURES)
        uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
        if uploaded is not None:
            df_batch = pd.read_csv(uploaded)
            # Basic validation: ensure expected columns present
            missing_cols = [c for c in EXPECTED_FEATURES if c not in df_batch.columns]
            if missing_cols:
                st.error(f"Missing columns in upload: {missing_cols}")
            else:
                if st.button("Run batch prediction"):
                    X_batch = df_batch[EXPECTED_FEATURES].values
                    preds = safe_predict(models, X_batch)
                    # create results dataframe
                    results = df_batch.copy()
                    for mname, arr in preds.items():
                        results[f"pred_{mname}"] = arr
                    if preds:
                        # ensemble
                        results["pred_ensemble"] = np.mean(np.column_stack(list(preds.values())), axis=1)
                    st.success("Batch prediction complete")
                    st.dataframe(results.head(50))
                    # append to log
                    for _, row in results.iterrows():
                        log_entry = row[EXPECTED_FEATURES].to_dict()
                        log_entry.update({"timestamp": datetime.now().isoformat()})
                        # add predictions
                        for c in results.columns:
                            if c.startswith("pred_"):
                                log_entry[c] = float(row[c])
                        log_prediction(log_entry)

# ---------------------------
# Page: Evaluate Models
# ---------------------------
elif page == "Evaluate Models":
    st.header("Model Evaluation on Uploaded Test Set")
    st.markdown("Upload a test CSV that contains the actual `Malaria Incidence` column (ground truth).")
    uploaded = st.file_uploader("Upload test CSV (must contain 'Malaria Incidence')", type=["csv"])
    if uploaded is not None:
        test_df = pd.read_csv(uploaded)
        if "Malaria Incidence" not in test_df.columns:
            st.error("Uploaded file must contain 'Malaria Incidence' column.")
        else:
            # Ensure features present
            missing = [c for c in EXPECTED_FEATURES if c not in test_df.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing}")
            else:
                X_test = test_df[EXPECTED_FEATURES].values
                y_true = test_df["Malaria Incidence"].values
                preds = safe_predict(models, X_test)
                st.subheader("Metrics per model")
                metrics_table = []
                for name, y_pred in preds.items():
                    metrics = compute_metrics(y_true, y_pred)
                    metrics_table.append({
                        "Model": name,
                        "R2": metrics["R2"],
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"]
                    })
                if metrics_table:
                    metrics_df = pd.DataFrame(metrics_table).sort_values("R2", ascending=False)
                    st.dataframe(metrics_df.style.format({"R2":"{:.3f}","RMSE":"{:.3f}","MAE":"{:.3f}"}))
                # Visualizations for best model (highest R2)
                if preds:
                    best_model = max(preds.keys(), key=lambda k: compute_metrics(y_true, preds[k])["R2"])
                    st.write(f"Showing detailed plots for **{best_model}**")
                    y_pred_best = preds[best_model]
                    # Pred vs Actual
                    fig, ax = plt.subplots()
                    ax.scatter(y_true, y_pred_best, alpha=0.7)
                    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                    ax.set_xlabel("Actual Malaria Incidence")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"Predicted vs Actual ({best_model})")
                    st.pyplot(fig)
                    # Residuals
                    residuals = y_true - y_pred_best
                    fig2, ax2 = plt.subplots()
                    sns.histplot(residuals, kde=True, ax=ax2)
                    ax2.set_title("Residuals Distribution")
                    st.pyplot(fig2)

# ---------------------------
# Page: Analytics & Logs
# ---------------------------
elif page == "Analytics & Logs":
    st.header("Analytics & Logs")
    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        # Basic cleanup
        if "timestamp" in logs.columns:
            logs["timestamp"] = pd.to_datetime(logs["timestamp"])
            logs = logs.sort_values("timestamp")
            st.subheader("Recent Prediction Log")
            st.dataframe(logs.tail(20))
            # Trend
            st.subheader("Prediction Trend Over Time")
            if "ensemble" in logs.columns:
                st.line_chart(logs.set_index("timestamp")["ensemble"].dropna())
            else:
                # pick a model col if exists
                pred_cols = [c for c in logs.columns if c.startswith("pred_")]
                if pred_cols:
                    st.line_chart(logs.set_index("timestamp")[pred_cols[0]])
            # Distribution
            st.subheader("Prediction Distribution")
            fig, ax = plt.subplots()
            sns.histplot(logs.filter(regex="^pred_").mean(axis=1), kde=True, ax=ax)
            st.pyplot(fig)
            # Correlation heatmap if enough numeric columns
            st.subheader("Correlation (inputs & predictions)")
            numeric = logs.select_dtypes(include=[np.number])
            if numeric.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        else:
            st.warning("No timestamp column in log file.")
    else:
        st.info("No logs yet. Make predictions to populate logs.")

# ---------------------------
# Page: Manage Models
# ---------------------------
elif page == "Manage Models":
    st.header("Model management")
    st.markdown("""
    - Place your saved models (joblib .pkl files) into the app folder.
    - Supported names (configurable at top of script): 
    """ )
    for k,v in MODEL_FILES.items():
        st.write(f"- {k}: `{v}`")
    st.markdown("After placing new model files, refresh the app (F5) to load them.")
    # Save models: provide option to download a sample model? (not included)
    
    
    # ---------------------------
# Page: Project Overview / About
# ---------------------------
if page == "Home":
    st.title("🌍 A Global Web Tool for Environmental Health Insight")
    st.markdown("""
    ### **Abstract**
    Malaria incidence is strongly influenced by environmental factors such as rainfall, temperature, and location.  
    In this project, we developed machine learning models to predict malaria incidence based on these parameters using curated data from **WHO**, **World Bank Climate Knowledge Portal**, and **Google Public Data**.

    ---

    ### 🧠 **Background & Problem Statement**
    With over **600,000 deaths annually**, malaria remains a major global health challenge—especially in **sub-Saharan Africa**.  
    Climate conditions significantly influence mosquito breeding and malaria outbreaks, but real-time tools to use this data for **predictive insights** are limited.

    This project aims to:
    - Analyze how **climate (precipitation, air temperature)** and **geography (lat/lon)** affect malaria incidence  
    - Train and evaluate multiple regression and classification models  
    - Deploy the best-performing model into an **interactive web tool**  
    - Offer a globally accessible resource for **researchers, policymakers, and the public**

    ---

    ### 🗂 **Dataset Sources**
    - **WHO Global Health Repository:** Malaria incidence data (2000–2021)  
    - **World Bank CCKP:** Monthly rainfall and temperature (1901–2022)  
    - **Google Public Data:** Country coordinates (latitude & longitude)

    **Preprocessing Steps**
    - Filtered and matched country data  
    - Aggregated monthly climate data into yearly averages  
    - Created new engineered features (pairwise interactions, mean, std, etc.)  
    - Introduced a binary classification label:  
      *High incidence (≥10)* vs *Low incidence (<10)*

    **Final Features Used**
    - `year`, `precipitation`, `AvMeanSurAirTemp`, `AvMaxSurAirTemp`, `AvMinSurAirTemp`, `longitude`, `latitude`  
    - Target: `incidence` (for regression) or `group` (for classification)

    ---

    ### ⚙️ **Approach & Methodology**

    **Regression Models (9 Total)**
    - Linear, Ridge, Lasso  
    - K-Nearest Neighbors, Decision Tree, Random Forest  
    - XGBoost, CatBoost, AdaBoost  

    **Classification Models (10 Total)**
    - Random Forest, Gradient Boosting, MLP, SVC  
    - Logistic Regression, XGBoost, CatBoost, AdaBoost, Decision Tree, KNN  

    **Evaluation Metrics**
    - Regression: RMSE, MAE, R²  
    - Classification: F1 Score, Accuracy, ROC AUC  
    - Cross-validation: 5-fold stratified  
    - Hyperparameter tuning: GridSearchCV  
    - Feature Selection: Recursive addition & elimination

    ---

    ### 📈 **Key Results**
    - **Best Regression Model:** CatBoost Regressor → *R² = 0.9774*  
    - **Best Classifier:** MLPClassifier → *F1 Score = 0.990*  
    - Predictions closely matched WHO real-world data (2000–2021)  
    - Web app provides:
        - 🌍 Global interactive choropleth maps  
        - 🧮 Prediction panel with live model inference  
        - 🕹 User input sliders for environmental variables  

    ---

    ### 💡 **Lessons Learned**
    - Feature selection improved interpretability but not always accuracy  
    - Normalization had marginal effects on tree-based models  
    - Removing **year** or **country** drastically reduced model accuracy  
    - Feature engineering improved performance for CatBoost and XGBoost  

    ---

    ### 💬 **Discussion**
    This project demonstrates that environmental data can be used to accurately predict malaria incidence.  
    Unlike black-box epidemiological models, this ML-driven approach is **transparent, testable, and scalable**.

    Its greatest strength lies in **accessibility** — enabling anyone to simulate future scenarios or evaluate risk levels under different environmental conditions.

    ---

    ### 🚀 **How to Use the App**
    1. Navigate to **“Predict”** to test single or batch malaria predictions.  
    2. Upload a dataset under **“Evaluate Models”** to view model performance metrics.  
    3. Explore **“Analytics & Logs”** to visualize trends from your own predictions.

    ---
    """)
    st.info("Use the sidebar to navigate between sections of the malaria prediction platform.")


"""Interactive demo. Run: streamlit run app.py"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "models" / "best_error_regressor.joblib")
model, features = bundle["model"], bundle["features"]
clf_bundle = joblib.load(ROOT / "models" / "out_of_tolerance_classifier.joblib")
clf = clf_bundle["model"]
tol = bundle["tolerance_V"]

st.set_page_config(page_title="Instrument Calibration AI", page_icon="📏", layout="centered")
st.title("AI-Based Instrument Calibration Error Prediction")
st.caption("Educational demo using a synthetic dataset. Do not use for safety-critical calibration decisions.")
with st.form("prediction"):
    st.subheader("Enter operating conditions")
    c1, c2 = st.columns(2)
    ref = c1.number_input("Reference input (V)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
    temp = c2.number_input("Ambient temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
    humidity = c1.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    load = c2.number_input("Load fraction", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    hours = c1.number_input("Operating hours", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)
    supply = c2.number_input("Supply voltage (V)", min_value=0.0, max_value=10.0, value=5.0, step=0.05)
    submitted = st.form_submit_button("Predict calibration status")
if submitted:
    row = pd.DataFrame([[ref,temp,humidity,load,hours,supply]], columns=features)
    error = float(model.predict(row)[0])
    measured_est = ref + error
    status = int(clf.predict(row)[0])
    st.metric("Predicted calibration error", f"{error:+.5f} V")
    st.metric("Estimated measured value", f"{measured_est:.5f} V")
    if status:
        st.error(f"Model flags this reading as OUT OF TOLERANCE (configured tolerance ±{tol:.2f} V).")
    else:
        st.success(f"Model flags this reading as within tolerance (configured tolerance ±{tol:.2f} V).")
    st.caption("Classifier and regressor are learned from synthetic demonstration data. Confirm with a calibrated reference instrument.")
st.divider()
st.subheader("Dataset preview")
data = pd.read_csv(ROOT / "data" / "calibration_dataset_synthetic.csv")
st.dataframe(data.head(12), use_container_width=True)
st.image(str(ROOT / "outputs" / "plots" / "error_vs_input.png"), caption="Calibration error versus reference input")

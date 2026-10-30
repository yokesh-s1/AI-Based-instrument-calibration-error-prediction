# AI-Based Instrument Calibration Error Prediction

A complete Python capstone project that estimates instrument calibration error using regression and identifies readings that may be outside a configured tolerance using classification.

## Important: dataset status
The included `data/calibration_dataset_synthetic.csv` is **synthetic demonstration data**, not actual measurements from a calibrated multimeter or sensor. It is useful for demonstrating the complete ML workflow. For a real calibration study, replace it with traceable reference/readings and document the instrument, range, resolution, environmental conditions, and reference-standard uncertainty.

## Project objectives
1. Collect reference and measured readings over several input levels and operating conditions.
2. Calculate calibration error: measured value minus reference value.
3. Train regression models to predict error.
4. Train a classifier to flag readings outside the configured tolerance.
5. Evaluate on a held-out test set and visualize error versus input.

## Folder structure
```text
AI_Instrument_Calibration_Error_Prediction/
├── app.py
├── data/
│   └── calibration_dataset_synthetic.csv
├── models/
│   ├── best_error_regressor.joblib
│   └── out_of_tolerance_classifier.joblib
├── outputs/
│   ├── metrics.json
│   └── plots/
├── reports/
│   ├── project_report.md
│   └── presentation.pptx
├── src/
│   ├── prepare_data.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Dataset fields
| Column | Meaning |
|---|---|
| `reference_value_V` | Traceable reference input (V) |
| `measured_value_V` | Instrument reading (V) |
| `ambient_temperature_C` | Ambient temperature (°C) |
| `relative_humidity_pct` | Relative humidity (%) |
| `load_fraction` | Load fraction, 0 to 1 |
| `operating_hours` | Approximate accumulated operating hours |
| `supply_voltage_V` | Instrument supply voltage (V) |
| `calibration_error_V` | Measured − reference (V) |
| `out_of_tolerance` | 1 if absolute error exceeds ±0.10 V, otherwise 0 |

**Tolerance note:** ±0.10 V is an example setting used to demonstrate classification. It is not a universal calibration tolerance. Change it to the instrument's actual specification.

## Setup
Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Re-train models
```bash
python src/train.py
```
This compares Linear Regression, Random Forest, and Gradient Boosting with an 80/20 train/test split. It saves the best regression model (selected by lowest test RMSE), a Random Forest tolerance classifier, and evaluation metrics.

## Run the prediction demo
```bash
streamlit run app.py
```
Open the local URL shown by Streamlit. Enter input voltage and operating conditions to view predicted error, estimated measured value, and tolerance status.

## Use your own calibration readings
Prepare a CSV with at least the reference and measured columns. Include the same feature columns used by the model if you plan to train it. Calculate errors and tolerance flags with:

```bash
python src/prepare_data.py your_measurements.csv --reference reference_value_V --measured measured_value_V --tolerance 0.10 --output prepared.csv
```

Then replace the demo data with your validated dataset and run `python src/train.py`. Keep a separate untouched test set for a fair final evaluation when collecting real measurements.

## Metrics
- Regression: MAE, RMSE, and R² on unseen test measurements.
- Classification: precision, recall, F1-score, and confusion matrix.
- Plots: error vs input, actual vs predicted error, and residual distribution.

See `outputs/metrics.json` for the metrics generated with the included synthetic dataset.

## GitHub upload
1. Create a new empty repository on GitHub.
2. Extract this ZIP.
3. In the extracted folder, run:
   ```bash
   git init
   git add .
   git commit -m "Add AI instrument calibration prediction capstone"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```
4. Replace the placeholder remote URL with your own repository URL.

## Limitations and responsible use
- A model trained on synthetic data learns the assumptions used to simulate it; it does not establish real instrument accuracy.
- Model predictions do not replace calibration against a traceable standard.
- Use instrument-specific tolerance, reference uncertainty, calibration procedure, and environmental limits.
- Avoid extrapolating outside the input and operating-condition ranges represented in training data.

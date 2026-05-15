# Predictive Analytics SaaS MVP — Stage 8

A lightweight Streamlit-based analytics platform for:

- Uploading CSV files
- Cleaning data
- Descriptive statistics
- Graph creation
- Smart AutoML target selection and model choice
- Regression and classification detection
- R² Score and comprehensive metrics
- Actual vs Predicted visualizations
- Feature importance analysis
- Prediction form for new inputs
- Persistent SQLite storage for uploads, experiments, and saved models

> No `src` folder is required for this MVP — the app runs directly from `app.py`. Please keep the workspace clean and compact.

## Run locally

1. Create and activate a Python environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend API server:

```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

4. Start the Streamlit frontend:

```bash
streamlit run app.py
```

## Features included

- CSV upload and data preview
- Automatic cleaning and duplicate removal
- Numeric summary and missing-value overview
- Correlation heatmap, histogram, scatter plots
- Smart AutoML target selection and model choice
- Model selection: Random Forest or Linear/Logistic Regression
- Regression: MAE, MSE, RMSE, R² Score + Actual vs Predicted scatter
- Classification: Accuracy, Classification Report + Confusion Matrix
- Feature importance charts
- Prediction form for new inputs

## Notes

This is a stage-5 MVP intended for rapid prototyping of a predictive analytics SaaS concept. It can be extended with authentication, cloud storage, multi-file dashboards, and production-grade ML pipelines.

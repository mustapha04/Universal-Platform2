import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

app = FastAPI(title="Universal Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    experiment_id: int
    input_data: Dict[str, Any]


class PredictResponse(BaseModel):
    experiment_id: int
    predicted_value: Any
    model_path: str


class TrainRequest(BaseModel):
    user_id: Optional[int]
    upload_id: int
    target: str
    model_type: str


class TrainResponse(BaseModel):
    experiment_id: int
    model_path: str
    metrics: Dict[str, Any]
    feature_names: List[str]
    problem: str


def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.replace(["NA", "na", "n/a", "null", "None", ""], pd.NA)
    cleaned = cleaned.drop_duplicates()
    return cleaned


def preprocess_features(df: pd.DataFrame, target: str):
    df_model = df.copy()
    df_model = df_model.dropna(subset=[target])
    label_encoders = {}

    categorical = df_model.select_dtypes(include=["object", "category", "string", "bool"]).columns.drop([target], errors="ignore")
    for col in categorical:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    return df_model, label_encoders


def auto_model_choice(df: pd.DataFrame, target: str, problem: str):
    feature_count = df.drop(columns=[target]).shape[1]
    if problem == "regression":
        return "Linear Regression" if feature_count < 4 else "Random Forest"
    return "Logistic Regression" if feature_count < 4 else "Random Forest"


def normalize_model_type(model_type: str, problem: str) -> str:
    if model_type == "Auto":
        return auto_model_choice(pd.DataFrame(), "", problem) if False else model_type
    if model_type == "Linear/Logistic Regression":
        return "Linear Regression" if problem == "regression" else "Logistic Regression"
    return model_type


def train_model(upload_id: int, target: str, model_type: str, user_id: Optional[int] = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stored_path FROM uploads WHERE id = ?", (upload_id,))
    upload = cursor.fetchone()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    data_path = Path(upload["stored_path"])
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file missing")

    try:
        df = pd.read_csv(data_path)
    except Exception:
        df = pd.read_csv(data_path, encoding="utf-8", engine="python")

    df = clean_data(df)
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in uploaded data")

    df_model, label_encoders = preprocess_features(df, target)
    X = df_model.drop(columns=[target])
    y = df_model[target]
    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No features available for model training")

    problem = "classification" if y.nunique() <= 10 and y.dtype == int else "regression"
    if y.dtype == object or pd.api.types.is_string_dtype(y.dtype) or pd.api.types.is_categorical_dtype(y.dtype):
        problem = "classification"
        y = LabelEncoder().fit_transform(y.astype(str))

    if model_type == "Auto":
        model_type = auto_model_choice(df_model, target, problem)
    else:
        if model_type == "Linear/Logistic Regression":
            model_type = "Linear Regression" if problem == "regression" else "Logistic Regression"

    if problem == "regression":
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
    else:
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "problem": problem,
        "model_type": model_type,
        "mae": float(mean_absolute_error(y_test, preds)) if problem == "regression" else None,
        "mse": float(mean_squared_error(y_test, preds)) if problem == "regression" else None,
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))) if problem == "regression" else None,
        "r2_score": float(r2_score(y_test, preds)) if problem == "regression" else None,
        "accuracy": float(accuracy_score(y_test, preds)) if problem != "regression" else None,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{timestamp}_{target}_{model_type.replace(' ', '_')}.pkl"
    model_path = DATA_DIR / "models" / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "encoders": {col: label_encoders[col].classes_.tolist() for col in label_encoders},
        "feature_names": X.columns.tolist(),
        "target": target,
        "problem": problem,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)

    cursor.execute(
        "INSERT INTO experiments (user_id, upload_id, target, problem_type, model_type, metrics, trained_at, model_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            upload_id,
            target,
            problem,
            model_type,
            json.dumps(metrics),
            datetime.now().isoformat(),
            str(model_path),
        ),
    )
    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "experiment_id": experiment_id,
        "model_path": str(model_path),
        "metrics": metrics,
        "feature_names": X.columns.tolist(),
        "problem": problem,
    }


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, filename TEXT, stored_path TEXT, uploaded_at TEXT, rows INTEGER, cols INTEGER, missing INTEGER, FOREIGN KEY(user_id) REFERENCES users(id))"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, upload_id INTEGER, target TEXT, problem_type TEXT, model_type TEXT, metrics TEXT, trained_at TEXT, model_path TEXT, FOREIGN KEY(upload_id) REFERENCES uploads(id), FOREIGN KEY(user_id) REFERENCES users(id))"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER, input_data TEXT, predicted_value TEXT, predicted_at TEXT, FOREIGN KEY(experiment_id) REFERENCES experiments(id))"
    )
    cursor.execute("PRAGMA table_info(uploads)")
    existing_uploads_columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in existing_uploads_columns:
        cursor.execute("ALTER TABLE uploads ADD COLUMN user_id INTEGER")
    cursor.execute("PRAGMA table_info(experiments)")
    existing_experiments_columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in existing_experiments_columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN user_id INTEGER")
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, target, problem_type, model_type, metrics, trained_at, model_path FROM experiments WHERE id = ?",
        (experiment_id,),
    )
    experiment = cursor.fetchone()
    conn.close()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    metrics = experiment["metrics"]
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            metrics = metrics

    return {
        "id": experiment["id"],
        "target": experiment["target"],
        "problem_type": experiment["problem_type"],
        "model_type": experiment["model_type"],
        "metrics": metrics,
        "trained_at": experiment["trained_at"],
        "model_path": experiment["model_path"],
    }


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    result = train_model(
        upload_id=request.upload_id,
        target=request.target,
        model_type=request.model_type,
        user_id=request.user_id,
    )
    return TrainResponse(
        experiment_id=result["experiment_id"],
        model_path=result["model_path"],
        metrics=result["metrics"],
        feature_names=result["feature_names"],
        problem=result["problem"],
    )


class UploadRecord(BaseModel):
    user_id: Optional[int]
    filename: str
    stored_path: str
    uploaded_at: str
    rows: int
    cols: int
    missing: int


class ExperimentRecord(BaseModel):
    user_id: Optional[int]
    upload_id: int
    target: str
    problem_type: str
    model_type: str
    metrics: Dict[str, Any]
    trained_at: str
    model_path: str


class PredictionRecord(BaseModel):
    experiment_id: int
    input_data: Dict[str, Any]
    predicted_value: Any
    predicted_at: str


@app.post("/uploads")
def create_upload(record: UploadRecord):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (user_id, filename, stored_path, uploaded_at, rows, cols, missing) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record.user_id,
            record.filename,
            record.stored_path,
            record.uploaded_at,
            record.rows,
            record.cols,
            record.missing,
        ),
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"upload_id": upload_id}


@app.post("/experiments")
def create_experiment(record: ExperimentRecord):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO experiments (user_id, upload_id, target, problem_type, model_type, metrics, trained_at, model_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.user_id,
            record.upload_id,
            record.target,
            record.problem_type,
            record.model_type,
            json.dumps(record.metrics),
            record.trained_at,
            record.model_path,
        ),
    )
    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"experiment_id": experiment_id}


@app.get("/experiments")
def list_experiments(user_id: Optional[int] = Query(None)):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT e.id, u.filename, u.uploaded_at, e.target, e.problem_type, e.model_type, e.metrics, e.trained_at FROM experiments e LEFT JOIN uploads u ON e.upload_id = u.id"
    params: List[Any] = []
    if user_id is not None:
        query += " WHERE e.user_id = ?"
        params.append(user_id)
    query += " ORDER BY e.trained_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "uploaded_at": row["uploaded_at"],
            "target": row["target"],
            "problem_type": row["problem_type"],
            "model_type": row["model_type"],
            "metrics": json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"],
            "trained_at": row["trained_at"],
        }
        for row in rows
    ]


@app.post("/predictions")
def create_prediction(record: PredictionRecord):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (experiment_id, input_data, predicted_value, predicted_at) VALUES (?, ?, ?, ?)",
        (
            record.experiment_id,
            json.dumps(record.input_data),
            json.dumps(record.predicted_value),
            record.predicted_at,
        ),
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"prediction_id": prediction_id}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model_path FROM experiments WHERE id = ?", (request.experiment_id,))
    experiment = cursor.fetchone()
    conn.close()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    model_path = Path(experiment["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Saved model file not found")

    try:
        with open(model_path, "rb") as f:
            model_package = pickle.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    if isinstance(model_package, dict) and "model" in model_package:
        model_obj = model_package["model"]
        encoders = model_package.get("encoders", {})
        features = model_package.get("feature_names", list(request.input_data.keys()))
    else:
        model_obj = model_package
        encoders = {}
        features = list(request.input_data.keys())

    try:
        sample = pd.DataFrame([request.input_data])
        for col in sample.columns:
            if col not in encoders:
                sample[col] = pd.to_numeric(sample[col], errors="coerce")
            else:
                raw_value = sample.at[0, col]
                classes = encoders.get(col, [])
                if raw_value in classes:
                    sample.at[0, col] = classes.index(raw_value)
                else:
                    sample.at[0, col] = -1

        sample = sample.reindex(columns=features, fill_value=0)
        prediction = model_obj.predict(sample)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictResponse(
        experiment_id=request.experiment_id,
        predicted_value=prediction[0].item() if hasattr(prediction[0], "item") else prediction[0],
        model_path=str(model_path),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)

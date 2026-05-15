import os
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predictive Analytics SaaS MVP", layout="wide")

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
BACKEND_URL = "http://localhost:8000"


def ensure_storage():
    for directory in [DATA_DIR, UPLOAD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    ensure_storage()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def call_backend_post(path, payload):
    try:
        response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


def call_backend_get(path, params=None):
    try:
        response = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


def backend_available():
    result = call_backend_get("/health")
    return isinstance(result, dict) and result.get("status") == "ok"


def backend_train(upload_id, target, model_type, user_id=None):
    payload = {
        "upload_id": upload_id,
        "target": target,
        "model_type": model_type,
        "user_id": user_id,
    }
    result = call_backend_post("/train", payload)
    if not result or result.get("error"):
        return None

    metrics = result.get("metrics", {})
    problem = result.get("problem", "regression")

    st.write("**Backend training results**")
    if problem == "regression":
        st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        st.metric("MSE", f"{metrics.get('mse', 0):.3f}")
        st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        st.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
    else:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

    return {
        "model": None,
        "feature_names": result.get("feature_names", []),
        "encoder": {},
        "target": target,
        "problem": problem,
        "experiment_id": result.get("experiment_id"),
        "backend_only": True,
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
        "CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, upload_id INTEGER, target TEXT, problem_type TEXT, model_type TEXT, metrics TEXT, trained_at TEXT, model_path TEXT, FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(upload_id) REFERENCES uploads(id))"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER, input_data TEXT, predicted_value TEXT, predicted_at TEXT, FOREIGN KEY(experiment_id) REFERENCES experiments(id))"
    )

    # Migrate existing schema by adding user_id columns if missing
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


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_user_by_username(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def register_user(username, password):
    if get_user_by_username(username):
        return None, "اسم المستخدم موجود بالفعل."
    password_hash = hash_password(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, password_hash, datetime.now().isoformat()),
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id, None


def authenticate_user(username, password):
    user = get_user_by_username(username)
    if not user:
        return None
    if user["password_hash"] != hash_password(password):
        return None
    return user


def login_user(user):
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]
    st.session_state["logged_in"] = True


def logout_user():
    for key in ["user_id", "username", "logged_in", "upload_id", "uploaded_file_key"]:
        if key in st.session_state:
            del st.session_state[key]


init_db()

st.title("Predictive Analytics SaaS Platform — Stage 8")
st.markdown(
    "Upload any CSV, explore your data, and train models through the separated FastAPI backend service."
)


def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return pd.read_csv(uploaded_file, encoding="utf-8", engine="python")


def clean_data(df):
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.replace(["NA", "na", "n/a", "null", "None", ""], np.nan)
    cleaned = cleaned.drop_duplicates()
    return cleaned


def save_uploaded_file(uploaded_file, df, user_id=None):
    filename = uploaded_file.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stored_name = f"{timestamp}_{filename}"
    stored_path = UPLOAD_DIR / stored_name
    with open(stored_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    upload_payload = {
        "user_id": user_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "uploaded_at": datetime.now().isoformat(),
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
    }
    backend_result = call_backend_post("/uploads", upload_payload)
    if backend_result and backend_result.get("upload_id") is not None:
        return backend_result["upload_id"]

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (user_id, filename, stored_path, uploaded_at, rows, cols, missing) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            filename,
            str(stored_path),
            upload_payload["uploaded_at"],
            df.shape[0],
            df.shape[1],
            int(df.isna().sum().sum()),
        ),
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return upload_id


def save_prediction(experiment_id, input_data, predicted_value):
    prediction_payload = {
        "experiment_id": experiment_id,
        "input_data": input_data,
        "predicted_value": predicted_value,
        "predicted_at": datetime.now().isoformat(),
    }
    backend_result = call_backend_post("/predictions", prediction_payload)
    if backend_result and backend_result.get("prediction_id") is not None:
        return backend_result["prediction_id"]
    return None


def get_experiments(user_id=None):
    params = {"user_id": user_id} if user_id is not None else None
    backend_result = call_backend_get("/experiments", params=params)
    if isinstance(backend_result, list):
        return backend_result

    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT e.id, u.filename, u.uploaded_at, e.target, e.problem_type, e.model_type, e.metrics, e.trained_at FROM experiments e LEFT JOIN uploads u ON e.upload_id = u.id"
    params = []
    if user_id is not None:
        query += " WHERE e.user_id = ?"
        params.append(user_id)
    query += " ORDER BY e.trained_at DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def suggest_target(df):
    candidates = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            continue
        if df[col].isna().mean() > 0.5:
            continue

        dtype = df[col].dtype
        unique_count = df[col].nunique(dropna=True)
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            candidates.append((col, "classification", unique_count))
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
            if unique_count <= 10:
                candidates.append((col, "classification", unique_count))
            else:
                candidates.append((col, "regression", unique_count))

    if not candidates:
        return df.columns[0], "regression"

    candidates = sorted(candidates, key=lambda x: (x[1] != "classification", x[2]))
    return candidates[0][0], candidates[0][1]


def auto_model_choice(df, target, problem):
    feature_count = df.drop(columns=[target]).shape[1]
    if problem == "regression":
        return "Linear Regression" if feature_count < 4 else "Random Forest"
    return "Logistic Regression" if feature_count < 4 else "Random Forest"


def describe_data(df):
    numeric = df.select_dtypes(include='number')
    desc = numeric.describe().T
    desc["missing"] = df.isna().sum()
    return desc


def plot_graphs(df):
    numeric = df.select_dtypes(include='number')

    st.subheader("Graphs")
    if numeric.shape[1] == 0:
        st.info("No numeric columns available for plotting.")
        return

    st.write("### Correlation matrix")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    col = st.selectbox("Choose numeric column for histogram", numeric.columns, index=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(numeric[col].dropna(), kde=True, ax=ax2, color="#4C72B0")
    ax2.set_title(f"Distribution of {col}")
    st.pyplot(fig2)

    if numeric.shape[1] > 1:
        x_col = st.selectbox("X axis for scatter", numeric.columns, index=0)
        y_col = st.selectbox("Y axis for scatter", numeric.columns, index=1)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=numeric, x=x_col, y=y_col, ax=ax3)
        ax3.set_title(f"Scatter: {x_col} vs {y_col}")
        st.pyplot(fig3)


def model_training(df, target, model_type="Auto", upload_id=None, user_id=None):
    if upload_id is not None:
        backend_state = backend_train(upload_id, target, model_type, user_id=user_id)
        if backend_state is not None:
            return backend_state

    if not backend_available():
        st.error(
            "❌ **Backend Service Unavailable**\n\n"
            "The FastAPI backend is not running. Please start it in a separate terminal:\n\n"
            "```bash\npython backend.py\n```\n\n"
            "Or use:\n\n"
            "```bash\nuvicorn backend:app --reload --host 0.0.0.0 --port 8000\n```\n\n"
            "**Make sure the backend starts BEFORE using this app.**"
        )
    else:
        st.error("❌ Backend service error. Check the backend console output for more details.")
    return None


def prediction_panel(state, df):
    if not state:
        return

    st.subheader("Prediction / Forecast")
    features = state["feature_names"]
    values = {}

    with st.form(key="predict_form"):
        for feature in features:
            values[feature] = st.text_input(f"Value for {feature}", value="")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    sample = pd.DataFrame([values]).reindex(columns=features, fill_value=0)
    if not state.get("experiment_id"):
        st.error("Experiment ID is required for backend prediction.")
        return

    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"experiment_id": state["experiment_id"], "input_data": sample.to_dict(orient="records")[0]},
            timeout=10,
        )
        response.raise_for_status()
        predicted_value = response.json().get("predicted_value")
    except requests.exceptions.RequestException as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.success(f"Predicted {state['target']}: {predicted_value}")
    save_prediction(state["experiment_id"], values, predicted_value)


uploaded_file = None
page = "Upload & Overview"
show_raw = False

with st.sidebar:
    st.header("SaaS Navigation")
    if st.session_state.get("logged_in"):
        st.success(f"مرحباً، {st.session_state['username']}!")
        if st.button("Logout"):
            logout_user()
            st.success("تم تسجيل الخروج.")
            st.stop()

        st.markdown("---")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        st.markdown("---")
        page = st.radio("Choose section", ["Upload & Overview", "Explore Data", "Modeling", "History"], index=0)
        st.markdown("---")
        show_raw = st.checkbox("Show raw dataset")
        st.caption("Stage 8: FastAPI backend for training and prediction with frontend/backend separation.")
    else:
        auth_tab = st.radio("حساب المستخدم", ["Login", "Register"], index=0)
        if auth_tab == "Login":
            with st.form("login_form"):
                username = st.text_input("اسم المستخدم")
                password = st.text_input("كلمة المرور", type="password")
                submit_login = st.form_submit_button("Login")
                if submit_login:
                    user = authenticate_user(username.strip(), password)
                    if user is not None:
                        login_user(user)
                        st.success("تم تسجيل الدخول بنجاح.")
                        st.stop()
                    else:
                        st.error("اسم المستخدم أو كلمة المرور غير صحيحة.")
        else:
            with st.form("register_form"):
                username = st.text_input("اسم مستخدم جديد")
                password = st.text_input("كلمة المرور", type="password")
                confirm_password = st.text_input("تأكيد كلمة المرور", type="password")
                submit_register = st.form_submit_button("Register")
                if submit_register:
                    if not username.strip() or not password:
                        st.error("يرجى إدخال اسم مستخدم وكلمة مرور.")
                    elif password != confirm_password:
                        st.error("كلمات المرور غير متطابقة.")
                    else:
                        user_id, error = register_user(username.strip(), password)
                        if user_id:
                            user = get_user_by_id(user_id)
                            login_user(user)
                            st.success("تم إنشاء الحساب وتسجيل الدخول.")
                            st.stop()
                        else:
                            st.error(error)
        st.markdown("---")
        st.info("سجّل دخولك أو أنشئ حسابًا للوصول إلى التحليلات والملفات المحفوظة.")

if not st.session_state.get("logged_in"):
    st.subheader("يرجى تسجيل الدخول لمتابعة التطبيق")
    st.info("بعد تسجيل الدخول، يمكنك رفع ملفات CSV، تدريب النموذج، وعرض تاريخك الخاص.")
    st.stop()

else:
    if uploaded_file is None:
        st.info("اختر ملف CSV من الشريط الجانبي لبدء التحليلات.")
        st.stop()

    current_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("uploaded_file_key") != current_key:
        df = load_data(uploaded_file)
        df = clean_data(df)
        st.session_state["upload_id"] = save_uploaded_file(uploaded_file, df, user_id=st.session_state.get("user_id"))
        st.session_state["uploaded_file_key"] = current_key
    else:
        df = load_data(uploaded_file)
        df = clean_data(df)

    if show_raw:
        st.subheader("Raw dataset preview")
        st.dataframe(df, use_container_width=True)

    row_col_1, row_col_2, row_col_3 = st.columns(3)
    row_col_1.metric("Rows", df.shape[0])
    row_col_2.metric("Columns", df.shape[1])
    row_col_3.metric("Missing values", int(df.isna().sum().sum()))

    if page == "Upload & Overview":
        st.header("Data Overview")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Sample records")
            st.dataframe(df.head(8), use_container_width=True)
        with col2:
            st.subheader("Column summary")
            st.write(df.dtypes.astype(str))

        st.markdown("---")
        st.subheader("Data Quality")
        st.write(describe_data(df))

    elif page == "Explore Data":
        st.header("Exploratory Analysis")
        st.markdown("Use the controls below to visualize your dataset and spot trends.")
        plot_graphs(df)

    elif page == "History":
        st.header("Saved Experiments")
        experiments = get_experiments(user_id=st.session_state.get("user_id"))
        if experiments:
            data = [
                {
                    "ID": row["id"],
                    "Uploaded file": row["filename"],
                    "Uploaded at": row["uploaded_at"],
                    "Target": row["target"],
                    "Problem": row["problem_type"],
                    "Model": row["model_type"],
                    "Metrics": row["metrics"],
                    "Trained at": row["trained_at"],
                }
                for row in experiments
            ]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No saved experiments found yet.")

    else:
        st.header("Modeling")
        st.markdown("Choose a target variable, then train using the backend API.")

        with st.expander("Model settings", expanded=True):
            use_auto = st.checkbox("Enable Smart AutoML", value=True)
            if use_auto:
                suggested_target, suggested_problem = suggest_target(df)
                st.info(f"Auto target: **{suggested_target}** ({suggested_problem})")
                model_type = auto_model_choice(df, suggested_target, suggested_problem)
                st.success(f"Auto model: **{model_type}**")
                target_col = suggested_target
            else:
                model_options = ["Random Forest", "Linear/Logistic Regression"]
                model_type = st.selectbox("Choose model type", model_options)
                target_col = st.selectbox("Choose target column", df.columns)

            train_button = st.button("Train Model 🚀")

        if train_button:
            state = model_training(
                df,
                target_col,
                model_type,
                upload_id=st.session_state.get("upload_id"),
                user_id=st.session_state.get("user_id"),
            )
            if state:
                prediction_panel(state, df)

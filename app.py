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

st.set_page_config(
    page_title="Predictive Analytics SaaS MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
BACKEND_URL = "http://localhost:8000"
BACKEND_TIMEOUT = 30  # Increased timeout

# ==================== STORAGE & DB ====================

def ensure_storage():
    """Create necessary directories."""
    for directory in [DATA_DIR, UPLOAD_DIR, DATA_DIR / "models"]:
        directory.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    """Get SQLite connection with proper settings."""
    ensure_storage()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database with all required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            stored_path TEXT,
            uploaded_at TEXT,
            rows INTEGER,
            cols INTEGER,
            missing INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            upload_id INTEGER,
            target TEXT,
            problem_type TEXT,
            model_type TEXT,
            metrics TEXT,
            trained_at TEXT,
            model_path TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(upload_id) REFERENCES uploads(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            input_data TEXT,
            predicted_value TEXT,
            predicted_at TEXT,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        )
    """)

    # Add columns if they don't exist (for migrations)
    cursor.execute("PRAGMA table_info(uploads)")
    uploads_cols = [row[1] for row in cursor.fetchall()]
    if "user_id" not in uploads_cols:
        cursor.execute("ALTER TABLE uploads ADD COLUMN user_id INTEGER")

    cursor.execute("PRAGMA table_info(experiments)")
    experiments_cols = [row[1] for row in cursor.fetchall()]
    if "user_id" not in experiments_cols:
        cursor.execute("ALTER TABLE experiments ADD COLUMN user_id INTEGER")

    conn.commit()
    conn.close()


# ==================== BACKEND COMMUNICATION ====================

def call_backend_post(path, payload, timeout=BACKEND_TIMEOUT):
    """Make POST request to backend with error handling."""
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Backend request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Make sure it's running."}
    except requests.exceptions.HTTPError as e:
        try:
            return {"error": e.response.json().get("detail", str(e))}
        except:
            return {"error": f"HTTP Error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


def call_backend_get(path, params=None, timeout=BACKEND_TIMEOUT):
    """Make GET request to backend with error handling."""
    try:
        response = requests.get(
            f"{BACKEND_URL}{path}",
            params=params,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Backend request timed out"}
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


def backend_available():
    """Check if backend is running."""
    try:
        result = call_backend_get("/health", timeout=5)
        return isinstance(result, dict) and result.get("status") == "ok"
    except:
        return False


# ==================== AUTHENTICATION ====================

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_user_by_username(username):
    """Retrieve user by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """Retrieve user by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def register_user(username, password):
    """Register a new user."""
    if get_user_by_username(username):
        return None, "Username already exists."
    
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
    """Authenticate user credentials."""
    user = get_user_by_username(username)
    if not user:
        return None
    if user["password_hash"] != hash_password(password):
        return None
    return user


def login_user(user):
    """Set user session state."""
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]
    st.session_state["logged_in"] = True


def logout_user():
    """Clear user session state."""
    for key in ["user_id", "username", "logged_in", "upload_id", "uploaded_file_key", "uploaded_file"]:
        if key in st.session_state:
            del st.session_state[key]


# ==================== DATA PROCESSING ====================

def load_data(uploaded_file):
    """Load CSV file with fallback encoding."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return pd.read_csv(uploaded_file, encoding="utf-8", engine="python")


def clean_data(df):
    """Clean and standardize dataframe."""
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned = cleaned.replace(["NA", "na", "n/a", "null", "None", ""], np.nan)
    cleaned = cleaned.drop_duplicates()
    return cleaned


def save_uploaded_file(uploaded_file, df, user_id=None):
    """Save uploaded file and register in database."""
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
    
    # Try backend first
    backend_result = call_backend_post("/uploads", upload_payload)
    if backend_result and backend_result.get("upload_id") is not None:
        return backend_result["upload_id"]

    # Fallback to local database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO uploads (user_id, filename, stored_path, uploaded_at, rows, cols, missing) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
    """Save prediction result."""
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


# ==================== MODEL OPERATIONS ====================

def suggest_target(df):
    """Auto-suggest best target column."""
    candidates = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            continue
        if df[col].isna().mean() > 0.5:
            continue

        dtype = df[col].dtype
        unique_count = df[col].nunique(dropna=True)
        
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
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
    """Auto-select model based on data characteristics."""
    feature_count = df.drop(columns=[target]).shape[1]
    if problem == "regression":
        return "Linear Regression" if feature_count < 4 else "Random Forest"
    return "Logistic Regression" if feature_count < 4 else "Random Forest"


def backend_train(upload_id, target, model_type, user_id=None):
    """Train model via backend."""
    payload = {
        "upload_id": upload_id,
        "target": target,
        "model_type": model_type,
        "user_id": user_id,
    }
    result = call_backend_post("/train", payload)
    
    if not result or result.get("error"):
        st.error(f"❌ Training error: {result.get('error', 'Unknown error')}")
        return None

    metrics = result.get("metrics", {})
    problem = result.get("problem", "regression")

    col1, col2, col3, col4 = st.columns(4)
    
    if problem == "regression":
        col1.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
        col2.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        col3.metric("MSE", f"{metrics.get('mse', 0):.3f}")
        col4.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    else:
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")

    return {
        "model": None,
        "feature_names": result.get("feature_names", []),
        "encoder": {},
        "target": target,
        "problem": problem,
        "experiment_id": result.get("experiment_id"),
        "backend_only": True,
    }


def model_training(df, target, model_type="Auto", upload_id=None, user_id=None):
    """Handle model training with backend check."""
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
            "Make sure the backend starts BEFORE using this app."
        )
    else:
        st.error("❌ Backend error. Check the backend console for details.")
    return None


def get_experiments(user_id=None):
    """Retrieve experiments for user."""
    params = {"user_id": user_id} if user_id is not None else None
    backend_result = call_backend_get("/experiments", params=params)
    
    if isinstance(backend_result, list):
        return backend_result

    # Fallback to local database
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """SELECT e.id, u.filename, u.uploaded_at, e.target, e.problem_type, 
               e.model_type, e.metrics, e.trained_at FROM experiments e 
               LEFT JOIN uploads u ON e.upload_id = u.id"""
    params_list = []
    
    if user_id is not None:
        query += " WHERE e.user_id = ?"
        params_list.append(user_id)
    
    query += " ORDER BY e.trained_at DESC"
    cursor.execute(query, params_list)
    rows = cursor.fetchall()
    conn.close()
    return rows


# ==================== DATA VISUALIZATION ====================

def describe_data(df):
    """Generate descriptive statistics."""
    numeric = df.select_dtypes(include='number')
    if numeric.empty:
        return pd.DataFrame()
    
    desc = numeric.describe().T
    desc["missing"] = df[numeric.columns].isna().sum()
    return desc


def plot_graphs(df):
    """Create interactive visualizations."""
    numeric = df.select_dtypes(include='number')

    st.subheader("📊 Data Visualizations")
    
    if numeric.shape[1] == 0:
        st.info("No numeric columns available for plotting.")
        return

    # Correlation heatmap
    st.write("#### Correlation Matrix")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Correlation'})
    st.pyplot(fig)

    # Histogram
    col_hist = st.selectbox("Choose numeric column for histogram", numeric.columns, key="hist_col")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(numeric[col_hist].dropna(), kde=True, ax=ax2, color="#4C72B0", bins=30)
    ax2.set_title(f"Distribution of {col_hist}")
    ax2.set_xlabel(col_hist)
    st.pyplot(fig2)

    # Scatter plot
    if numeric.shape[1] > 1:
        col1_scatter = st.selectbox("X axis for scatter", numeric.columns, index=0, key="scatter_x")
        col2_scatter = st.selectbox("Y axis for scatter", numeric.columns, index=min(1, numeric.shape[1]-1), key="scatter_y")
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=numeric, x=col1_scatter, y=col2_scatter, ax=ax3, alpha=0.6)
        ax3.set_title(f"Scatter: {col1_scatter} vs {col2_scatter}")
        st.pyplot(fig3)


def prediction_panel(state, df):
    """Handle prediction input and display results."""
    if not state:
        return

    st.subheader("🔮 Make Predictions")
    features = state["feature_names"]
    
    if not features:
        st.warning("No features available for prediction.")
        return

    with st.form(key="predict_form"):
        values = {}
        cols = st.columns(min(3, len(features)))
        
        for idx, feature in enumerate(features):
            with cols[idx % len(cols)]:
                values[feature] = st.text_input(
                    f"{feature}",
                    value="0",
                    key=f"pred_{feature}"
                )
        
        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if not submitted:
        return

    if not state.get("experiment_id"):
        st.error("Experiment ID required for prediction.")
        return

    try:
        sample = pd.DataFrame([values]).reindex(columns=features, fill_value=0)
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "experiment_id": state["experiment_id"],
                "input_data": sample.to_dict(orient="records")[0]
            },
            timeout=BACKEND_TIMEOUT,
        )
        response.raise_for_status()
        predicted_value = response.json().get("predicted_value")
        
        st.success(f"✅ Predicted **{state['target']}**: `{predicted_value}`")
        save_prediction(state["experiment_id"], values, predicted_value)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")


# ==================== INITIALIZATION ====================

init_db()

# Page configuration
st.title("🎯 Predictive Analytics SaaS Platform")
st.markdown(
    "**Stage 8** — Upload CSV, explore data, and train ML models with FastAPI backend"
)

# ==================== AUTHENTICATION UI ====================

with st.sidebar:
    st.header("🔐 Account")
    
    if st.session_state.get("logged_in"):
        st.success(f"👤 {st.session_state['username']}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.success("Logged out successfully.")
            st.rerun()

        st.markdown("---")
        
        uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"], key="csv_upload")
        
        # Store uploaded file in session state if provided
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
        
        st.markdown("---")
        
        page = st.radio(
            "📑 Navigation",
            ["Upload & Overview", "Explore Data", "Modeling", "History"],
            index=0
        )
        
        st.markdown("---")
        show_raw = st.checkbox("👁️ Show raw dataset", value=False)
        
        st.caption("⚡ Backend-powered ML training & prediction")
        
    else:
        auth_tab = st.radio("Choose:", ["Login", "Register"], index=0)
        
        if auth_tab == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
                if submit_login:
                    if not username or not password:
                        st.error("Please enter username and password.")
                    else:
                        user = authenticate_user(username.strip(), password)
                        if user is not None:
                            login_user(user)
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
        else:
            with st.form("register_form"):
                username = st.text_input("New username")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                submit_register = st.form_submit_button("Register", use_container_width=True)
                
                if submit_register:
                    if not username or not password:
                        st.error("Please enter username and password.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        user_id, error = register_user(username.strip(), password)
                        if user_id:
                            user = get_user_by_id(user_id)
                            login_user(user)
                            st.success("Account created and logged in!")
                            st.rerun()
                        else:
                            st.error(error or "Registration failed.")
        
        st.markdown("---")
        st.info("🔒 Create an account or login to get started.")

# ==================== MAIN APP ====================

if not st.session_state.get("logged_in"):
    st.info("Please login or register above to continue.")
    st.stop()

# Get uploaded file from session state
uploaded_file = st.session_state.get("uploaded_file")

if uploaded_file is None:
    st.info("👆 Select a CSV file from the sidebar to begin analysis.")
    st.stop()

# Load and process data
current_key = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.get("uploaded_file_key") != current_key:
    try:
        df = load_data(uploaded_file)
        df = clean_data(df)
        st.session_state["upload_id"] = save_uploaded_file(
            uploaded_file, df, user_id=st.session_state.get("user_id")
        )
        st.session_state["uploaded_file_key"] = current_key
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()
else:
    try:
        df = load_data(uploaded_file)
        df = clean_data(df)
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.stop()

# Display data summary
col1, col2, col3 = st.columns(3)
col1.metric("📊 Rows", f"{df.shape[0]:,}")
col2.metric("📋 Columns", df.shape[1])
col3.metric("⚠️ Missing Values", f"{int(df.isna().sum().sum()):,}")

if show_raw:
    with st.expander("Raw Dataset Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

# ==================== PAGE: UPLOAD & OVERVIEW ====================

if page == "Upload & Overview":
    st.header("📥 Data Overview")
    
    col_data, col_info = st.columns([2, 1])
    
    with col_data:
        st.subheader("Sample Records")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col_info:
        st.subheader("Column Types")
        st.write(df.dtypes.astype(str))

    st.markdown("---")
    st.subheader("📈 Data Quality Report")
    st.dataframe(describe_data(df), use_container_width=True)

# ==================== PAGE: EXPLORE DATA ====================

elif page == "Explore Data":
    st.header("🔍 Exploratory Data Analysis")
    st.markdown("Visualize your data to discover patterns and relationships.")
    plot_graphs(df)

# ==================== PAGE: MODELING ====================

elif page == "Modeling":
    st.header("🤖 Model Training")
    st.markdown("Select a target variable and train an ML model.")

    with st.expander("⚙️ Model Configuration", expanded=True):
        use_auto = st.checkbox("🧠 Enable Smart AutoML", value=True)
        
        if use_auto:
            suggested_target, suggested_problem = suggest_target(df)
            st.info(f"**Auto target:** `{suggested_target}` ({suggested_problem})")
            
            model_type = auto_model_choice(df, suggested_target, suggested_problem)
            st.success(f"**Auto model:** `{model_type}`")
            
            target_col = suggested_target
        else:
            col_model, col_target = st.columns(2)
            with col_model:
                model_type = st.selectbox(
                    "Model Type",
                    ["Random Forest", "Linear/Logistic Regression"]
                )
            with col_target:
                target_col = st.selectbox("Target Column", df.columns)

        train_button = st.button("🚀 Train Model", use_container_width=True)

    if train_button:
        if st.session_state.get("upload_id") is None:
            st.error("❌ Upload ID not found. Please re-upload your file.")
        else:
            with st.spinner("⏳ Training model..."):
                state = model_training(
                    df,
                    target_col,
                    model_type,
                    upload_id=st.session_state.get("upload_id"),
                    user_id=st.session_state.get("user_id"),
                )
            
            if state:
                st.success("✅ Model trained successfully!")
                st.markdown("---")
                prediction_panel(state, df)

# ==================== PAGE: HISTORY ====================

elif page == "History":
    st.header("📚 Experiment History")
    st.markdown("View all your trained models and their performance metrics.")
    
    experiments = get_experiments(user_id=st.session_state.get("user_id"))
    
    if experiments:
        data = []
        for row in experiments:
            metrics_str = row["metrics"] if isinstance(row["metrics"], str) else "{}"
            try:
                metrics = json.loads(metrics_str)
            except:
                metrics = {}
            
            data.append({
                "ID": row["id"],
                "File": row["filename"],
                "Target": row["target"],
                "Problem": row["problem_type"],
                "Model": row["model_type"],
                "Accuracy/R²": metrics.get("accuracy") or metrics.get("r2_score"),
                "Trained": row["trained_at"][:10] if row["trained_at"] else "N/A",
            })
        
        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No experiments found. Train a model to get started!")

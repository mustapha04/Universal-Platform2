# Predictive Analytics SaaS Platform — Stage 8

A modern full-stack SaaS application for data analysis and machine learning model training.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mustapha04/Universal-Platform2.git
cd Universal-Platform2

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Terminal 1 - Start Backend (MUST run first):**
```bash
python backend.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

**Terminal 2 - Start Frontend (after backend is running):**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## ✨ Features

- 👤 **User Authentication** - Register and login with secure password hashing
- 📊 **Data Upload & Exploration** - Upload CSV files and view statistics
- 📈 **Interactive Visualizations** - Correlation heatmaps, histograms, scatter plots
- 🤖 **Automated ML Training** - Smart AutoML target selection and model choice
- 🎯 **Multiple Model Types** - Random Forest, Linear/Logistic Regression
- 🔮 **Predictions** - Make predictions with trained models
- 💾 **Experiment History** - Track all your past models and experiments
- 📉 **Comprehensive Metrics** - R² Score, Accuracy, MAE, MSE, RMSE

---

## 🏗️ Architecture

```
Frontend (Streamlit)  ←→  Backend (FastAPI)  ←→  SQLite Database
    app.py                 backend.py              app.db
```

### Tech Stack
- **Frontend**: Streamlit (interactive web UI)
- **Backend**: FastAPI (async REST API)
- **ML**: scikit-learn (model training)
- **Data**: Pandas, NumPy (data processing)
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite (lightweight storage)

---

## 🔌 API Endpoints

### Health
- `GET /health` - Check if backend is running

### Training
- `POST /train` - Train a new ML model
- `GET /experiments` - List all experiments
- `GET /experiments/{id}` - Get experiment details

### Predictions
- `POST /predict` - Make predictions with a trained model
- `POST /predictions` - Save prediction results

### Data Management
- `POST /uploads` - Register uploaded file
- `POST /experiments` - Register experiment
- `POST /predictions` - Register prediction

---

## 📁 Project Structure

```
Universal-Platform2/
├── backend.py              # FastAPI backend server
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── data/                  # Auto-created data directory
    ├── uploads/           # User uploaded CSV files
    ├── models/            # Trained model files (.pkl)
    └── app.db             # SQLite database
```

---

## 🛠️ Troubleshooting

### "Backend unavailable" Error

**Problem**: Frontend can't connect to backend
- ✅ Make sure `python backend.py` is running in Terminal 1
- ✅ Wait for backend to fully start (look for "Application startup complete")
- ✅ Start frontend in a separate terminal

**Port Already in Use?**
```bash
# macOS/Linux - Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Database Corrupted

```bash
# Remove database and let it regenerate
rm -rf data/app.db

# Restart both backend and frontend
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

---

## 📚 Usage Workflow

1. **Register/Login** - Create an account or log in with existing credentials
2. **Upload Data** - Select a CSV file from the sidebar
3. **Explore** - View data statistics and visualizations
4. **Train Model** - Choose target variable and let AutoML select the best model
5. **Predict** - Make predictions with new data
6. **History** - Review all past experiments and models

---

## 🔐 Security Notes

- Passwords are hashed with SHA-256
- CORS is enabled for all origins (development only)
- Database uses foreign keys for referential integrity
- User data is isolated by user_id

---

## 📝 Notes

- Stage 8: Separated FastAPI backend and Streamlit frontend
- Models are saved as pickle files in `data/models/`
- All experiments and predictions are logged to SQLite database
- The platform supports both regression and classification problems

---

## 👨‍💻 Author

mustapha04

## 📄 License

MIT

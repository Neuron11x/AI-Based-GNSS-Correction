# GNSS AI Correction Dashboard
**GLA University — B.Tech Project 2025-26**

A full-stack web application for AI-based GNSS error correction using XGBoost + IMU Fusion.
Features a stunning 3D Three.js frontend and a FastAPI backend.

---

## Project Structure

```
gnss_project/
├── backend/
│   ├── main.py              ← FastAPI app (all routes)
│   ├── requirements.txt     ← Python dependencies
│   └── models/              ← Put your .pkl model files here
│       ├── xgb_gnss_model.pkl   (from your Colab training)
│       ├── rf_gnss_model.pkl
│       └── scaler.pkl
├── frontend/
│   ├── public/index.html
│   ├── src/
│   │   ├── App.js           ← Main React app (3 tabs)
│   │   ├── index.js
│   │   └── components/
│   │       ├── useThreeScene.js  ← Three.js 3D globe + satellites
│   │       └── Charts.js         ← Recharts line, trajectory, heatmap
│   └── package.json
└── sample_data/
    └── sample_gnss.csv      ← Use this to test the Upload tab
```

---

## Quick Start

### 1 — Backend (Python)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API will be live at: http://localhost:8000
Interactive docs at: http://localhost:8000/docs

### 2 — Frontend (React)

```bash
cd frontend
npm install
npm start
```

Frontend will open at: http://localhost:3000

---

## Using Your Trained Models

After running your Colab notebook (`GNSS_Correction_Final.py`),
copy the saved model files into `backend/models/`:

```
xgb_gnss_model.pkl
rf_gnss_model.pkl
scaler.pkl
```

The API automatically loads them on startup.
If no models are found, the backend falls back to a rule-based simulation
so the UI still works for demos.

---

## API Endpoints

| Method | Path           | Description                                 |
|--------|----------------|---------------------------------------------|
| GET    | /api/health    | Check if models are loaded                  |
| GET    | /api/demo      | Get demo dashboard data                     |
| POST   | /api/predict   | Predict correction for a single GPS epoch   |
| POST   | /api/upload    | Upload device_gnss.csv — batch correction   |

### POST /api/predict  — example body
```json
{
  "mean_cn0": 32.0,
  "min_elevation": 45.0,
  "multipath_sum": 2.0,
  "iono_delay": 1.2,
  "tropo_delay": 0.8,
  "num_satellites": 10.0,
  "displacement": 5.0,
  "speed_derived": 2.0,
  "acceleration": 0.1,
  "heading_change": 5.0,
  "pos_variance": 1.0,
  "accel_x": 0.1, "accel_y": 0.1, "accel_z": 9.8,
  "gyro_x": 0.01, "gyro_y": 0.01, "gyro_z": 0.01,
  "raw_lat": 28.613939,
  "raw_lon": 77.209023
}
```

---

## Frontend Features

### Dashboard tab
- Rotating 3D globe with satellites and signal lines (mouse-interactive)
- Live error chart (raw vs corrected vs CN0)
- Spatial error heatmap
- Trajectory comparison (raw vs AI-corrected)
- Quick Predict panel

### Predict tab
- Full signal + IMU parameter form
- Returns corrected lat/lon, improvement %, quality score

### Upload tab
- Drag-and-drop CSV upload
- Processes entire device_gnss.csv files
- Returns charts, heatmap, trajectory, and summary stats
- Use `sample_data/sample_gnss.csv` to test

---

## Tech Stack

| Layer    | Technology                              |
|----------|-----------------------------------------|
| Frontend | React 18, Three.js, Recharts, Axios     |
| Backend  | FastAPI, Uvicorn, Pandas, NumPy         |
| ML Model | XGBoost, Scikit-learn, RandomForest     |
| Fonts    | Syne (display), JetBrains Mono (data)   |

---

GLA University · B.Tech CSE · 2025-26

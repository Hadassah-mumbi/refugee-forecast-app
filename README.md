# Refugee Flow Forecasting and Resource Allocation System

The active application is a React + Vite single-page frontend with a FastAPI backend. The original trained model, training script, and CSV inputs remain in the repository root and are loaded by the API in place; no database or replacement model is introduced.

## Current implementation

- Historical origin/asylum country pair counts from `data/persons_of_concern.csv` (2016–2024 in the supplied extract).
- Interactive MapLibre map with pan/zoom, country-centre population circles, country boundaries, and animated country-to-country lines derived from the recorded origin/asylum pairs. Map routes are illustrative country-level connections, not verified journey paths.
- Play/pause year timeline, draggable year slider, origin/asylum modes, country filtering, country selection, and historical country charts.
- FastAPI endpoints for historical map data, country details, existing-model scenarios, and the last observed host-capacity values.
- Clearly separated model scenarios; the existing model predicts a country-level total and is not an origin-specific route forecast. Forecast assumptions and capacity comparisons are explicitly labeled.

The map uses OpenStreetMap/CARTO raster tiles and Natural Earth country geometry distributed via the `datasets/geo-countries` project. Their attribution is shown on the map; network access is required for those map resources and Google Fonts.

## Project structure

- `frontend/` — React, Vite, MapLibre GL JS, D3 geography utilities, Recharts.
- `api/` — FastAPI application and data/model API.
- `data/` — original conflict, climate, host-capacity, and persons-of-concern CSVs (preserved).
- `model.joblib`, `train_model.py` — original trained artifact and training/preprocessing script (preserved).
- `refugee-forecast-app/` — pre-existing nested duplicate project copy; not used by the new root React application.

## Run locally

Requirements: Python 3.10+ and Node.js 20+.

### Windows PowerShell

From the repository root, create and activate a Python virtual environment and install the backend requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

In PowerShell terminal 1, run the API:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In PowerShell terminal 2, install and run the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). The API is available at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs`.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

## Data and model notes

- Historical counts are filtered to refugee records; they are not live data.
- `train_model.py` and its original preprocessing are unchanged. The existing model was trained to predict a country-level refugee total from conflict count, conflict intensity, drought index, and year. Its projection is not an origin-to-asylum flow prediction.
- Forecast inputs hold the latest available country conflict/climate inputs constant while advancing the year. Treat outputs as exploratory scenarios, not operational forecasts.
- `Host_capacity.csv` values are historical observations. Any comparison with future model output is a rough planning signal, not an assessment of current shelter availability.
- A scikit-learn version warning may appear if the model is opened with a version different from the one used to train it. Keep the existing artifact and align scikit-learn versions before relying on its predictions.

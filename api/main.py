from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model.joblib"

# Country names are canonicalized to make labels and model features consistent.
COUNTRY_ALIASES = {
    "United Rep. of Tanzania": "Tanzania",
    "United Republic of Tanzania": "Tanzania",
    "DR Congo": "Democratic Republic of the Congo",
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
}

app = FastAPI(
    title="Refugee Flow Forecast API",
    description="Historical refugee totals and clearly labeled model projections.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def normalize_country(value: object) -> str:
    name = str(value).strip()
    return COUNTRY_ALIASES.get(name, name)


@lru_cache(maxsize=1)
def load_history() -> pd.DataFrame:
    path = DATA_DIR / "persons_of_concern.csv"
    if not path.exists():
        raise RuntimeError(f"Historical dataset not found: {path}")
    frame = pd.read_csv(path)
    required = {"Year", "Country of Asylum", "Country of Origin", "Population type", "Total"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Historical dataset is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["Population type"].eq("Refugees")].copy()
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["Total"] = pd.to_numeric(frame["Total"], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["Year", "Country of Asylum", "Country of Origin"])
    frame["Year"] = frame["Year"].astype(int)
    frame["origin"] = frame["Country of Origin"].map(normalize_country)
    frame["asylum"] = frame["Country of Asylum"].map(normalize_country)
    return frame


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained model not found: {MODEL_PATH}")
    # The checked-in artifact was trained with scikit-learn 1.8. Keep the model
    # and surface compatibility warnings rather than retraining or altering it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        return joblib.load(MODEL_PATH)


def get_model_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    conflicts_path = DATA_DIR / "East_Africa_conflicts.csv"
    climate_path = DATA_DIR / "Climate_data.csv"
    conflicts = pd.read_csv(conflicts_path)
    climate = pd.read_csv(climate_path)
    conflicts["year"] = pd.to_datetime(conflicts["WEEK"], errors="coerce").dt.year
    conflicts["country"] = conflicts["COUNTRY"].map(normalize_country)
    conflicts["FATALITIES"] = pd.to_numeric(conflicts["FATALITIES"], errors="coerce").fillna(0)
    conflict_features = conflicts.dropna(subset=["year"]).groupby(["country", "year"]).agg(
        Conflict_Count=("COUNTRY", "size"),
        fatalities=("FATALITIES", "sum"),
    ).reset_index()
    conflict_features["Conflict_Intensity"] = np.clip(conflict_features["fatalities"] / 10, 1, 10)

    climate["country"] = climate["country"].map(normalize_country)
    climate["year"] = pd.to_numeric(climate["year"], errors="coerce")
    climate["Weather"] = pd.to_numeric(climate["drought_index"], errors="coerce")
    climate = climate.dropna(subset=["year"])
    return conflict_features, climate


def get_latest_capacity(country: str) -> dict | None:
    capacity_path = DATA_DIR / "Host_capacity.csv"
    if not capacity_path.exists():
        return None
    capacity = pd.read_csv(capacity_path)
    required = {"country", "year", "total_capacity", "estimated_occupancy"}
    if not required.issubset(capacity.columns):
        return None
    capacity["country"] = capacity["country"].map(normalize_country)
    capacity["year"] = pd.to_numeric(capacity["year"], errors="coerce")
    capacity = capacity.loc[capacity["country"].eq(country)].dropna(subset=["year"])
    if capacity.empty:
        return None
    row = capacity.sort_values("year").iloc[-1]
    total_capacity = int(row["total_capacity"])
    occupancy = int(row["estimated_occupancy"])
    return {
        "year": int(row["year"]),
        "total_capacity": total_capacity,
        "estimated_occupancy": occupancy,
        "available_capacity": max(0, total_capacity - occupancy),
        "recorded_shortage": max(0, occupancy - total_capacity),
        "source": "data/Host_capacity.csv",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "historical_years": sorted(load_history()["Year"].unique().tolist())}


@app.get("/api/years")
def years():
    history = load_history()
    return {
        "years": sorted(history["Year"].unique().tolist()),
        "min_year": int(history["Year"].min()),
        "max_year": int(history["Year"].max()),
        "source": "data/persons_of_concern.csv",
        "population_type": "Refugees",
    }


@app.get("/api/map")
def map_data(
    year: int = Query(..., ge=1900, le=2100),
    mode: Literal["origin", "asylum"] = "origin",
    country: str | None = None,
):
    history = load_history()
    rows = history.loc[history["Year"].eq(year)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No refugee records for {year}.")

    country = normalize_country(country) if country else None
    flows = rows.groupby(["origin", "asylum"], as_index=False)["Total"].sum()
    if country and mode == "origin":
        flows = flows.loc[flows["origin"].eq(country)]
    elif country and mode == "asylum":
        flows = flows.loc[flows["asylum"].eq(country)]

    aggregate_key = "origin" if mode == "origin" else "asylum"
    totals = rows.groupby(aggregate_key, as_index=False)["Total"].sum()
    countries = [
        {"name": str(row[aggregate_key]), "total": int(row["Total"])}
        for _, row in totals.iterrows()
    ]
    flow_records = [
        {"origin": row.origin, "asylum": row.asylum, "total": int(row.Total)}
        for row in flows.itertuples(index=False)
    ]
    return {
        "year": year,
        "mode": mode,
        "selected_country": country,
        "population_type": "Refugees",
        "total_refugees": int(rows["Total"].sum()),
        "countries": countries,
        "flows": flow_records,
        "source": "UNHCR persons of concern extract supplied with this project (refugees only)",
        "note": "Counts are recorded totals; map coordinates are country-level reference points, not individual locations or verified travel routes.",
    }


@app.get("/api/country/{country_name}")
def country_detail(
    country_name: str,
    mode: Literal["origin", "asylum"] = "origin",
    year: int | None = Query(default=None, ge=1900, le=2100),
):
    name = normalize_country(country_name)
    history = load_history()
    key = "origin" if mode == "origin" else "asylum"
    rows = history.loc[history[key].eq(name)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No refugee records found for {name}.")
    annual = rows.groupby("Year", as_index=False)["Total"].sum().sort_values("Year")
    route_key = "asylum" if mode == "origin" else "origin"
    routes = rows.loc[rows["Year"].eq(year)] if year is not None else rows
    top_routes = routes.groupby(route_key, as_index=False)["Total"].sum().nlargest(5, "Total")
    return {
        "country": name,
        "mode": mode,
        "year": year,
        "total_refugees": int(rows.loc[rows["Year"].eq(year), "Total"].sum()) if year is not None else int(rows["Total"].sum()),
        "annual": [{"year": int(row.Year), "total": int(row.Total)} for row in annual.itertuples(index=False)],
        "top_routes": [{"country": str(row[route_key]), "total": int(row["Total"])} for _, row in top_routes.iterrows()],
        "source": "data/persons_of_concern.csv",
    }


@app.get("/api/forecast")
def forecast(
    country: str,
    start_year: int = Query(2025, ge=2016, le=2035),
    years: int = Query(3, ge=1, le=10),
):
    country = normalize_country(country)
    try:
        model = load_model()
        conflict_features, climate = get_model_inputs()
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    host_conflicts = conflict_features.loc[conflict_features["country"].eq(country)].sort_values("year")
    host_climate = climate.loc[climate["country"].eq(country)].sort_values("year")
    conflict_row = host_conflicts.iloc[-1] if not host_conflicts.empty else None
    climate_row = host_climate.iloc[-1] if not host_climate.empty else None
    conflict_count = int(conflict_row["Conflict_Count"]) if conflict_row is not None else 0
    intensity = float(conflict_row["Conflict_Intensity"]) if conflict_row is not None else 1.0
    weather = float(climate_row["Weather"]) if climate_row is not None and pd.notna(climate_row["Weather"]) else 0.5

    results = []
    for target_year in range(start_year, start_year + years):
        if target_year > 2035:
            break
        features = pd.DataFrame(
            [[conflict_count, intensity, weather, target_year]],
            columns=["Conflict_Count", "Conflict_Intensity", "Weather", "year"],
        )
        predicted_total = max(0, int(round(float(model.predict(features)[0]))))
        results.append({"year": target_year, "total": predicted_total, "kind": "prediction"})
    return {
        "country": country,
        "modeled_population": "host-country refugee total (not origin-specific flows)",
        "input_basis": "latest available country-level conflict and drought inputs, held constant; only the year feature advances",
        "observed_through": int(load_history()["Year"].max()),
        "latest_host_capacity": get_latest_capacity(country),
        "predictions": results,
        "source": "Existing model.joblib and the project's conflict/climate CSVs",
        "caveat": "Model-based scenario only; predictions are not observed counts, origin-specific routes, or an independently validated forecast.",
    }

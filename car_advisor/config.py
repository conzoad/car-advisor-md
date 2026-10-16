"""Центральная конфигурация проекта: пути, источник данных, база данных."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Пути ---
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "cars_moldova.csv"
RAW_META = DATA_DIR / "raw" / "SOURCE.json"
NEW_LISTINGS_CSV = DATA_DIR / "new" / "new_listings_holdout.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
QUALITY_REPORT_JSON = REPORTS_DIR / "data_quality.json"
QUALITY_REPORT_MD = REPORTS_DIR / "data_quality.md"
BASELINE_REPORT_JSON = REPORTS_DIR / "baseline.json"

# --- Источник данных (Kaggle, лицензия CC0) ---
KAGGLE_DATASET = "alexandrududa/cars-moldova"
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"

# Объявления собраны с 999.md в 2021 году: возраст авто считается относительно года публикации.
SNAPSHOT_YEAR = 2021
MIN_YEAR = 1980
RANDOM_STATE = 42


def _sqlite_url(path: Path) -> str:
    return URL.create("sqlite", database=str(path)).render_as_string(hide_password=False)


DATABASE_URL = os.getenv("DATABASE_URL") or _sqlite_url(DATA_DIR / "cars.db")

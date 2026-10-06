"""Загрузка открытого датасета «Cars for sale in Moldova» (Kaggle, CC0) через публичный Kaggle API.

Запуск: python -m car_advisor.data.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from car_advisor import config

EXPECTED_COLUMNS = [
    "Make", "Model", "Year", "Style", "Distance", "Engine_capacity(cm3)",
    "Fuel_type", "Transmission", "Price(euro)",
]


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Скачивает архив датасета (анонимно — датасет публичный), распаковывает CSV и сохраняет метаданные источника."""
    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    response = requests.get(config.KAGGLE_DOWNLOAD_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        content = archive.read(csv_name)

    df = pd.read_csv(io.BytesIO(content))
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В скачанном файле нет ожидаемых колонок: {sorted(missing)}")

    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    config.RAW_CSV.write_bytes(content)
    config.RAW_META.write_text(json.dumps({
        "dataset": config.KAGGLE_DATASET,
        "url": config.KAGGLE_DATASET_URL,
        "license": "CC0: Public Domain",
        "original_source": "999.md — объявления о продаже автомобилей в Республике Молдова (2021)",
        "file": csv_name,
        "rows": int(len(df)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="скачать заново, даже если файл уже есть")
    args = parser.parse_args()
    df = download_dataset(force=args.force)
    print(f"Датасет: {config.RAW_CSV} — {len(df):,} строк, {df.shape[1]} колонок")


if __name__ == "__main__":
    main()

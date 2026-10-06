"""Слой базы данных (SQLAlchemy Core): SQLite по умолчанию, PostgreSQL через DATABASE_URL.

Таблицы:
  listings_raw     — сырые объявления в том виде, в каком пришли из источника;
  listings         — очищенные объявления (после правил качества);
  quality_issues   — журнал проблем качества данных (что удалено/исправлено и почему).

Запуск: python -m car_advisor.data.database  (создать таблицы и загрузить сырые данные)
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Float, Index, Integer, MetaData, String, Table, Text,
    create_engine, delete, event, insert, select,
)
from sqlalchemy.engine import Engine

from car_advisor import config

metadata = MetaData()

LISTING_COLUMNS = [
    "make", "model", "year", "style", "distance_km", "engine_cc",
    "fuel_type", "transmission", "price_eur",
]
RAW_RENAME = {
    "Make": "make", "Model": "model", "Year": "year", "Style": "style",
    "Distance": "distance_km", "Engine_capacity(cm3)": "engine_cc",
    "Fuel_type": "fuel_type", "Transmission": "transmission", "Price(euro)": "price_eur",
}


def _car_columns() -> list[Column]:
    return [
        Column("make", String(64), nullable=False),
        Column("model", String(128), nullable=False),
        Column("year", Integer, nullable=False),
        Column("style", String(32)),
        Column("distance_km", Float),
        Column("engine_cc", Float),
        Column("fuel_type", String(32)),
        Column("transmission", String(32)),
    ]


listings_raw = Table(
    "listings_raw", metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    *_car_columns(),
    Column("price_eur", Float),
    Column("loaded_at", DateTime),
)

listings = Table(
    "listings", metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    *_car_columns(),
    Column("price_eur", Float, nullable=False),
    Column("listing_year", Integer, nullable=False),
    Index("ix_listings_make_model", "make", "model"),
    Index("ix_listings_price", "price_eur"),
)

quality_issues = Table(
    "quality_issues", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("raw_id", Integer, index=True),
    Column("rule", String(64), nullable=False),
    Column("action", String(16), nullable=False),
    Column("details", Text),
)


def _now() -> datetime:
    return datetime.now().replace(microsecond=0)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    engine = create_engine(url or config.DATABASE_URL, future=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    return engine


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or get_engine()
    if engine.dialect.name == "sqlite":
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata.create_all(engine)
    return engine


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame → список словарей с нативными типами Python (NaN → None)."""
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def _replace_table(table: Table, df: pd.DataFrame, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    columns = [c.name for c in table.columns if c.name in df.columns]
    with engine.begin() as conn:
        conn.execute(delete(table))
        if len(df):
            conn.execute(insert(table), _records(df[columns]))


# ---------- Сырые и очищенные объявления ----------

def load_raw_listings(raw: pd.DataFrame, engine: Engine | None = None) -> int:
    """Загружает исходный CSV в listings_raw (id = номер строки в файле)."""
    df = raw.rename(columns=RAW_RENAME)[LISTING_COLUMNS].copy()
    df.insert(0, "id", range(1, len(df) + 1))
    df["loaded_at"] = _now()
    _replace_table(listings_raw, df, engine)
    return len(df)


def read_table(name: str, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    table = metadata.tables[name]
    with engine.connect() as conn:
        return pd.read_sql(select(table), conn)


def replace_listings(clean: pd.DataFrame, engine: Engine | None = None) -> None:
    _replace_table(listings, clean, engine)


def replace_quality_issues(issues: pd.DataFrame, engine: Engine | None = None) -> None:
    _replace_table(quality_issues, issues, engine)


def main() -> None:
    from car_advisor.data.download import download_dataset

    raw = download_dataset()
    engine = init_db()
    n = load_raw_listings(raw, engine)
    print(f"БД: {engine.url.render_as_string(hide_password=True)} — загружено {n:,} сырых объявлений в listings_raw")


if __name__ == "__main__":
    main()

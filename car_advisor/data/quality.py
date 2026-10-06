"""Проверка качества данных и очистка объявлений.

Правила сформулированы по результатам разведочного анализа (см. notebooks/01_eda_data_quality.ipynb):
  * ~9% объявлений — точные дубликаты (повторные публикации);
  * год выпуска 1900 и пробег 0/1 км — «заглушки», которые продавцы ставят вместо реальных значений;
  * пробег 1 111 111 / 22 222 222 / 100 000 000 км — заведомо невозможные значения;
  * объём двигателя 200/220/300 «см³» — на самом деле литры × 100 (2.0 л → 200): исправляем × 10;
  * у электромобилей объём двигателя не имеет смысла — обнуляем;
  * часть цен указана, судя по всему, в леях вместо евро (KIA Sportage 2018 за 490 000 €):
    ловим групповым правилом — цена отличается от медианы той же марки/модели/года более чем в 3 раза.

Запуск: python -m car_advisor.data.quality
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from car_advisor import config
from car_advisor.data import database as db

ALLOWED_FUEL = ["Diesel", "Petrol", "Metan/Propan", "Hybrid", "Plug-in Hybrid", "Electric"]
ALLOWED_TRANSMISSION = ["Manual", "Automatic"]
ALLOWED_STYLE = [
    "Sedan", "Universal", "Hatchback", "Crossover", "SUV", "Minivan", "Coupe",
    "Microvan", "Combi", "Pickup", "Cabriolet", "Roadster",
]

PRICE_MIN, PRICE_MAX = 300, 250_000
DISTANCE_MAX = 1_000_000
PLACEHOLDER_DISTANCE, PLACEHOLDER_MIN_AGE = 1_000, 3
ENGINE_UNIT_ERROR_RANGE = (90, 500)
ENGINE_MIN, ENGINE_MAX = 500, 7_000
GROUP_RATIO_MAX, GROUP_MIN_SIZE = 3.0, 5
HOLDOUT_SHARE = 0.05

RULES: dict[str, dict[str, str]] = {
    "missing_values": {"title": "Пропуски в обязательных полях", "action": "drop"},
    "invalid_category": {"title": "Недопустимое значение категории (топливо/КПП/кузов)", "action": "drop"},
    "duplicate": {"title": "Точный дубликат объявления", "action": "drop"},
    "engine_unit_error": {"title": "Объём двигателя указан в литрах×100 (200 → 2000 см³)", "action": "repair"},
    "electric_engine": {"title": "Объём двигателя у электромобиля (обнулён)", "action": "repair"},
    "year_out_of_range": {"title": f"Год выпуска вне диапазона {config.MIN_YEAR}–{config.SNAPSHOT_YEAR}", "action": "drop"},
    "price_out_of_range": {"title": f"Цена вне диапазона {PRICE_MIN}–{PRICE_MAX:,} €", "action": "drop"},
    "mileage_out_of_range": {"title": f"Пробег больше {DISTANCE_MAX:,} км", "action": "drop"},
    "placeholder_mileage": {
        "title": f"Пробег < {PLACEHOLDER_DISTANCE} км у авто старше {PLACEHOLDER_MIN_AGE} лет (заглушка)",
        "action": "drop",
    },
    "engine_invalid": {"title": f"Объём двигателя вне {ENGINE_MIN}–{ENGINE_MAX} см³", "action": "drop"},
    "price_group_outlier": {
        "title": f"Цена отличается от медианы марки/модели/года более чем в {GROUP_RATIO_MAX:g} раза",
        "action": "drop",
    },
}


@dataclass
class QualityResult:
    clean: pd.DataFrame
    issues: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def _column_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df[["year", "distance_km", "engine_cc", "price_eur"]].describe(
        percentiles=[0.01, 0.5, 0.99]).round(1)
    return {
        "rows": int(len(df)),
        "missing": {c: int(v) for c, v in df[db.LISTING_COLUMNS].isna().sum().items()},
        "duplicates": int(df.duplicated(subset=db.LISTING_COLUMNS).sum()),
        "numeric": json.loads(numeric.to_json()),
        "cardinality": {c: int(df[c].nunique()) for c in ["make", "model", "style", "fuel_type", "transmission"]},
    }


def run_quality_checks(raw: pd.DataFrame) -> QualityResult:
    """Применяет правила качества к сырым объявлениям (колонки listings_raw, включая id)."""
    df = raw.copy()
    issues: list[dict[str, Any]] = []

    def flag(mask: pd.Series, rule: str, details: pd.Series | str = "") -> None:
        if not mask.any():
            return
        det = details[mask] if isinstance(details, pd.Series) else pd.Series(details, index=df.index[mask])
        issues.extend({"raw_id": int(i), "rule": rule, "action": RULES[rule]["action"], "details": str(d)}
                      for i, d in zip(df.loc[mask, "id"], det, strict=True))

    profile_before = _column_profile(df)

    # 1. Схема и типы
    schema_errors = [c for c in db.LISTING_COLUMNS if c not in df.columns]
    if schema_errors:
        raise ValueError(f"Нет обязательных колонок: {schema_errors}")
    for col in ["make", "model", "style", "fuel_type", "transmission"]:
        df[col] = df[col].astype("string").str.strip()
    for col in ["year", "distance_km", "engine_cc", "price_eur"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop = pd.Series(False, index=df.index)

    def drop_rows(mask: pd.Series, rule: str, details: pd.Series | str = "") -> None:
        nonlocal drop
        new = mask & ~drop
        flag(new, rule, details)
        drop |= new

    # 2. Пропуски и недопустимые категории
    drop_rows(df[db.LISTING_COLUMNS].isna().any(axis=1), "missing_values")
    bad_cat = (~df["fuel_type"].isin(ALLOWED_FUEL) | ~df["transmission"].isin(ALLOWED_TRANSMISSION)
               | ~df["style"].isin(ALLOWED_STYLE))
    drop_rows(bad_cat.fillna(True), "invalid_category",
              df["fuel_type"].astype(str) + " / " + df["transmission"].astype(str) + " / " + df["style"].astype(str))

    # 3. Дубликаты (оставляем первое объявление)
    drop_rows(df.duplicated(subset=db.LISTING_COLUMNS, keep="first"), "duplicate", "повтор уже опубликованного объявления")

    # 4. Исправления (строки остаются в выборке)
    lo, hi = ENGINE_UNIT_ERROR_RANGE
    unit_err = ~drop & (df["fuel_type"] != "Electric") & df["engine_cc"].between(lo, hi, inclusive="left")
    flag(unit_err, "engine_unit_error", df["engine_cc"].map(lambda v: f"{v:g} → {v * 10:g}"))
    df.loc[unit_err, "engine_cc"] = df.loc[unit_err, "engine_cc"] * 10

    electric = ~drop & (df["fuel_type"] == "Electric") & (df["engine_cc"] != 0)
    flag(electric, "electric_engine", df["engine_cc"].map(lambda v: f"{v:g} → 0"))
    df.loc[electric, "engine_cc"] = 0.0

    # 5. Доменные правила
    age = config.SNAPSHOT_YEAR - df["year"]
    drop_rows(~df["year"].between(config.MIN_YEAR, config.SNAPSHOT_YEAR), "year_out_of_range",
              df["year"].astype(str))
    drop_rows(~df["price_eur"].between(PRICE_MIN, PRICE_MAX), "price_out_of_range",
              df["price_eur"].map(lambda v: f"{v:,.0f} €"))
    drop_rows(df["distance_km"] > DISTANCE_MAX, "mileage_out_of_range",
              df["distance_km"].map(lambda v: f"{v:,.0f} км"))
    drop_rows((df["distance_km"] < PLACEHOLDER_DISTANCE) & (age >= PLACEHOLDER_MIN_AGE), "placeholder_mileage",
              df["distance_km"].map(lambda v: f"{v:,.0f} км") + ", год " + df["year"].astype(str))
    engine_bad = (df["fuel_type"] != "Electric") & ~df["engine_cc"].between(ENGINE_MIN, ENGINE_MAX)
    drop_rows(engine_bad, "engine_invalid", df["engine_cc"].map(lambda v: f"{v:g} см³"))

    # 6. Статистические выбросы цены внутри группы «марка + модель + год»
    kept = df[~drop]
    log_price = np.log(kept["price_eur"])
    groups = log_price.groupby([kept["make"], kept["model"], kept["year"]])
    size = groups.transform("size")
    ratio = np.exp(log_price - groups.transform("median"))
    outlier = (size >= GROUP_MIN_SIZE) & ((ratio > GROUP_RATIO_MAX) | (ratio < 1 / GROUP_RATIO_MAX))
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[outlier.index] = outlier
    ratio_text = pd.Series("", index=df.index)
    ratio_text.loc[ratio.index] = ratio.map(lambda r: f"цена = {r:.2f} × медианы группы")
    drop_rows(outlier_mask, "price_group_outlier", ratio_text)

    clean = df[~drop].copy()
    clean["year"] = clean["year"].astype(int)
    clean["listing_year"] = config.SNAPSHOT_YEAR
    for col in ["make", "model", "style", "fuel_type", "transmission"]:
        clean[col] = clean[col].astype(str)
    issues_df = pd.DataFrame(issues, columns=["raw_id", "rule", "action", "details"])

    counts = issues_df.groupby("rule").size().to_dict() if len(issues_df) else {}
    report = {
        "source": config.KAGGLE_DATASET_URL,
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean)),
        "rows_dropped": int(drop.sum()),
        "rows_repaired": int(issues_df.loc[issues_df["action"] == "repair", "raw_id"].nunique()) if len(issues_df) else 0,
        "rules": [
            {"code": code, "title": meta["title"], "action": meta["action"], "count": int(counts.get(code, 0))}
            for code, meta in RULES.items()
        ],
        "before": profile_before,
        "after": _column_profile(clean),
    }
    return QualityResult(clean=clean, issues=issues_df, report=report)


def split_holdout(clean: pd.DataFrame, share: float = HOLDOUT_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Откладывает часть объявлений как «новые» (не участвуют в обучении) — для будущей проверки
    модели на данных, которых она не видела."""
    holdout = clean.sample(frac=share, random_state=config.RANDOM_STATE)
    return clean.drop(holdout.index), holdout


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Отчёт о качестве данных",
        "",
        f"Источник: {report['source']}",
        "",
        f"* Сырых объявлений: **{report['rows_raw']:,}**",
        f"* После очистки: **{report['rows_clean']:,}** ({report['rows_clean'] / report['rows_raw']:.1%})",
        f"* Удалено: **{report['rows_dropped']:,}**, исправлено: **{report['rows_repaired']:,}**",
        "",
        "| Правило | Действие | Строк |",
        "|---|---|---:|",
    ]
    action_ru = {"drop": "удаление", "repair": "исправление"}
    lines += [f"| {r['title']} | {action_ru[r['action']]} | {r['count']:,} |" for r in report["rules"]]
    lines += ["", "## Пропуски до очистки", ""]
    lines += [f"* {k}: {v}" for k, v in report["before"]["missing"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    engine = db.init_db()
    raw = db.read_table("listings_raw", engine)
    if raw.empty:
        raise SystemExit("listings_raw пуста — сначала выполните: python -m car_advisor.data.database")

    result = run_quality_checks(raw)
    market, holdout = split_holdout(result.clean)
    result.report["rows_market"] = int(len(market))
    result.report["rows_holdout"] = int(len(holdout))

    db.replace_listings(market, engine)
    db.replace_quality_issues(result.issues, engine)

    config.NEW_LISTINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    holdout.drop(columns=["loaded_at"], errors="ignore").to_csv(config.NEW_LISTINGS_CSV, index=False)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.QUALITY_REPORT_JSON.write_text(json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8")
    config.QUALITY_REPORT_MD.write_text(report_to_markdown(result.report), encoding="utf-8")

    print(report_to_markdown(result.report))
    print(f"В БД (listings): {len(market):,}; отложено как новые объявления: {len(holdout):,} → {config.NEW_LISTINGS_CSV}")


if __name__ == "__main__":
    main()

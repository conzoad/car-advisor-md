"""Разбиение train/test и первые метрики (бейзлайн этапа 1).

Три уровня сравнения:
  1. константа — медианная цена train: «модель», которая ничего не знает об автомобиле;
  2. медиана похожих объявлений — так оценивает цену покупатель без ML;
  3. Ridge-регрессия на log(цены) с полным препроцессингом (one-hot, target encoding, сплайны, масштабирование).

Протокол такой же, как будет у основных моделей: train/test = 80/20 со стратификацией по децилям цены,
5-fold кросс-валидация на train, финальная проверка на test.

Запуск: python -m car_advisor.ml.baseline
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split

from car_advisor import config, viz
from car_advisor.data import database as db
from car_advisor.ml.features import INPUT_COLUMNS, FeatureBuilder
from car_advisor.ml.models import MODEL_TITLES, ridge_model

TEST_SIZE = 0.2
N_FOLDS = 5
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
}
# Уровни поиска «похожих» объявлений: от самого точного к самому общему
GROUP_LEVELS = (("model", "age"), ("model",), ("make",))


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    """Медиана цены похожих объявлений из train: та же модель того же возраста → та же модель →
    та же марка → весь рынок. Группа используется, если в ней не меньше min_count объявлений."""

    def __init__(self, min_count: int = 3):
        self.min_count = min_count

    def fit(self, X, y):
        F = FeatureBuilder().transform(X).assign(y=np.asarray(y, float))
        self.tables_ = []
        for keys in GROUP_LEVELS:
            g = F.groupby(list(keys))["y"].agg(["median", "size"])
            self.tables_.append(g.loc[g["size"] >= self.min_count, "median"])
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X):
        F = FeatureBuilder().transform(X)
        pred = pd.Series(np.nan, index=F.index)
        for keys, table in zip(GROUP_LEVELS, self.tables_):
            lookup = pd.MultiIndex.from_frame(F[list(keys)]) if len(keys) > 1 else pd.Index(F[keys[0]])
            pred = pred.fillna(pd.Series(table.reindex(lookup).to_numpy(), index=F.index))
        return pred.fillna(self.global_median_).to_numpy()


BASELINES = {
    "dummy_median": ("Константа: медианная цена", lambda: DummyRegressor(strategy="median")),
    "group_median": ("Медиана похожих объявлений", GroupMedianRegressor),
    "ridge": (MODEL_TITLES["ridge"], ridge_model),
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(ape.mean()),
        "medape": float(np.median(ape)),
        "within_10pct": float((ape <= 0.10).mean()),
        "within_20pct": float((ape <= 0.20).mean()),
    }


def load_market() -> tuple[pd.DataFrame, pd.Series]:
    market = db.read_table("listings")
    if market.empty:
        raise SystemExit("Таблица listings пуста — сначала выполните: make data")
    return market[INPUT_COLUMNS + ["listing_year"]], market["price_eur"].astype(float)


def split(X: pd.DataFrame, y: pd.Series):
    """80/20 со стратификацией по децилям цены: дешёвые и дорогие авто попадают в test в той же пропорции."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=config.RANDOM_STATE,
                            stratify=pd.qcut(y, 10, labels=False))


def evaluate(X_train, X_test, y_train, y_test, n_folds: int = N_FOLDS) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = KFold(n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}
    for name, (title, make) in BASELINES.items():
        cv = cross_validate(make(), X_train, y_train, cv=folds, scoring=SCORING)
        model = make().fit(X_train, y_train)
        test = regression_metrics(y_test, model.predict(X_test))
        rows.append({
            "name": name, "title": title,
            "cv_mae": -cv["test_mae"].mean(), "cv_mae_std": cv["test_mae"].std(),
            "cv_rmse": -cv["test_rmse"].mean(), "cv_r2": cv["test_r2"].mean(), "cv_mape": -cv["test_mape"].mean(),
            **{f"test_{k}": v for k, v in test.items()},
        })
        fitted[name] = model
    return pd.DataFrame(rows), fitted


def plot_baselines(comparison: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    data = comparison.iloc[::-1]
    colors = [viz.SERIES[0] if n == "ridge" else viz.AXIS for n in data["name"]]
    panels = [(axes[0], data["test_mae"], "MAE на test, €", "{:,.0f} €"),
              (axes[1], data["test_mape"] * 100, "MAPE на test, %", "{:.1f}%")]
    for ax, values, title, fmt in panels:
        ax.barh(data["title"], values, color=colors, height=0.55)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for i, v in enumerate(values):
            ax.text(v, i, " " + fmt.format(v).replace(",", " "), va="center", fontsize=9, color=viz.INK_SECONDARY)
        ax.margins(x=0.2)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    return fig


def main() -> dict[str, Any]:
    viz.use_matplotlib_style()
    X, y = load_market()
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Данные: {len(X):,} объявлений → train {len(X_train):,} / test {len(X_test):,}")
    comparison, _ = evaluate(X_train, X_test, y_train, y_test)
    for row in comparison.itertuples():
        print(f"  {row.title:38s} CV MAE {row.cv_mae:7,.0f} € ± {row.cv_mae_std:4,.0f}  "
              f"test MAE {row.test_mae:7,.0f} €  MAPE {row.test_mape:.1%}  R² {row.test_r2:.3f}  "
              f"±20%: {row.test_within_20pct:.0%}")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_baselines(comparison)
    figure = config.FIGURES_DIR / "baseline_comparison.png"
    fig.savefig(figure)
    plt.close(fig)
    report = {
        "rows_market": int(len(X)), "rows_train": int(len(X_train)), "rows_test": int(len(X_test)),
        "test_size": TEST_SIZE, "cv_folds": N_FOLDS, "random_state": config.RANDOM_STATE,
        "stratify": "децили цены", "selection_metric": "MAE",
        "models": comparison.round(4).to_dict(orient="records"),
        "figure": str(figure.relative_to(config.ROOT)),
    }
    config.BASELINE_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {config.BASELINE_REPORT_JSON.relative_to(config.ROOT)}")
    return report


if __name__ == "__main__":
    main()

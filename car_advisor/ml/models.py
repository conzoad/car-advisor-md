"""Модели оценки цены. Обучаются на log(цены): ошибки становятся относительными,
а распределение цены — почти симметричным. На выходе TransformedTargetRegressor возвращает евро.

На этапе 1 здесь только линейный бейзлайн; Random Forest и градиентный бустинг — следующий этап."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler, TargetEncoder

from car_advisor.config import RANDOM_STATE
from car_advisor.ml.features import FeatureBuilder

MODEL_TITLES = {
    "ridge": "Ridge-регрессия (сплайны + one-hot)",
}


def _target_encoder() -> TargetEncoder:
    return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))


def _log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp, check_inverse=False)


def ridge_preprocessor() -> ColumnTransformer:
    """Кодирование категорий и масштабирование числовых признаков для линейной модели.

    * марка, кузов, топливо, КПП — one-hot; редкие значения (< 10 объявлений) объединяются в одну колонку;
    * модель (≈900 значений) — target encoding с кросс-фиттингом (5 фолдов), чтобы не было утечки целевой;
    * возраст и log(пробега) — сплайны: цена зависит от них нелинейно;
    * объём двигателя и log(пробега в год) — стандартизация.
    """
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    return ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
         ["make", "style", "fuel_type", "transmission"]),
        ("te", _target_encoder(), ["model"]),
        ("age", SplineTransformer(n_knots=8, degree=3), ["age"]),
        ("dist", Pipeline([("log", log1p), ("spline", SplineTransformer(n_knots=6, degree=3))]), ["distance_km"]),
        ("engine", StandardScaler(), ["engine_cc"]),
        ("kmy", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), ["km_per_year"]),
    ])


def ridge_model() -> TransformedTargetRegressor:
    """Линейный бейзлайн: нелинейность возраста и пробега — через сплайны, модель — target encoding."""
    return _log_target(Pipeline([("features", FeatureBuilder()), ("prep", ridge_preprocessor()),
                                 ("model", Ridge(alpha=1.0))]))

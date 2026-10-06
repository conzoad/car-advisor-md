"""Препроцессинг: из полей объявления строим признаки, от которых зависит цена (по выводам EDA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from car_advisor.config import SNAPSHOT_YEAR

# Поля объявления об автомобиле (+ необязательный listing_year — год объявления)
INPUT_COLUMNS = ["make", "model", "year", "style", "distance_km", "engine_cc", "fuel_type", "transmission"]

CAT_FEATURES = ["make", "model", "style", "fuel_type", "transmission"]
NUM_FEATURES = ["age", "distance_km", "engine_cc", "km_per_year"]
MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES

FEATURE_LABELS = {
    "make": "Марка",
    "model": "Модель",
    "style": "Тип кузова",
    "fuel_type": "Топливо",
    "transmission": "Коробка передач",
    "age": "Возраст",
    "distance_km": "Пробег",
    "engine_cc": "Объём двигателя",
    "km_per_year": "Пробег в год",
}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Поля объявления → признаки модели (без обучаемых параметров).

    * возраст = год объявления − год выпуска: модель будет учить зависимость цены от возраста,
      а не от календарного года, поэтому ей можно пользоваться и после 2021 года;
    * модель = «марка + модель» (одинаковые названия моделей у разных марок не смешиваются);
    * пробег в год — интенсивность эксплуатации.
    """

    def __init__(self, category_dtype: bool = False):
        self.category_dtype = category_dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X)
        if "listing_year" in X.columns:
            listing_year = pd.to_numeric(X["listing_year"]).fillna(SNAPSHOT_YEAR)
        else:
            listing_year = SNAPSHOT_YEAR
        age = (listing_year - pd.to_numeric(X["year"])).clip(lower=0).astype(float)
        distance = pd.to_numeric(X["distance_km"]).astype(float)
        out = pd.DataFrame({
            "make": X["make"].astype(str),
            "model": X["make"].astype(str) + " " + X["model"].astype(str),
            "style": X["style"].astype(str),
            "fuel_type": X["fuel_type"].astype(str),
            "transmission": X["transmission"].astype(str),
            "age": age,
            "distance_km": distance,
            "engine_cc": pd.to_numeric(X["engine_cc"]).astype(float),
            "km_per_year": distance / (age + 1),
        }, index=X.index)
        if self.category_dtype:
            out[CAT_FEATURES] = out[CAT_FEATURES].astype("category")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_FEATURES, dtype=object)

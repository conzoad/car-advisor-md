import numpy as np
import pandas as pd

from car_advisor.ml.baseline import GroupMedianRegressor, regression_metrics
from car_advisor.ml.models import ridge_model

BASE = {"make": "Toyota", "model": "Corolla", "style": "Sedan", "distance_km": 150_000.0, "engine_cc": 1600.0,
        "fuel_type": "Petrol", "transmission": "Manual", "listing_year": 2021}


def make_market(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический рынок: цена = уровень марки × амортизация по возрасту × шум."""
    rng = np.random.default_rng(seed)
    makes = rng.choice(["Toyota", "BMW", "Dacia"], n)
    year = rng.integers(2000, 2021, n)
    X = pd.DataFrame({**BASE, "make": makes, "model": np.where(makes == "BMW", "X5", "Base"), "year": year,
                      "distance_km": rng.uniform(20_000, 300_000, n),
                      "transmission": rng.choice(["Manual", "Automatic"], n)})
    level = pd.Series(makes).map({"Toyota": 9.0, "BMW": 9.6, "Dacia": 8.5}).to_numpy()
    return X, pd.Series(np.exp(level - 0.08 * (2021 - year) + rng.normal(0, 0.05, n)))


def test_group_median_falls_back_to_broader_groups():
    train = pd.DataFrame([dict(BASE, year=2011)] * 3 + [dict(BASE, year=2016)]
                         + [dict(BASE, make="Dacia", model="Logan", year=2011)] * 3)
    y = [5000, 6000, 7000, 9000, 3000, 3000, 3000]
    model = GroupMedianRegressor(min_count=3).fit(train, y)
    new = pd.DataFrame([
        dict(BASE, year=2011),                              # та же модель и возраст: медиана 3 объявлений
        dict(BASE, year=2016),                              # в группе 1 объявление → медиана модели
        dict(BASE, model="Yaris", year=2011),               # модели нет в train → медиана марки
        dict(BASE, make="Lada", model="Niva", year=2011),   # марки нет в train → медиана рынка
    ])
    assert model.predict(new).tolist() == [6000, 6500, 6500, 5000]


def test_ridge_baseline_learns_price_and_handles_unseen_category():
    X, y = make_market()
    model = ridge_model().fit(X, y)
    assert regression_metrics(y, model.predict(X))["mape"] < 0.15
    unseen = model.predict(pd.DataFrame([dict(BASE, make="Lada", model="Niva", year=2015)]))
    assert np.isfinite(unseen).all() and (unseen > 0).all()


def test_regression_metrics():
    m = regression_metrics([100, 200, 400], [110, 150, 400])
    assert np.isclose(m["mae"], 20)
    assert np.isclose(m["mape"], (0.1 + 0.25 + 0) / 3)
    assert np.isclose(m["within_20pct"], 2 / 3)

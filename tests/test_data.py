import numpy as np
import pandas as pd

from car_advisor.data.quality import run_quality_checks
from car_advisor.ml.features import FeatureBuilder


def make_raw() -> pd.DataFrame:
    base = {"make": "Toyota", "model": "Corolla", "year": 2012, "style": "Sedan", "distance_km": 150_000.0,
            "engine_cc": 1600.0, "fuel_type": "Petrol", "transmission": "Manual", "price_eur": 7000.0}
    rows = [dict(base, price_eur=p) for p in (6800, 7000, 7100, 7300, 6900)]      # нормальная группа
    rows += [dict(base, price_eur=70_000)]                                         # цена в 10 раз выше группы
    rows += [dict(base, price_eur=6800)]                                           # точный дубликат
    rows += [dict(base, year=1900)]                                                # заглушка года
    rows += [dict(base, price_eur=1)]                                              # заглушка цены
    rows += [dict(base, distance_km=100_000_000)]                                  # невозможный пробег
    rows += [dict(base, distance_km=1)]                                            # пробег-заглушка у старого авто
    rows += [dict(base, model="Auris", engine_cc=200)]                             # литры × 100
    rows += [dict(base, model="Leaf", fuel_type="Electric", engine_cc=367)]        # объём у электромобиля
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def test_quality_rules_drop_and_repair():
    result = run_quality_checks(make_raw())
    counts = {r["code"]: r["count"] for r in result.report["rules"]}
    assert counts["duplicate"] == 1
    assert counts["year_out_of_range"] == 1
    assert counts["price_out_of_range"] == 1
    assert counts["mileage_out_of_range"] == 1
    assert counts["placeholder_mileage"] == 1
    assert counts["price_group_outlier"] == 1
    assert counts["engine_unit_error"] == 1 and counts["electric_engine"] == 1

    clean = result.clean.set_index("model")
    assert clean.loc["Auris", "engine_cc"] == 2000
    assert clean.loc["Leaf", "engine_cc"] == 0
    assert len(result.clean) == 7
    assert (result.clean["listing_year"] == 2021).all()


def test_feature_builder_age_and_intensity():
    X = pd.DataFrame([{"make": "BMW", "model": "X5", "year": 2011, "style": "SUV", "distance_km": 220_000,
                       "engine_cc": 3000, "fuel_type": "Diesel", "transmission": "Automatic", "listing_year": 2021}])
    out = FeatureBuilder().transform(X)
    assert out.loc[0, "age"] == 10
    assert out.loc[0, "model"] == "BMW X5"
    assert np.isclose(out.loc[0, "km_per_year"], 20_000)
    # без listing_year возраст считается от года среза данных (2021)
    assert FeatureBuilder().transform(X.drop(columns="listing_year")).loc[0, "age"] == 10


import shutil

import pytest

from car_advisor import config
from car_advisor.data import database as db


@pytest.fixture(autouse=True, scope="session")
def isolated_database(tmp_path_factory):
    """Тесты работают с копией БД, а не с рабочей базой."""
    source = config.DATA_DIR / "cars.db"
    tmp = tmp_path_factory.mktemp("db") / "cars.db"
    for suffix in ("", "-wal", "-shm"):  # в режиме WAL свежие данные могут быть ещё в журнале
        if source.with_name(source.name + suffix).exists():
            shutil.copy(source.with_name(source.name + suffix), tmp.with_name(tmp.name + suffix))
    original = config.DATABASE_URL
    config.DATABASE_URL = config._sqlite_url(tmp)
    db.get_engine.cache_clear()
    db.init_db()
    yield
    config.DATABASE_URL = original
    db.get_engine.cache_clear()

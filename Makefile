PY ?= .venv/bin/python

.PHONY: setup download data test notebooks

setup:            ## виртуальное окружение и зависимости
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

download:         ## загрузка датасета (Kaggle API)
	$(PY) -m car_advisor.data.download

data:             ## загрузка датасета → БД → проверка качества
	$(PY) -m car_advisor.data.database
	$(PY) -m car_advisor.data.quality

test:
	$(PY) -m pytest -q

notebooks:        ## пересчитать ноутбук с результатами
	cd notebooks && ../$(PY) -m jupyter nbconvert --to notebook --execute --inplace *.ipynb

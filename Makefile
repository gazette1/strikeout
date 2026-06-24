.PHONY: install backfill predict backtest test dashboard clean

install:
	pip install -e ".[dev]"

backfill:
	python -m src.data.backfill --start-year 2022 --end-year 2025

backfill-dev:
	python -m src.data.backfill --start-year 2024 --end-year 2024 --start-month 7 --end-month 7

predict:
	python -m src.pipeline.daily_runner --date $(DATE)

backtest:
	python -m src.evaluation.walk_forward

blind-test:
	python -m src.evaluation.blind_test

train:
	python -m src.model.train

tune:
	python -m src.model.hyperparameter_tuning

dashboard:
	streamlit run src/dashboard/streamlit_app.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

setup:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

score:
	python scripts/precompute_scores.py

run-api:
	uvicorn src.api.main:app --port 8000

dashboard:
	streamlit run src/dashboard/app.py

backtest:
	python -m src.backtest.run


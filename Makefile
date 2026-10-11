setup:
	pip install -r requirements.txt

data:
	python -m scripts.download_data

score:
	python -m scripts.precompute_scores

run-api:
	uvicorn src.api.main:app --port 8000

dashboard:
	streamlit run src/dashboard/app.py

backtest:
	python -m src.backtest.run


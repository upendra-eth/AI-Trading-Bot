# AI Trading Bot

A local paper-trading system for Indian stocks using an ensemble of XGBoost, LSTM, and FinBERT with a FastAPI backend and a static dashboard frontend.

## Features

- Live scanner for NSE watchlist symbols
- Ensemble AI signals (XGBoost + LSTM + FinBERT)
- Historical backtesting with trade logs
- Paper trading engine with scheduled cycles
- SQLite-backed portfolio, trades, and opportunities

## Tech Stack

- Python
- FastAPI
- SQLAlchemy + SQLite
- PyTorch + Transformers
- XGBoost + scikit-learn
- Vanilla HTML/CSS/JS frontend (`static/`)

## Project Structure

- `app.py` - FastAPI app and paper trading endpoints
- `backtest.py` - backtesting logic
- `engine.py` - trading engine helpers
- `models.py` - model and ensemble logic
- `features.py` - technical indicators
- `data_fetcher.py` - market/news data fetchers
- `database.py` - database models and session init
- `static/` - dashboard frontend

## Local Setup

1. Create and activate a virtual environment:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the API server:
   - `python app.py`
4. Open:
   - `http://127.0.0.1:8000`

## Tests

- `pytest`

## GitHub Pages Deployment

This repository includes a GitHub Actions workflow that publishes the `static/` folder to GitHub Pages.

After pushing:

1. Open repository Settings -> Pages.
2. Ensure Source is set to **GitHub Actions**.
3. Wait for the "Deploy static site to Pages" workflow to complete.
4. Your site will be available at:
   - `https://upendra-eth.github.io/AI-Trading-Bot/`

## Notes

- `venv/`, local DB files, and cache artifacts are ignored via `.gitignore`.
- The GitHub Pages deployment serves only static files. FastAPI endpoints are not hosted on GitHub Pages.

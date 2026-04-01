# AIQuant — Complete Project Documentation

> **A Local Indian Stock Market Paper Trading System** powered by ML ensemble signals (XGBoost + LSTM + FinBERT), built on FastAPI, SQLite, and Lightweight Charts.

---

## 1. System Architecture

```
┌──────────────┐    HTTP     ┌──────────────────────────────────────┐
│   Browser    │ ◄─────────► │  FastAPI (app.py)  port 8000         │
│  index.html  │             │  ├─ /api/portfolio                    │
│  script.js   │             │  ├─ /api/scan                        │
│  style.css   │             │  ├─ /api/backtest                    │
└──────────────┘             │  ├─ /api/paper-trading/start         │
                             │  ├─ /api/paper-trading/stop          │
                             │  ├─ /api/paper-trading/run-now       │
                             │  ├─ /api/paper-trading/status        │
                             │  └─ /api/paper-trading/reset         │
                             └──────┬──────────────┬────────────────┘
                                    │              │
                        ┌───────────▼──┐    ┌──────▼────────────────┐
                        │  models.py   │    │    database.py         │
                        │  XGBoost     │    │    SQLite (trading.db) │
                        │  LSTM        │    │    Portfolio           │
                        │  FinBERT     │    │    Trades              │
                        └──────────────┘    │    Opportunities       │
                                            └───────────────────────┘
                                    │
                        ┌───────────▼──────────────────────────┐
                        │  External Data Sources                │
                        │  Yahoo Finance (yfinance)             │
                        │  Google News RSS Feed                 │
                        └──────────────────────────────────────┘
```

---

## 2. Data Sources & APIs

| Source | What it provides | How it's used |
|--------|-----------------|---------------|
| **Yahoo Finance** (`yfinance`) | OHLCV price data | Fetches 1-year daily data for the scanner. 60-day max for 15m, 730-day for 1h |
| **Google News RSS** (`feedparser`) | Latest news headlines per stock | Free RSS feed. No API key needed. Fetches top 10 articles per symbol |
| **HuggingFace** (`transformers`) | FinBERT NLP model weights | Auto-downloaded on first run. Runs 100% locally, no API calls after download |
| **SQLite** (`SQLAlchemy`) | Persistent storage | Stores portfolio balance, open/closed trades, signal opportunities |

**NSE Watchlist (default):** RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS

---

## 3. File Structure

| File | Role |
|------|------|
| [app.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py) | Main FastAPI server + paper trading background engine |
| [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | XGBoost, LSTM, FinBERT model definitions + EnsembleStrategy |
| [backtest.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/backtest.py) | Historical simulation engine |
| [engine.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/engine.py) | Single-cycle paper trading logic (called by app.py) |
| [features.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/features.py) | Technical indicator calculation (SMA, EMA, RSI, MACD, etc.) |
| [data_fetcher.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/data_fetcher.py) | Yahoo Finance + Google News RSS wrappers |
| [database.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/database.py) | SQLAlchemy models: Portfolio, Trade, Opportunity |
| [static/index.html](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/index.html) | Frontend: Dashboard, Scanner, Backtester, Paper Trading tabs |
| [static/script.js](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js) | All frontend logic: charting, API calls, paper trading controls |
| [static/style.css](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/style.css) | Dark glassmorphic design system |

---

## 4. The Three AI Models (Plain English)

### 4.1 XGBoost (Weight: 35%)

**What it is:** A decision-tree based ML model. Think of it as a very advanced *"pattern matcher"* that has read thousands of past days of price action.

**How it trains:**
1. Takes 1 year of daily OHLCV data → adds 19 technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Creates the "question": *"Given today's indicators, what will tomorrow's % change be?"*
3. Trains 200 decision trees on the last 365 days of data

**How it predicts:**
- Runs the current indicators through the trained model
- Gets a predicted % change for tomorrow (e.g., `+0.5%` or `-0.3%`)
- Converts to signal: `> +0.5%` = BUY, `< -0.5%` = SELL, else HOLD

**Technical indicators used (19):**
`SMA_20`, `SMA_50`, `EMA_9`, `EMA_21`, `RSI_14`, `STOCH_RSI_K`, `STOCH_RSI_D`, `MACD`, `MACD_Histogram`, `MACD_Signal`, `BB_LOWER`, `BB_MID`, `BB_UPPER`, `ATR_14`, `OBV`, `VWAP`, `DIST_SMA20`, `DIST_SMA50`, `BB_WIDTH`

---

### 4.2 LSTM (Weight: 35%)

**What it is:** A type of neural network specifically designed for *sequences of time-series data*. It "remembers" patterns across a window of days.

**How it trains:**
1. Uses the last 20 days of data as one "sequence" (seq_length=20)
2. Creates hundreds of overlapping 20-day windows from 1 year of history
3. For each window: the input is the 20 days of all indicators, the output is the closing price of day 21
4. Trains for up to 10 epochs with early stopping

**How it predicts:**
- Feeds the most recent 20 days into the trained network
- Gets back a predicted "next close" on a scaled 0–1 range
- Compares prediction vs current price on the same scale
- `delta > 0.001` = BULLISH, `delta < -0.001` = BEARISH, else NEUTRAL

**Architecture:** `Input → LSTM(64 units, 2 layers) → Linear → Output`

---

### 4.3 FinBERT (Weight: 30%)

**What it is:** A BERT-based NLP model pre-trained specifically on financial news text. It reads news headlines and says: "Is this good news, bad news, or neutral for the stock?"

**How it works:**
1. Fetches up to 10 latest Google News RSS headlines for the stock
2. Passes each headline through FinBERT (`ProsusAI/finbert`)
3. Gets 3 probabilities: POSITIVE, NEGATIVE, NEUTRAL (they add up to 100%)
4. Score = `positive_prob - negative_prob` (range: -1.0 to +1.0)
5. Averages all headline scores → final sentiment score

**Live Scanner:** Uses real-time RSS headlines (current sentiment)
**Backtester:** Cannot fetch historical news. Uses a calculated simulation based on the `finbert_weight` parameter passed in.

> [!NOTE]
> FinBERT requires ~500MB of model weights downloaded from HuggingFace on first run.

---

### 4.4 Ensemble Combination (How all 3 models merge)

```
Final Score = (XGBoost_score × 0.35) + (LSTM_score × 0.35) + (FinBERT_score × 0.30)

If Final Score > +0.7  →  BUY
If Final Score < -0.7  →  SELL
Else                   →  HOLD
```

**Example output:** `XGB:SELL(35%) + LSTM:BUY(35%) + FinBERT:HOLD(30%) = +0.00 → HOLD`

---

## 5. Implemented Features (✅ Live)

### 5.1 Portfolio Dashboard
- Real-time paper trading balance display
- Active open positions table (symbol, entry price, quantity)
- Recent closed trades table (entry, exit, PnL)

### 5.2 Live Scanner
- Runs all 3 models on 5 NSE stocks simultaneously
- Shows per-model breakdown: score, direction, confidence, data points
- Displays all news headlines with individual sentiment scores (positive%, negative%, neutral%)
- Full ensemble reasoning equation shown

### 5.3 Backtester
- Tests the strategy on historical data (daily, 1h, 15m timeframes)
- Shows **Candlestick chart** (OHLCV) with BUY▲ and SELL▼ markers
- Chart watermark shows symbol + timeframe
- SELL markers display exact P&L amount
- Detailed unified trade ledger showing:
  - Entry Date, Exit Date, Duration
  - Entry Price, Exit Price
  - P&L (color-coded ✅/❌)
  - AI reasoning equation
  - Per-model signals

### 5.4 Paper Trading Engine
- **Start/Stop**: Runs a background thread cycling every N minutes
- **Run Now**: Triggers an immediate manual cycle
- **Reset**: Returns balance to ₹1,00,000
- **Configurable Interval**: 15 min / 30 min / 1 hour / 4 hours
- **Status Badge**: OFFLINE / ACTIVE (Waiting) / RUNNING
- **Position Sizing**: 20% of available balance per BUY
- **Risk Management**: 1 open position per symbol max
- **Cycle Log**: Color-coded execution log (green=BUY, yellow=SELL, red=ERROR)
- **Auto-refresh**: Portfolio and status poll every 10 seconds when tab is open

---

## 6. TODO / Planned Features

| Feature | Priority | Complexity |
|---------|----------|------------|
| Stop-Loss / Take-Profit orders | High | Medium |
| Email/push notification on trade execution | Medium | Medium |
| Per-symbol configurable position size | Medium | Low |
| Historical news API integration for real backtest sentiment | High | Hard |
| More NSE symbols in watchlist | Low | Easy |
| Multi-stock portfolio equity curve on Dashboard | Medium | Medium |
| Model accuracy tracking over time | Medium | Medium |
| Intraday paper trading (using 15m/1h data in engine) | High | Hard |
| Export trade history to CSV | Low | Easy |
| Mobile-responsive UI | Low | Medium |

---

## 7. How Each Function Works (Plain English)

| Function | File | What it does simply |
|----------|------|---------------------|
| [run_paper_trading_cycle()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py#33-123) | [app.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py) | Goes through each stock, scans it, and if the AI says BUY, spends 20% of your cash. If SELL, cashes out. |
| [paper_trading_loop()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py#125-154) | [app.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py) | Background thread that keeps calling [run_paper_trading_cycle()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/app.py#33-123) every N minutes then sleeps. |
| [run_backtest()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/backtest.py#12-214) | [backtest.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/backtest.py) | Replays history day by day. On each day, the AI "sees" only past data, makes a decision, and we track the fake portfolio. |
| `EnsembleStrategy.get_signal()` | [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | Combines scores from all 3 models with weights. Returns BUY/SELL/HOLD with full explanation string. |
| `EnsembleStrategy.train_models()` | [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | Trains XGBoost and LSTM on the historical data you gave it. Must be called before [get_signal()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py#352-425). |
| `XGBoostModel.train()` | [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | Learns what features (RSI, MACD, etc.) predict tomorrow's price movement from 1 year of data. |
| `LSTMModel.train()` | [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | Learns time-series patterns by looking at sliding 20-day windows of price+indicator sequences. |
| `SentimentModel.analyze()` | [models.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/models.py) | Passes news headlines through FinBERT and averages the positive/negative scores. |
| [add_technical_indicators()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/features.py#4-70) | [features.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/features.py) | Takes raw OHLCV data, adds 19 calculated columns like RSI, MACD, Bollinger Bands etc. |
| [fetch_historical_data()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/data_fetcher.py#5-20) | [data_fetcher.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/data_fetcher.py) | Calls Yahoo Finance (free, no key needed) and returns a DataFrame of price data. |
| [fetch_news_sentiment_raw()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/data_fetcher.py#21-35) | [data_fetcher.py](file:///Users/upendrasingh/data/My-Learnings/ai-trading/data_fetcher.py) | Reads Google News RSS feed for the stock and returns 10 headline strings. |
| [loadPortfolio()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js#19-53) | [script.js](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js) | Calls `/api/portfolio` and updates the Dashboard tab tables on the screen. |
| [runPaperNow()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js#459-479) | [script.js](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js) | Calls `/api/paper-trading/run-now`. Shows a loading message then updates cycle log with results. |
| [loadPaperStatus()](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js#352-425) | [script.js](file:///Users/upendrasingh/data/My-Learnings/ai-trading/static/script.js) | Polls every 10 seconds. Updates engine badge color, portfolio numbers, open/closed trade tables. |

---

## 8. Technical Indicators Reference

| Indicator | What it measures |
|-----------|-----------------|
| SMA_20 / SMA_50 | Average price over 20 or 50 days (trend direction) |
| EMA_9 / EMA_21 | Exponential average (more weight on recent prices, reacts faster) |
| RSI_14 | 0-100. Above 70 = potentially overbought, below 30 = oversold |
| STOCH_RSI | RSI applied to RSI — even faster momentum signal |
| MACD | Difference between 12-day and 26-day EMA (momentum crossover) |
| MACD_Signal | 9-day average of MACD (trigger line) |
| MACD_Histogram | MACD minus Signal (shows acceleration) |
| BB_UPPER/MID/LOWER | Bollinger Bands: ±2 standard deviations from 20-day SMA |
| BB_WIDTH | Upper - Lower (measures volatility) |
| ATR_14 | Average True Range over 14 days (pure volatility measure) |
| OBV | On-Balance Volume: running total of volume. Trend confirmation |
| VWAP | Volume-Weighted Average Price (institutional reference price) |
| DIST_SMA20 | % distance of current price from SMA_20 |
| DIST_SMA50 | % distance of current price from SMA_50 |

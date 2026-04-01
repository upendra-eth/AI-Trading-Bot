from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime
import threading
import time
import os

from database import init_db, Portfolio, Trade, Opportunity
from backtest import run_backtest, _ensemble
from data_fetcher import fetch_historical_data, fetch_news_sentiment_raw
from features import add_technical_indicators

# ─── Paper Trading Engine State ──────────────────────────────────────────────────

WATCHLIST = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
TRADE_ALLOCATION_PCT = 0.20   # Use 20% of remaining balance per position
PAPER_TRADING_ENABLED = False  # start as OFF, user toggles via API
_paper_trading_thread = None
_paper_trading_lock = threading.Lock()

paper_engine_status = {
    "running": False,
    "last_run": None,
    "next_run": None,
    "interval_minutes": 30,
    "last_cycle_log": [],
}

def run_paper_trading_cycle():
    """Runs one full cycle of paper trading across the watchlist."""
    Session = init_db()
    session = Session()
    cycle_log = []

    try:
        portfolio = session.query(Portfolio).first()
        cycle_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle started. Balance: ₹{portfolio.balance:,.0f}")

        for symbol in WATCHLIST:
            try:
                df = fetch_historical_data(symbol, period='1y', interval='1d')
                if df is None or df.empty:
                    cycle_log.append(f"  {symbol}: No data, skipping.")
                    continue

                df = add_technical_indicators(df)
                if df is None or df.empty or len(df) < 50:
                    cycle_log.append(f"  {symbol}: Not enough indicators, skipping.")
                    continue

                from backtest import _ensemble
                _ensemble.train_models(df)
                news = fetch_news_sentiment_raw(symbol)
                signal_data = _ensemble.get_signal(df, news)
                current_price = float(df['Close'].iloc[-1])

                signal = signal_data['final_signal']
                explanation = signal_data.get('explanation', '')
                cycle_log.append(f"  {symbol} @ ₹{current_price:.2f} → {signal} ({explanation})")

                # Log opportunity
                opp = Opportunity(
                    symbol=symbol,
                    xgb_signal=signal_data.get('xgb_signal', 0),
                    lstm_signal=signal_data.get('lstm_signal', 0),
                    finbert_signal=signal_data.get('finbert_signal', 0),
                    final_signal=signal,
                    executed=False
                )
                session.add(opp)

                active_trade = session.query(Trade).filter_by(symbol=symbol, status='OPEN').first()

                if signal == 'BUY' and not active_trade:
                    alloc = portfolio.balance * TRADE_ALLOCATION_PCT
                    quantity = int(alloc / current_price)
                    cost = quantity * current_price
                    if quantity > 0 and portfolio.balance >= cost:
                        new_trade = Trade(
                            symbol=symbol,
                            entry_time=datetime.now(),
                            entry_price=current_price,
                            quantity=quantity,
                            status='OPEN'
                        )
                        portfolio.balance -= cost
                        opp.executed = True
                        session.add(new_trade)
                        cycle_log.append(f"    → BUY EXECUTED: {quantity} shares @ ₹{current_price:.2f} (Cost: ₹{cost:,.0f})")
                    else:
                        cycle_log.append(f"    → BUY SKIPPED: Insufficient balance or quantity=0")

                elif signal == 'SELL' and active_trade:
                    revenue = current_price * active_trade.quantity
                    pnl = revenue - (active_trade.entry_price * active_trade.quantity)
                    active_trade.exit_time = datetime.now()
                    active_trade.exit_price = current_price
                    active_trade.pnl = pnl
                    active_trade.status = 'CLOSED'
                    portfolio.balance += revenue
                    opp.executed = True
                    cycle_log.append(f"    → SELL EXECUTED: {active_trade.quantity} shares @ ₹{current_price:.2f} | PnL: {'+'if pnl>=0 else ''}₹{pnl:,.0f}")
                else:
                    cycle_log.append(f"    → HOLD (no action)")

            except Exception as sym_err:
                cycle_log.append(f"  {symbol}: ERROR - {str(sym_err)}")
                continue

        session.commit()
        cycle_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete. Balance: ₹{portfolio.balance:,.0f}")

    except Exception as e:
        cycle_log.append(f"CYCLE ERROR: {str(e)}")
    finally:
        session.close()

    return cycle_log


def paper_trading_loop():
    global paper_engine_status
    print("[PaperEngine] Background thread started.", flush=True)
    while PAPER_TRADING_ENABLED:
        paper_engine_status["running"] = True
        paper_engine_status["last_run"] = datetime.now().isoformat()
        try:
            logs = run_paper_trading_cycle()
            paper_engine_status["last_cycle_log"] = logs
            for line in logs:
                print(f"[PaperEngine] {line}", flush=True)
        except Exception as e:
            paper_engine_status["last_cycle_log"] = [f"ERROR: {str(e)}"]
            print(f"[PaperEngine] Error: {e}", flush=True)

        interval_secs = paper_engine_status["interval_minutes"] * 60
        paper_engine_status["next_run"] = datetime.fromtimestamp(
            time.time() + interval_secs
        ).isoformat()
        paper_engine_status["running"] = False

        # Sleep in 5-second intervals so we can check for stop signal
        elapsed = 0
        while elapsed < interval_secs and PAPER_TRADING_ENABLED:
            time.sleep(5)
            elapsed += 5

    paper_engine_status["running"] = False
    print("[PaperEngine] Background thread stopped.", flush=True)


# ─── FastAPI App ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="AI Trading System API", lifespan=lifespan)
Session = init_db()

os.makedirs("/Users/upendrasingh/data/My-Learnings/ai-trading/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="/Users/upendrasingh/data/My-Learnings/ai-trading/static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("/Users/upendrasingh/data/My-Learnings/ai-trading/static/index.html")


@app.get("/api/portfolio")
def get_portfolio():
    try:
        session = Session()
        port = session.query(Portfolio).first()
        active_trades = session.query(Trade).filter_by(status='OPEN').all()
        closed_trades = session.query(Trade).filter_by(status='CLOSED').order_by(Trade.exit_time.desc()).limit(20).all()

        total_pnl = sum(t.pnl or 0 for t in closed_trades)
        winning = len([t for t in closed_trades if (t.pnl or 0) > 0])
        win_rate = (winning / len(closed_trades) * 100) if closed_trades else 0

        return {
            "balance": port.balance,
            "total_pnl": total_pnl,
            "win_rate_pct": round(win_rate, 1),
            "total_closed_trades": len(closed_trades),
            "active_trades": [
                {
                    "symbol": t.symbol.replace('.NS', ''),
                    "entry_price": t.entry_price,
                    "quantity": t.quantity,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None
                } for t in active_trades
            ],
            "recent_closed_trades": [
                {
                    "symbol": t.symbol.replace('.NS', ''),
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                } for t in closed_trades
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/paper-trading/start")
def start_paper_trading(interval_minutes: int = 60):
    global PAPER_TRADING_ENABLED, _paper_trading_thread
    with _paper_trading_lock:
        if PAPER_TRADING_ENABLED:
            return {"status": "already_running", "message": "Paper trading engine is already running."}
        PAPER_TRADING_ENABLED = True
        paper_engine_status["interval_minutes"] = interval_minutes
        _paper_trading_thread = threading.Thread(target=paper_trading_loop, daemon=True)
        _paper_trading_thread.start()
    return {"status": "started", "message": f"Paper trading engine started. Cycles every {interval_minutes} minutes."}


@app.post("/api/paper-trading/stop")
def stop_paper_trading():
    global PAPER_TRADING_ENABLED
    PAPER_TRADING_ENABLED = False
    paper_engine_status["next_run"] = None
    return {"status": "stopped", "message": "Paper trading engine will stop after the current cycle."}


@app.post("/api/paper-trading/run-now")
def run_paper_now():
    """Trigger a manual paper trading cycle immediately."""
    try:
        logs = run_paper_trading_cycle()
        paper_engine_status["last_run"] = datetime.now().isoformat()
        paper_engine_status["last_cycle_log"] = logs
        return {"status": "completed", "cycle_log": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper-trading/status")
def get_paper_trading_status():
    return {
        "enabled": PAPER_TRADING_ENABLED,
        "running": paper_engine_status["running"],
        "last_run": paper_engine_status["last_run"],
        "next_run": paper_engine_status["next_run"],
        "interval_minutes": paper_engine_status["interval_minutes"],
        "last_cycle_log": paper_engine_status["last_cycle_log"],
    }


@app.post("/api/paper-trading/reset")
def reset_paper_trading():
    """Reset paper trading portfolio to initial capital."""
    try:
        session = Session()
        port = session.query(Portfolio).first()
        port.balance = 100000.0
        # Close all open trades
        open_trades = session.query(Trade).filter_by(status='OPEN').all()
        for t in open_trades:
            t.status = 'CLOSED'
            t.pnl = 0
        session.commit()
        return {"status": "reset", "message": "Portfolio reset to ₹1,00,000"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scan")
def scan_market():
    """Runs the trading engine on a watchlist and returns live signals with news."""
    results = []
    try:
        for symbol in WATCHLIST:
            df = fetch_historical_data(symbol, period='1y', interval='1d')
            if df is None or df.empty: continue

            df = add_technical_indicators(df)
            if df is None or df.empty or len(df) < 50: continue

            _ensemble.train_models(df)
            news = fetch_news_sentiment_raw(symbol)
            signal_data = _ensemble.get_signal(df, news)

            results.append({
                'symbol': symbol.replace('.NS', ''),
                'price': float(df['Close'].iloc[-1]),
                'signal': signal_data
            })

        return {'scan_results': results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str = None
    end_date: str = None
    interval: str = "1d"


@app.post("/api/backtest")
def api_run_backtest(req: BacktestRequest):
    try:
        result = run_backtest(
            req.symbol,
            req.start_date,
            req.end_date,
            models_to_use=['xgboost', 'lstm', 'finbert'],
            interval=req.interval
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

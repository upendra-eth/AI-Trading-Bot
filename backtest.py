import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import fetch_historical_data, fetch_news_sentiment_raw
from features import add_technical_indicators
from models import EnsembleStrategy

# Load models ONCE at module level — FinBERT loads here and stays in memory
print("[Startup] Initializing AI models (one-time)...", flush=True)
_ensemble = EnsembleStrategy()
print("[Startup] AI models ready.", flush=True)

def run_backtest(symbol: str, start_date: str, end_date: str, initial_capital: float = 100000.0, models_to_use: list = None, interval: str = '1d') -> dict:
    """Runs a historical backtest for a specific symbol."""
    import yfinance as yf
    import traceback
    
    ticker_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
    
    # Calculate warmup padding depending on interval
    pad_days = 120
    if interval == '1h': pad_days = 20
    elif interval in ['15m', '5m']: pad_days = 5
    
    try:
        if start_date:
            padded_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=pad_days)).strftime('%Y-%m-%d')
        else:
            return {"error": "start_date is required."}
    except ValueError:
        return {"error": f"Invalid date format. Use YYYY-MM-DD. Got start_date={start_date}"}
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=padded_start, end=end_date, interval=interval)
    except Exception as e:
        print(f"yfinance error: {traceback.format_exc()}", flush=True)
        return {"error": f"Failed to fetch data from Yahoo Finance for {ticker_symbol}. Check your internet connection. Detail: {str(e)}"}
    
    if df.empty:
        err_msg = f"No data found for {ticker_symbol} between {start_date} and {end_date}. Verify the symbol is a valid NSE stock."
        if interval != '1d':
            err_msg += f" Note: Yahoo Finance restricts intraday historical data (e.g. 15m is limited to 60 days, 1h to 730 days). Try a more recent start_date."
        return {"error": err_msg}
    
    print(f"[Backtest] Fetched {len(df)} bars for {ticker_symbol} ({padded_start} to {end_date})", flush=True)
        
    df = add_technical_indicators(df)
    if df.empty or len(df) < 50:
        return {"error": f"Not enough data for technical indicators. Got {len(df)} bars after indicator calculation (need at least 50). Try a wider date range."}
        
    # Use cached ensemble — only retrain XGBoost and LSTM on the new data
    try:
        _ensemble.train_models(df)
    except Exception as e:
        print(f"Model training error: {traceback.format_exc()}", flush=True)
        return {"error": f"Model training failed: {str(e)}"}
        
    print(f"[Backtest] Fetching news for {ticker_symbol}...", flush=True)
    news_list = fetch_news_sentiment_raw(ticker_symbol)
    print(f"[Backtest] Found {len(news_list)} recent news articles.", flush=True)
    
    balance = initial_capital
    position = 0
    current_trade = None
    trades = []
    equity_curve = []
    
    total_days = len(df) - 1 - 50
    print(f"[Backtest] Starting simulation: {total_days} trading days...", flush=True)
    
    import time
    sim_start = time.time()
    
    # Simulate day by day starting from day 50
    for i in range(50, len(df)-1):
        current_date = df.index[i]
        next_date = df.index[i+1]
        
        hist_df = df.iloc[:i+1]
        current_close = hist_df['Close'].iloc[-1]
        
        # Pass news to get actual FinBERT scores
        signal_data = _ensemble.get_signal(hist_df, news_list) 
        signal = signal_data['final_signal']
        
        next_open = df['Open'].iloc[i+1]
        
        # Store signal snapshot for the trade log
        signal_snapshot = {
            'final_signal': signal,
            'weighted_score': signal_data.get('weighted_score', 0),
            'explanation': signal_data.get('explanation', ''),
            'xgb': signal_data.get('model_details', {}).get('xgboost', {}).get('direction', 'NEUTRAL'),
            'lstm': signal_data.get('model_details', {}).get('lstm', {}).get('direction', 'NEUTRAL'),
            'finbert': signal_data.get('model_details', {}).get('finbert', {}).get('direction', 'NEUTRAL')
        }
        
        if signal == 'BUY' and position == 0:
            quantity = int(balance * 0.95 / next_open)
            if quantity > 0:
                position = quantity
                entry_price = next_open
                balance -= position * entry_price
                timestamp_val_trade = int(next_date.timestamp()) if interval != '1d' else next_date.strftime('%Y-%m-%d')
                
                current_trade = {
                    'entry_date': next_date.strftime('%Y-%m-%d %H:%M') if interval != '1d' else next_date.strftime('%Y-%m-%d'),
                    'entry_time': timestamp_val_trade,
                    'entry_price': float(next_open),
                    'quantity': position,
                    'buy_signal': signal_snapshot
                }
                
        elif signal == 'SELL' and position > 0 and current_trade:
            revenue = position * next_open
            pnl = revenue - (position * current_trade['entry_price'])
            balance += revenue
            timestamp_val_trade = int(next_date.timestamp()) if interval != '1d' else next_date.strftime('%Y-%m-%d')
            
            exit_date_str = next_date.strftime('%Y-%m-%d %H:%M') if interval != '1d' else next_date.strftime('%Y-%m-%d')
            
            # Duration logic
            try:
                fmt = '%Y-%m-%d %H:%M' if interval != '1d' else '%Y-%m-%d'
                dt_in = datetime.strptime(current_trade['entry_date'], fmt)
                dt_out = datetime.strptime(exit_date_str, fmt)
                duration_str = str(dt_out - dt_in)
            except:
                duration_str = '-'
                
            current_trade.update({
                'exit_date': exit_date_str,
                'exit_time': timestamp_val_trade,
                'exit_price': float(next_open),
                'pnl': float(pnl),
                'sell_signal': signal_snapshot,
                'duration': duration_str
            })
            trades.append(current_trade)
            current_trade = None
            position = 0
            
        # Time format for Lightweight Charts: string for 1d, unix timestamp for intraday
        timestamp_val = current_date.strftime('%Y-%m-%d') if interval == '1d' else int(current_date.timestamp())
        
        current_equity = balance + (position * current_close)
        equity_curve.append({
            'time': timestamp_val,
            'value': float(current_equity),
            'open': float(hist_df['Open'].iloc[-1]),
            'high': float(hist_df['High'].iloc[-1]),
            'low': float(hist_df['Low'].iloc[-1]),
            'close': float(current_close)
        })
        
        # Progress log every 50 days
        day_num = i - 50 + 1
        if day_num % 50 == 0 or day_num == total_days:
            elapsed = time.time() - sim_start
            print(f"[Backtest] Day {day_num}/{total_days} ({current_date.strftime('%Y-%m-%d')}) | Equity: ₹{current_equity:,.0f} | {elapsed:.1f}s elapsed", flush=True)
        
    print(f"[Backtest] Simulation complete in {time.time()-sim_start:.1f}s", flush=True)
    if position > 0 and current_trade:
        final_price = df['Close'].iloc[-1]
        final_date = df.index[-1]
        revenue = position * final_price
        pnl = revenue - (position * current_trade['entry_price'])
        balance += revenue
        timestamp_val_trade = int(final_date.timestamp()) if interval != '1d' else final_date.strftime('%Y-%m-%d')
        
        exit_date_str = final_date.strftime('%Y-%m-%d %H:%M') if interval != '1d' else final_date.strftime('%Y-%m-%d')
        try:
            fmt = '%Y-%m-%d %H:%M' if interval != '1d' else '%Y-%m-%d'
            dt_in = datetime.strptime(current_trade['entry_date'], fmt)
            dt_out = datetime.strptime(exit_date_str, fmt)
            duration_str = str(dt_out - dt_in)
        except:
            duration_str = '-'
            
        current_trade.update({
            'exit_date': exit_date_str,
            'exit_time': timestamp_val_trade,
            'exit_price': float(final_price),
            'pnl': float(pnl),
            'sell_signal': None,
            'duration': duration_str + ' (Auto-Close)'
        })
        trades.append(current_trade)
        
        timestamp_val = final_date.strftime('%Y-%m-%d') if interval == '1d' else int(final_date.timestamp())
        equity_curve.append({
            'time': timestamp_val,
            'value': float(balance),
            'open': float(df['Open'].iloc[-1]),
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'close': float(final_price)
        })
        
    total_return = (balance - initial_capital) / initial_capital * 100
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    total_closed_trades = len(trades)
    win_rate = (len(winning_trades) / total_closed_trades * 100) if total_closed_trades > 0 else 0
    
    return {
        'initial_capital': initial_capital,
        'final_balance': float(balance),
        'total_return_pct': float(total_return),
        'win_rate_pct': float(win_rate),
        'total_trades': total_closed_trades,
        'equity_curve': equity_curve,
        'trade_history': trades
    }

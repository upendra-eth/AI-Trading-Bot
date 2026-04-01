import time
from datetime import datetime
from database import init_db, Opportunity, Trade, Portfolio
from data_fetcher import fetch_historical_data, fetch_news_sentiment_raw
from features import add_technical_indicators
from models import EnsembleStrategy

NSE_SYMBOLS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
TRADE_QUANTITY = 10  # Fixed paper trade quantity

def run_trading_cycle():
    print(f"--- Starting Paper Trading Cycle at {datetime.now()} ---")
    Session = init_db()
    session = Session()
    portfolio = session.query(Portfolio).first()
    
    ensemble = EnsembleStrategy()
    
    for symbol in NSE_SYMBOLS:
        print(f"\nEvaluating {symbol}...")
        
        # 1. Fetch Data
        df = fetch_historical_data(symbol, period='6mo', interval='1d')
        if df is None or df.empty:
            print(f"No data for {symbol}, skipping.")
            continue
            
        # 2. Add Tech Indicators
        df = add_technical_indicators(df)
        if df is None or df.empty or len(df) < 50:
            print(f"Not enough data after features calculation for {symbol}, skipping.")
            continue
            
        # 3. Train models
        # For dynamic paper trading, we update weights with latest data
        ensemble.train_models(df)
        
        # 4. Fetch News
        news = fetch_news_sentiment_raw(symbol)
        
        # 5. Get Signal
        signal_data = ensemble.get_signal(df, news)
        print(f"Signal Strategy Result: {signal_data}")
        
        # 6. Log Opportunity
        current_price = df['Close'].iloc[-1]
        opp = Opportunity(
            symbol=symbol,
            xgb_signal=signal_data['xgb_signal'],
            lstm_signal=signal_data['lstm_signal'],
            finbert_signal=signal_data['finbert_signal'],
            final_signal=signal_data['final_signal'],
            executed=False
        )
        session.add(opp)
        
        # 7. Execute Paper Trades
        active_trade = session.query(Trade).filter_by(symbol=symbol, status='OPEN').first()
        
        if signal_data['final_signal'] == 'BUY' and not active_trade:
            cost = current_price * TRADE_QUANTITY
            if portfolio.balance >= cost:
                print(f"-> Executing Paper BUY for {symbol} at {current_price:.2f}")
                new_trade = Trade(
                    symbol=symbol,
                    entry_time=datetime.now(),
                    entry_price=float(current_price),
                    quantity=TRADE_QUANTITY,
                    status='OPEN'
                )
                portfolio.balance -= cost
                opp.executed = True
                session.add(new_trade)
            else:
                print(f"-> Insufficient balance to buy {symbol}")
                
        elif signal_data['final_signal'] == 'SELL' and active_trade:
            print(f"-> Executing Paper SELL for {symbol} at {current_price:.2f}")
            revenue = current_price * active_trade.quantity
            pnl = revenue - (active_trade.entry_price * active_trade.quantity)
            
            active_trade.exit_time = datetime.now()
            active_trade.exit_price = float(current_price)
            active_trade.pnl = float(pnl)
            active_trade.status = 'CLOSED'
            
            portfolio.balance += revenue
            opp.executed = True
            print(f"-> Trade closed with PnL: {pnl:.2f}")
            
        else:
            print("-> No trade action taken.")
            
        session.commit()
    
    print(f"\n--- Cycle Complete. Current Portfolio Balance: {portfolio.balance:.2f} ---")

if __name__ == '__main__':
    run_trading_cycle()

import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds a comprehensive set of technical indicators to the OHLCV dataframe."""
    if df is None or df.empty or len(df) < 50:
        return df
        
    # --- Trend Indicators ---
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['EMA_21'] = ta.ema(df['Close'], length=21)
    
    # --- Momentum Indicators ---
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    
    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df['Close'], length=14)
    if stoch_rsi is not None and not stoch_rsi.empty:
        cols = stoch_rsi.columns.tolist()
        df['STOCH_RSI_K'] = stoch_rsi[cols[0]]
        df['STOCH_RSI_D'] = stoch_rsi[cols[1]]
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        cols = macd.columns.tolist()
        df['MACD'] = macd[cols[0]]
        df['MACD_Histogram'] = macd[cols[1]]
        df['MACD_Signal'] = macd[cols[2]]
        
    # --- Volatility Indicators ---
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        cols = bbands.columns.tolist()
        df['BB_LOWER'] = bbands[cols[0]]
        df['BB_MID'] = bbands[cols[1]]
        df['BB_UPPER'] = bbands[cols[2]]
    
    # ATR (Average True Range)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    if atr is not None:
        df['ATR_14'] = atr
    
    # --- Volume Indicators ---
    # OBV (On-Balance Volume)
    obv = ta.obv(df['Close'], df['Volume'])
    if obv is not None:
        df['OBV'] = obv
    
    # VWAP approximation (cumulative for the dataset)
    if 'Volume' in df.columns:
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # --- Derived Features ---
    # Price distance from moving averages (normalized)
    df['DIST_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    df['DIST_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
    
    # BB Width (volatility squeeze indicator)
    if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID'] * 100
    
    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)
    
    return df

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Fix PyTorch thread deadlock on Apple Silicon (M-series Macs)
torch.set_num_threads(1)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ─── XGBoost Model ──────────────────────────────────────────────────────────────

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=0
        )
        self.is_trained = False
        self.feature_names = []
        self.feature_importance = {}
        
    def train(self, df: pd.DataFrame, target_col: str = 'Close'):
        """Trains the XGBoost model."""
        if len(df) < 100:
            print("  XGBoost: Not enough data (<100 rows), skipping.", flush=True)
            return
            
        features = df.drop(columns=['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], errors='ignore')
        target = df['Target'] if 'Target' in df else df['Close'].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]
        
        valid_idx = target.index.intersection(features.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        if len(features) > 0 and len(target) > 0:
            self.feature_names = features.columns.tolist()
            self.model.fit(features, target)
            self.is_trained = True
            self.trained_samples = len(features)
            
            # Log feature importance
            importances = self.model.feature_importances_
            self.feature_importance = dict(sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1], reverse=True
            ))
            top5 = list(self.feature_importance.items())[:5]
            print(f"  XGBoost: Trained on {len(features)} samples, {len(self.feature_names)} features", flush=True)
            print(f"  XGBoost: Top features: {', '.join(f'{k}({v:.3f})' for k,v in top5)}", flush=True)
            
    def predict(self, df: pd.DataFrame) -> dict:
        """Returns prediction dict with score, direction, and confidence."""
        if not self.is_trained or df.empty:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'confidence': 0.0, 'predicted_change_pct': 0.0, 'data_points_used': 0, 'features_used': 0}
            
        features = df.drop(columns=['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], errors='ignore').dropna()
        if len(features) == 0:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'confidence': 0.0}
        
        latest_features = features.iloc[-1:]
        pred = float(self.model.predict(latest_features)[0])
        if pd.isna(pred) or np.isinf(pred):
            pred = 0.0
        
        # Determine direction and confidence
        abs_pred = abs(pred)
        confidence = min(abs_pred / 0.02 * 100, 100)  # 2% change = 100% confidence
        
        if pred > 0.001:
            direction = 'BULLISH'
        elif pred < -0.001:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
            
        return {
            'score': pred,
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_change_pct': round(pred * 100, 4),
            'data_points_used': getattr(self, 'trained_samples', 0),
            'features_used': len(self.feature_names)
        }

# ─── LSTM Model ──────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMModel:
    def __init__(self, seq_length=10):
        self.seq_length = seq_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.train_losses = []
        
    def prepare_data(self, df):
        numeric_df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore').select_dtypes(include=[np.number]).dropna()
        if len(numeric_df) == 0:
            return None, None
            
        data = self.scaler.fit_transform(numeric_df.values)
        close_idx = numeric_df.columns.get_loc('Close') if 'Close' in numeric_df.columns else 0
            
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:(i + self.seq_length)])
            y.append(data[i + self.seq_length, close_idx])
            
        if not X:
            return None, None
            
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, df: pd.DataFrame):
        t0 = time.time()
        X, y = self.prepare_data(df)
        if X is None or len(X) < 10:
            print("  LSTM: Not enough sequences, skipping.", flush=True)
            return
        
        # Cap training data
        if len(X) > 200:
            X = X[-200:]
            y = y[-200:]
        
        n_features = X.shape[2]
        print(f"  LSTM: {X.shape[0]} sequences, {n_features} features", flush=True)
            
        self.model = LSTMNet(input_size=n_features)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-5)
        
        self.train_losses = []
        num_epochs = 10
        best_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            self.train_losses.append(loss_val)
            
            # Early stopping
            if loss_val < best_loss - 0.0001:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"  LSTM: Early stop at epoch {epoch+1}, Loss: {loss_val:.6f}", flush=True)
                break
                
        print(f"  LSTM: Done in {time.time()-t0:.1f}s, Final Loss: {self.train_losses[-1]:.6f}", flush=True)
        self.is_trained = True
        self.trained_samples = len(X)

    def predict(self, df: pd.DataFrame) -> dict:
        """Returns prediction dict with score, direction, and trend info."""
        numeric_df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore').select_dtypes(include=[np.number]).dropna()
        if not self.is_trained or len(numeric_df) < self.seq_length:
            return {'score': 0.0, 'direction': 'NEUTRAL', 'confidence': 0.0}
            
        self.model.eval()
        data = self.scaler.transform(numeric_df.values[-self.seq_length:])
        X = torch.tensor(np.array([data]), dtype=torch.float32)
        
        with torch.no_grad():
            pred = self.model(X).item()
            if pd.isna(pred) or np.isinf(pred):
                pred = 0.0
            
        close_idx = numeric_df.columns.get_loc('Close') if 'Close' in numeric_df.columns else 0
        current_close_scaled = data[-1, close_idx]
        delta = float(pred - current_close_scaled)
        
        abs_delta = abs(delta)
        confidence = min(abs_delta / 0.05 * 100, 100)
        
        if delta > 0.001:
            direction = 'BULLISH'
        elif delta < -0.001:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
            
        return {
            'score': delta,
            'direction': direction,
            'confidence': round(confidence, 1),
            'predicted_vs_current': round(delta, 6),
            'data_points_used': getattr(self, 'trained_samples', 0)
        }

# ─── FinBERT Sentiment Model ────────────────────────────────────────────────────

class SentimentModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True)
            print("FinBERT loaded from local cache.", flush=True)
        except Exception:
            try:
                print("Downloading FinBERT model (first time only)...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                print("FinBERT downloaded and cached.", flush=True)
            except Exception as e:
                print(f"Failed to load FinBERT: {e}", flush=True)
                self.model = None
            
    def predict(self, news_list: list) -> dict:
        """Returns detailed sentiment analysis with per-headline breakdown."""
        result = {
            'score': 0.0,
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'headlines_analyzed': 0,
            'headline_details': [],
            'summary': 'No news available'
        }
        
        if not news_list or not self.model:
            if not self.model:
                result['summary'] = 'FinBERT model not loaded'
            return result
            
        headlines = [n['title'] if isinstance(n, dict) else str(n) for n in news_list[:5]]
        
        try:
            inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='pt', max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            headline_details = []
            total_score = 0.0
            
            for i, (headline, pred) in enumerate(zip(headlines, predictions)):
                pos, neg, neu = pred.numpy()
                score = float(pos - neg)
                total_score += score
                
                # Determine sentiment label
                if pos > neg and pos > neu:
                    sentiment = 'POSITIVE'
                elif neg > pos and neg > neu:
                    sentiment = 'NEGATIVE'
                else:
                    sentiment = 'NEUTRAL'
                
                detail = {
                    'headline': headline,
                    'sentiment': sentiment,
                    'score': round(score, 4),
                    'positive': round(float(pos), 4),
                    'negative': round(float(neg), 4),
                    'neutral': round(float(neu), 4)
                }
                headline_details.append(detail)
                print(f"  FinBERT: [{sentiment:8s}] ({score:+.3f}) \"{headline[:60]}...\"", flush=True)
                
            avg_score = total_score / len(headlines)
            confidence = min(abs(avg_score) / 0.5 * 100, 100)
            
            if avg_score > 0.15:
                direction = 'BULLISH'
            elif avg_score < -0.15:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            pos_count = sum(1 for h in headline_details if h['sentiment'] == 'POSITIVE')
            neg_count = sum(1 for h in headline_details if h['sentiment'] == 'NEGATIVE')
            neu_count = sum(1 for h in headline_details if h['sentiment'] == 'NEUTRAL')
            
            result.update({
                'score': round(avg_score, 4),
                'direction': direction,
                'confidence': round(confidence, 1),
                'headlines_analyzed': len(headlines),
                'headline_details': headline_details,
                'summary': f"{pos_count} positive, {neg_count} negative, {neu_count} neutral out of {len(headlines)} headlines"
            })
            
            print(f"  FinBERT: Overall {direction} (avg={avg_score:+.3f}, conf={confidence:.0f}%)", flush=True)
            
        except Exception as e:
            print(f"  FinBERT: Error during analysis: {e}", flush=True)
            result['summary'] = f'Analysis error: {str(e)}'
            
        return result

# ─── Ensemble Strategy ───────────────────────────────────────────────────────────

class EnsembleStrategy:
    # Model weights for combining signals
    WEIGHTS = {
        'xgboost': 0.35,
        'lstm': 0.35,
        'finbert': 0.30
    }
    
    def __init__(self):
        self.xgb = XGBoostModel()
        self.lstm = LSTMModel()
        self.sentiment = SentimentModel()
        
    def train_models(self, df: pd.DataFrame):
        print("Training XGBoost Model...", flush=True)
        self.xgb.train(df)
        print("Training LSTM Model...", flush=True)
        self.lstm.train(df)
        print("All models trained.", flush=True)
        
    def get_signal(self, df: pd.DataFrame, news: list) -> dict:
        """Returns comprehensive signal with full model breakdown."""
        t0 = time.time()
        
        # Get predictions from all models
        xgb_result = self.xgb.predict(df)
        lstm_result = self.lstm.predict(df)
        sent_result = self.sentiment.predict(news)
        
        # Weighted scoring
        xgb_vote = 0
        if xgb_result['direction'] == 'BULLISH':
            xgb_vote = 1
        elif xgb_result['direction'] == 'BEARISH':
            xgb_vote = -1
            
        lstm_vote = 0
        if lstm_result['direction'] == 'BULLISH':
            lstm_vote = 1
        elif lstm_result['direction'] == 'BEARISH':
            lstm_vote = -1
            
        sent_vote = 0
        if sent_result['direction'] == 'BULLISH':
            sent_vote = 1
        elif sent_result['direction'] == 'BEARISH':
            sent_vote = -1
        
        # Weighted combination
        weighted_score = (
            xgb_vote * self.WEIGHTS['xgboost'] +
            lstm_vote * self.WEIGHTS['lstm'] +
            sent_vote * self.WEIGHTS['finbert']
        )
        
        # Decision thresholds
        if weighted_score >= 0.5:
            final_signal = 'BUY'
        elif weighted_score <= -0.5:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
        
        # Build vote explanation
        vote_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        explanation = (
            f"XGB:{vote_map[xgb_vote]}({self.WEIGHTS['xgboost']:.0%}) + "
            f"LSTM:{vote_map[lstm_vote]}({self.WEIGHTS['lstm']:.0%}) + "
            f"FinBERT:{vote_map[sent_vote]}({self.WEIGHTS['finbert']:.0%}) "
            f"= {weighted_score:+.2f} → {final_signal}"
        )
        
        # Sanitize floats to avoid NaN in JSON
        def safe_float(v):
            if pd.isna(v) or np.isinf(v): return 0.0
            return float(v)
            
        result = {
            'final_signal': final_signal,
            'weighted_score': safe_float(weighted_score),
            'explanation': explanation,
            'xgb_signal': safe_float(xgb_result['score']),
            'lstm_signal': safe_float(lstm_result['score']),
            'finbert_signal': safe_float(sent_result['score']),
            'model_details': {
                'xgboost': xgb_result,
                'lstm': lstm_result,
                'finbert': sent_result
            },
            'processing_time_ms': safe_float((time.time() - t0) * 1000)
        }
        
        return result

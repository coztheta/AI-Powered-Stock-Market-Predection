# -*- coding: utf-8 -*-
import warnings
import traceback
import os
import sys
import secrets
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import joblib
import hashlib
import json

from yfinance import ticker

# Deep Learning Libraries
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow not available. Install with: pip install tensorflow. Error: {e}")

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32)

# --- COMPREHENSIVE STOCK DATABASE ---
STOCK_UNIVERSE = {
    # Indian IT Sector
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'Information Technology', 'exchange': 'NSE', 'country': 'India'},
    'INFY.NS': {'name': 'Infosys Limited', 'sector': 'Information Technology', 'exchange': 'NSE', 'country': 'India'},
    'WIPRO.NS': {'name': 'Wipro Limited', 'sector': 'Information Technology', 'exchange': 'NSE', 'country': 'India'},
    'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'Information Technology', 'exchange': 'NSE', 'country': 'India'},
    'TECHM.NS': {'name': 'Tech Mahindra', 'sector': 'Information Technology', 'exchange': 'NSE', 'country': 'India'},
    
    # Indian Banking
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking', 'exchange': 'NSE', 'country': 'India'},
    'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking', 'exchange': 'NSE', 'country': 'India'},
    'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking', 'exchange': 'NSE', 'country': 'India'},
    'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking', 'exchange': 'NSE', 'country': 'India'},
    'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking', 'exchange': 'NSE', 'country': 'India'},
    
    # Indian Energy
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Oil & Gas', 'exchange': 'NSE', 'country': 'India'},
    'ONGC.NS': {'name': 'ONGC', 'sector': 'Oil & Gas', 'exchange': 'NSE', 'country': 'India'},
    'BPCL.NS': {'name': 'Bharat Petroleum', 'sector': 'Oil & Gas', 'exchange': 'NSE', 'country': 'India'},
    'ADANIGREEN.NS': {'name': 'Adani Green Energy', 'sector': 'Renewable Energy', 'exchange': 'NSE', 'country': 'India'},
    'TATAPOWER.NS': {'name': 'Tata Power', 'sector': 'Power', 'exchange': 'NSE', 'country': 'India'},
    
    # US Technology
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ', 'country': 'USA'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'exchange': 'NASDAQ', 'country': 'USA'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'exchange': 'NASDAQ', 'country': 'USA'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'E-commerce', 'exchange': 'NASDAQ', 'country': 'USA'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automobiles', 'exchange': 'NASDAQ', 'country': 'USA'},
    'META': {'name': 'Meta Platforms', 'sector': 'Technology', 'exchange': 'NASDAQ', 'country': 'USA'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'exchange': 'NASDAQ', 'country': 'USA'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Entertainment', 'exchange': 'NASDAQ', 'country': 'USA'},
    
    # US Financial
    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Banking', 'exchange': 'NYSE', 'country': 'USA'},
    'BAC': {'name': 'Bank of America', 'sector': 'Banking', 'exchange': 'NYSE', 'country': 'USA'},
    'V': {'name': 'Visa Inc.', 'sector': 'Financial Services', 'exchange': 'NYSE', 'country': 'USA'},
    'MA': {'name': 'Mastercard', 'sector': 'Financial Services', 'exchange': 'NYSE', 'country': 'USA'},
    
    # US Healthcare
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'exchange': 'NYSE', 'country': 'USA'},
    'PFE': {'name': 'Pfizer Inc.', 'sector': 'Pharmaceuticals', 'exchange': 'NYSE', 'country': 'USA'},
    'UNH': {'name': 'UnitedHealth Group', 'sector': 'Healthcare', 'exchange': 'NYSE', 'country': 'USA'},
    
    # ETFs
    'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': 'ETF', 'exchange': 'NYSE', 'country': 'USA'},
    'QQQ': {'name': 'Invesco QQQ Trust', 'sector': 'ETF', 'exchange': 'NASDAQ', 'country': 'USA'},
    'VTI': {'name': 'Vanguard Total Stock Market ETF', 'sector': 'ETF', 'exchange': 'NYSE', 'country': 'USA'},
}

# Simple user database (in production, use proper database)
USERS_DB = {}

class HybridLSTMARIMAPredictor:
    """Production-ready LSTM-ARIMA Hybrid Model for Stock Prediction"""
    
    def __init__(self, lstm_sequence_length: int = 60):
        self.lstm_sequence_length = lstm_sequence_length
        self.lstm_model = None
        self.lstm_scaler = None
        self.arima_model = None
        self.arima_order = (5, 1, 0)
        self.hybrid_weights = {'lstm': 0.6, 'arima': 0.4}
        self.models_dir = "models"
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def fetch_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            data = yf.download(ticker, period=period, interval="1d", progress=False)
            if data is None or data.empty:
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def prepare_lstm_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM training"""
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        X, y = [], []
        for i in range(self.lstm_sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.lstm_sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build enhanced LSTM architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(12, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def fit_arima_model(self, data: pd.DataFrame) -> None:
        """Fit ARIMA model for linear trend capture"""
        try:
            prices = data['Close'].values
            log_returns = np.log(prices[1:] / prices[:-1])
            
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            # Grid search for optimal ARIMA parameters
            for p in range(0, 4):
                for d in range(0, 2):
                    for q in range(0, 4):
                        try:
                            model = ARIMA(log_returns, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            self.arima_order = best_order
            self.arima_model = ARIMA(log_returns, order=best_order).fit()
            print(f"✅ ARIMA{best_order} fitted with AIC: {best_aic:.4f}")
            
        except Exception as e:
            print(f"⚠️ ARIMA fitting fallback: {e}")
            log_returns = np.log(data['Close'].values[1:] / data['Close'].values[:-1])
            self.arima_model = ARIMA(log_returns, order=(1, 1, 1)).fit()
    
    def train_hybrid_model(self, ticker: str, epochs: int = 50) -> Dict[str, Any]:
        """Train complete hybrid LSTM-ARIMA model"""
        try:
            print(f"🚀 Training Hybrid Model for {ticker}")
            
            data = self.fetch_data(ticker, period="3y")
            if data is None:
                return {"error": "Could not fetch data", "success": False}
            
            print(f"📊 Data fetched: {len(data)} days")
            
            # Train LSTM component
            X, y, scaler = self.prepare_lstm_data(data)
            self.lstm_scaler = scaler
            
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            self.lstm_model = self.build_lstm_model((X.shape[1], 1))
            
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = self.lstm_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            lstm_train_loss = float(np.min(history.history['loss']))
            lstm_val_loss = float(np.min(history.history['val_loss']))
            
            # Train ARIMA component
            self.fit_arima_model(data)
            
            # Optimize ensemble weights
            self.optimize_ensemble_weights(data.tail(100))

            # Save trained model artifacts
            os.makedirs(self.models_dir, exist_ok=True)

            # Save the LSTM model
            self.lstm_model.save(
                os.path.join(self.models_dir, f"{ticker}_lstm.keras")
            )

            # Save the fitted LSTM scaler
            joblib.dump(
                self.lstm_scaler,
                os.path.join(self.models_dir, f"{ticker}_scaler.joblib")
            )

            # Save the fitted ARIMA model, if available
            if getattr(self, "arima_model", None) is not None:
                joblib.dump(
                    self.arima_model,
                    os.path.join(self.models_dir, f"{ticker}_arima.joblib")
                )

            print(f"💾 Model artifacts saved for {ticker}")
            print(f"✅ Hybrid training completed!")
            print(f"🧠 LSTM Loss: {lstm_train_loss:.6f} / {lstm_val_loss:.6f}")
            print(f"⚖️ Weights: LSTM {self.hybrid_weights['lstm']:.2f}, ARIMA {self.hybrid_weights['arima']:.2f}")
            
            return {
                "success": True,
                "lstm_train_loss": lstm_train_loss,
                "lstm_val_loss": lstm_val_loss,
                "arima_order": self.arima_order,
                "ensemble_weights": self.hybrid_weights,
                "epochs_completed": len(history.history['loss']),
                "data_points": len(data),
                "model_type": "Hybrid LSTM-ARIMA"
            }
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return {"error": str(e), "success": False}
    
    def optimize_ensemble_weights(self, validation_data: pd.DataFrame) -> None:
        """Optimize hybrid ensemble weights"""
        try:
            lstm_preds = self.predict_lstm_only(validation_data, days=len(validation_data))
            arima_preds = self.predict_arima_only(validation_data, days=len(validation_data))
            actual_prices = validation_data['Close'].values
            
            best_mse = np.inf
            best_weights = {'lstm': 0.5, 'arima': 0.5}
            
            for lstm_weight in np.arange(0.3, 0.8, 0.05):
                arima_weight = 1 - lstm_weight
                
                if len(lstm_preds) > 0 and len(arima_preds) > 0:
                    hybrid_preds = (lstm_weight * np.array(lstm_preds[:len(actual_prices)]) + 
                                  arima_weight * np.array(arima_preds[:len(actual_prices)]))
                    
                    mse = mean_squared_error(actual_prices, hybrid_preds)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = {'lstm': lstm_weight, 'arima': arima_weight}
            
            self.hybrid_weights = best_weights
            print(f"🎯 Optimized: LSTM {best_weights['lstm']:.3f}, ARIMA {best_weights['arima']:.3f}")
            
        except Exception as e:
            print(f"⚠️ Weight optimization failed: {e}")
    
    def predict_lstm_only(self, data: pd.DataFrame, days: int = 30) -> List[float]:
        """LSTM predictions only"""
        try:
            if self.lstm_model is None or self.lstm_scaler is None:
                print(f"LSTM DEBUG: model is None = {self.lstm_model is None}")
                print(f"LSTM DEBUG: scaler is None = {self.lstm_scaler is None}")
                return []
            
            prices = data['Close'].tail(self.lstm_sequence_length).values.reshape(-1, 1)
            scaled_prices = self.lstm_scaler.transform(prices)
            
            predictions = []
            current_sequence = scaled_prices.copy()
            
            for _ in range(days):
                X_pred = current_sequence.reshape((1, self.lstm_sequence_length, 1))
                next_price_scaled = self.lstm_model.predict(X_pred, verbose=0)[0, 0]
                next_price = self.lstm_scaler.inverse_transform([[next_price_scaled]])[0, 0]
                
                predictions.append(float(next_price))
                
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1, 0] = next_price_scaled
            
            return predictions
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return []
    
    def predict_arima_only(self, data: pd.DataFrame, days: int = 30) -> List[float]:
        """Generate ARIMA-only price predictions."""
        try:
            if self.arima_model is None:
                return []

            close_data = data["Close"]

            # Handle normal columns and yfinance MultiIndex columns.
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]

            if isinstance(close_data, pd.Series):
                prices = close_data.to_numpy(dtype=float).reshape(-1)
            else:
                prices = np.asarray(close_data, dtype=float).reshape(-1)

            if len(prices) == 0 or not np.isfinite(prices[-1]):
                return []

            current_price = float(prices[-1])

            # Normalize the forecast to a one-dimensional array.
            forecast = np.asarray(
                self.arima_model.forecast(steps=days),
                dtype=float
            ).reshape(-1)

            predictions = []
            last_price = current_price

            for log_return in forecast:
                log_return = float(log_return)
                predicted_price = last_price * np.exp(log_return)

                if not np.isfinite(predicted_price):
                    raise ValueError("ARIMA generated a non-finite price.")

                predictions.append(float(predicted_price))
                last_price = predicted_price

            return predictions

        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            traceback.print_exc()
            return []
    
    def predict_hybrid(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Generate hybrid LSTM-ARIMA predictions"""
        try:
            data = self.fetch_data(ticker, period="1y")
            if data is None:
                return {"error": "Could not fetch data"}
            
            lstm_predictions = self.predict_lstm_only(data, days)
            arima_predictions = self.predict_arima_only(data, days)
            print(f"DEBUG {ticker}: data rows = {len(data)}")
            print(f"DEBUG {ticker}: LSTM predictions = {len(lstm_predictions)}")
            print(f"DEBUG {ticker}: ARIMA predictions = {len(arima_predictions)}")
            if not lstm_predictions or not arima_predictions:
                return {"error": "Model predictions failed"}
            
            hybrid_predictions = []
            for i in range(min(len(lstm_predictions), len(arima_predictions), days)):
                hybrid_pred = (self.hybrid_weights['lstm'] * lstm_predictions[i] + 
                             self.hybrid_weights['arima'] * arima_predictions[i])
                hybrid_predictions.append(hybrid_pred)
            
            close_data = data["Close"]

            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]

            current_price = float(close_data.iloc[-1])
            final_price = hybrid_predictions[-1] if hybrid_predictions else current_price * 1.05
            expected_return = ((final_price / current_price) - 1) * 100
            
            return {
                "success": True,
                "current_price": current_price,
                "hybrid_predictions": hybrid_predictions,
                "lstm_predictions": lstm_predictions[:days],
                "arima_predictions": arima_predictions[:days],
                "final_price": final_price,
                "expected_return": expected_return,
                "model_weights": self.hybrid_weights,
                "prediction_days": len(hybrid_predictions),
                "model_type": "Hybrid LSTM-ARIMA"
            }
            
        except Exception as e:
            return {"error": str(e)}

# Initialize hybrid predictor
hybrid_predictor = HybridLSTMARIMAPredictor()

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Main trading dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', stocks=STOCK_UNIVERSE)

@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/signup')
def signup():
    """Signup page"""
    return render_template('signup.html')

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    """User registration API"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if email in USERS_DB:
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Hash password (in production, use proper hashing like bcrypt)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        USERS_DB[email] = {
            'name': name,
            'password': hashed_password,
            'created_at': datetime.now().isoformat()
        }
        
        session['user_id'] = email
        session['user_name'] = name
        
        return jsonify({'success': True, 'message': 'Registration successful'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """User login API"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if email not in USERS_DB:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        if USERS_DB[email]['password'] != hashed_password:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
        session['user_id'] = email
        session['user_name'] = USERS_DB[email]['name']
        
        return jsonify({'success': True, 'message': 'Login successful'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """User logout API"""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/stocks')
def api_stocks():
    """Get all available stocks"""
    return jsonify({
        'stocks': STOCK_UNIVERSE,
        'count': len(STOCK_UNIVERSE),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stock/<ticker>')
def api_stock_info(ticker: str):
    """Get current stock information"""
    try:
        ticker = ticker.upper()
        
        if ticker not in STOCK_UNIVERSE:
            return jsonify({'error': 'Stock not found'}), 404
        
        stock_info = STOCK_UNIVERSE[ticker]
        
        try:
            data = yf.download(ticker, period="2d", interval="1d", progress=False)
            if data is not None and not data.empty:
                current_price = float(data['Close'].iloc[-1])
                previous_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close else 0
                
                return jsonify({
                    'symbol': ticker,
                    'name': stock_info['name'],
                    'sector': stock_info['sector'],
                    'exchange': stock_info['exchange'],
                    'country': stock_info['country'],
                    'currentPrice': round(current_price, 2),
                    'change': round(change, 2),
                    'changePercent': round(change_percent, 2),
                    'dayHigh': round(float(data['High'].iloc[-1]), 2),
                    'dayLow': round(float(data['Low'].iloc[-1]), 2),
                    'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0,
                    'dataSource': 'live'
                })
            else:
                raise Exception("No data available")
                
        except Exception:
            mock_price = 100.0 + (hash(ticker) % 500)
            return jsonify({
                'symbol': ticker,
                'name': stock_info['name'],
                'sector': stock_info['sector'],
                'exchange': stock_info['exchange'],
                'country': stock_info['country'],
                'currentPrice': round(mock_price, 2),
                'change': round(np.random.uniform(-5, 5), 2),
                'changePercent': round(np.random.uniform(-3, 3), 2),
                'dayHigh': round(mock_price * 1.02, 2),
                'dayLow': round(mock_price * 0.98, 2),
                'volume': np.random.randint(100000, 10000000),
                'dataSource': 'simulated'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_hybrid/<ticker>', methods=['POST'])
def api_train_hybrid(ticker: str):
    """Train hybrid LSTM-ARIMA model"""
    try:
        ticker = ticker.upper()
        
        if ticker not in STOCK_UNIVERSE:
            return jsonify({'error': 'Stock not found', 'success': False}), 404
        
        if not TENSORFLOW_AVAILABLE:
            return jsonify({
                'error': 'TensorFlow not available',
                'success': False
            }), 500
        
        result = hybrid_predictor.train_hybrid_model(ticker, epochs=50)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/predict_hybrid/<ticker>')
def api_predict_hybrid(ticker: str):
    """Get hybrid LSTM-ARIMA predictions"""
    try:
        ticker = ticker.upper()
        days = request.args.get('days', 30, type=int)
        days = max(1, min(days, 90))
        
        if ticker not in STOCK_UNIVERSE:
            return jsonify({'error': 'Stock not found'}), 404
        
        result = hybrid_predictor.predict_hybrid(ticker, days)
        
        if not result.get('success', False):
            return jsonify(result), 400
        
        stock_info = STOCK_UNIVERSE[ticker]

        # MSE requires matching actual and predicted values.
        # Do not report a misleading score for future forecasts.
        mse = None
        
        # Response with only Mean Squared Error
        
        return jsonify({
            **result,
            'symbol': ticker,
            'mean_squared_error': mse,
            'timestamp': datetime.now().isoformat()
        })

        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical/<ticker>')
def api_historical(ticker: str):
    """Get historical stock data"""
    try:
        ticker = ticker.upper()
        period = request.args.get('period', '1y')
        
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        
        if data is None or data.empty:
            return jsonify({'error': 'No data available'}), 404
        
        historical = []
        for index, row in data.iterrows():
            historical.append({
                'date': index.strftime('%Y-%m-%d'),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        
        return jsonify({
            'symbol': ticker,
            'period': period,
            'data': historical
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    print("🚀 Starting AI Trading Platform")
    print("=" * 60)
    print(f"🧠 Engine: Hybrid LSTM-ARIMA")
    print(f"📊 Stocks: {len(STOCK_UNIVERSE)}")
    print(f"🔥 TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
    print(f"🌐 Server: http://localhost:5000")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")

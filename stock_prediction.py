import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from prophet import Prophet
import concurrent.futures
import time
from data_cleaning import data_cleaning
import logging
from utils import check_internet_connection
from plot_prediction import plot_predictions
import os

# Optional: Enable mixed precision for GPU speedup (uncomment if you have a compatible GPU)
# mixed_precision.set_global_policy('mixed_float16')

# Setup logging
if not os.path.exists('log'):
    os.makedirs('log')
logging.basicConfig(filename='log/stock_prediction.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize session state
if 'predictions_cache' not in st.session_state:
    st.session_state['predictions_cache'] = {}

@st.cache_data
def get_symbol_from_name(company_name):
    """Cached symbol lookup to avoid repeated API calls"""
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return None
        search_result = yq.search(company_name)
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote and (quote['symbol'].endswith(".NS")):
                    return quote['symbol']
        st.error("This stock is not listed in NSE(National Stock Exchange).")
        return -1
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in get_symbol_from_name: {e}")
        return None

def build_rnn_model(input_shape, kind="gru"):
    """
    Build optimized RNN model for faster training
    kind: 'gru' (faster default) or 'lstm' (cuDNN-friendly)
    """
    try:
        model = Sequential([Input(shape=input_shape)])
        
        if kind == "gru":
            # Compact and fast GRU stack
            model.add(GRU(96, return_sequences=True))   # dropout=0 for GPU speed
            model.add(Dropout(0.2))
            model.add(GRU(48, return_sequences=False))
            model.add(Dropout(0.2))
        else:
            # cuDNN-friendly LSTM stack: keep dropout=0 and recurrent_dropout=0
            model.add(LSTM(96, return_sequences=True, dropout=0.0, recurrent_dropout=0.0))
            model.add(Dropout(0.2))
            model.add(LSTM(48, return_sequences=False, dropout=0.0, recurrent_dropout=0.0))
            model.add(Dropout(0.2))
        
        # Output layer (use dtype='float32' if mixed precision is enabled)
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
        return model
    except Exception as e:
        logging.error(f"Error in build_rnn_model: {e}")
        raise

@st.cache_data
def make_windows(data_2d, window=90):
    """
    Optimized window creation using numpy's sliding_window_view when available
    """
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data_2d, (window, data_2d.shape[1]))[:, 0, :, :]
        x = windows[:-1]
        y = data_2d[window:, 0]
        return x, y
    except Exception:
        # Fallback for older numpy versions
        x, y = [], []
        for i in range(window, len(data_2d)):
            x.append(data_2d[i-window:i])
            y.append(data_2d[i, 0])
        return np.array(x), np.array(y)

def rnn_prediction(scaled_data, scaler, training_data_len, window=90, kind="gru"):
    """
    Fast RNN prediction with early stopping and learning rate scheduling
    """
    try:
        # Split train/test sequences
        train_data = scaled_data[:training_data_len]
        x_train, y_train = make_windows(train_data, window=window)

        if x_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty. Ensure there is enough data for training.")

        model = build_rnn_model((x_train.shape[1], x_train.shape[2]), kind=kind)

        # Advanced callbacks for faster convergence
        early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, min_lr=1e-6, verbose=0)

        # Train with validation monitoring
        history = model.fit(
            x_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=128,  # Larger batch for speed
            verbose=0,
            callbacks=[early, reduce_lr]
        )

        # Build test windows
        test_data = scaled_data[training_data_len - window:]
        x_test, _ = make_windows(test_data, window=window)

        # Predict last 120 steps if available
        predictions_to_make = min(120, len(x_test))
        if predictions_to_make > 0:
            y_pred_scaled = model.predict(x_test[-predictions_to_make:], verbose=0)
            preds = scaler.inverse_transform(y_pred_scaled).flatten()
            
            # Pad with last prediction if needed
            if len(preds) < 120:
                padding = np.full(120 - len(preds), preds[-1] if len(preds) > 0 else 0)
                preds = np.concatenate([padding, preds])
            
            return preds[:120]
        else:
            # Fallback prediction
            last_sequence = scaled_data[-window:].reshape(1, window, -1)
            pred_scaled = model.predict(last_sequence, verbose=0)
            pred = scaler.inverse_transform(pred_scaled).flatten()[0]
            return np.full(120, pred)
            
    except Exception as e:
        logging.error(f"Error in rnn_prediction: {e}")
        raise

@st.cache_data
def prophet_prediction(stock_data):
    """Cached Prophet prediction to avoid recomputation"""
    try:
        if 'Close' not in stock_data.columns:
            raise ValueError("Stock data must contain a 'Close' column")
        
        stock_data.index = pd.to_datetime(stock_data.index)
        df = pd.DataFrame({
            'ds': stock_data.index, 
            'y': stock_data['Close'].iloc[:, 0] if stock_data['Close'].ndim > 1 else stock_data['Close']
        })
        
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # More conservative changepoints
            seasonality_prior_scale=10.0
        )
        
        # Suppress Prophet's verbose output
        with st.spinner("Training Prophet model..."):
            prophet_model.fit(df)
        
        future = prophet_model.make_future_dataframe(periods=120)
        forecast = prophet_model.predict(future)
        return forecast['yhat'].tail(120).values
        
    except Exception as e:
        logging.error(f"Error in prophet_prediction: {e}")
        raise

@st.cache_data
def download_and_clean_data(symbol, start_date="2023-01-01", end_date="2025-12-12"):
    """Cache data download and cleaning to avoid repeated processing"""
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    scaled_data, scaler, training_data_len = data_cleaning(stock_data)
    return stock_data, scaled_data, scaler, training_data_len

def display_stock_prediction():
    """Main Streamlit app with performance optimizations"""
    try:
        if 'predictions_cache' not in st.session_state:
            st.session_state['predictions_cache'] = {}
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
            
        st.session_state.input_text = ""
        
        # Model selection
        col1, col2 = st.columns([3, 1])
        with col1:
            company_name = st.text_input("Enter Company Name:", value=st.session_state.get('company_name', ''))
        with col2:
            model_type = st.selectbox("Model:", ["GRU (Faster)", "LSTM (Traditional)"], index=0)
        
        rnn_kind = "gru" if "GRU" in model_type else "lstm"
        
        if st.button("Predict") or company_name:
            if not check_internet_connection():
                st.error("No internet connection. Please check your connection and try again.")
                return
                
            st.session_state.company_name = company_name
            
            # Get stock symbol
            symbol = get_symbol_from_name(company_name)
            if not symbol or symbol == -1:
                return
            
            # Check cache first
            cache_key = f"{company_name}_{rnn_kind}"
            if cache_key in st.session_state['predictions_cache']:
                st.success("Using cached predictions.")
                cached_data = st.session_state['predictions_cache'][cache_key]
                plot_predictions(
                    cached_data['historical_data'], 
                    cached_data['combined_predictions'], 
                    symbol, 
                    company_name, 
                    False
                )
                return
            
            st.write(f"The company is listed on the NSE, and its stock symbol is '{symbol}'.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            
            # Download and clean data (cached)
            with st.spinner("Downloading and cleaning data..."):
                stock_data, scaled_data, scaler, training_data_len = download_and_clean_data(symbol)
                progress_bar.progress(0.3)
            
            # Train models in parallel
            with st.spinner(f"Training {model_type} and Prophet models..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks
                    rnn_future = executor.submit(
                        rnn_prediction, 
                        scaled_data, 
                        scaler, 
                        training_data_len, 
                        90,  # window size
                        rnn_kind
                    )
                    prophet_future = executor.submit(prophet_prediction, stock_data)
                    
                    # Get results
                    rnn_predictions = rnn_future.result()
                    prophet_predictions = prophet_future.result()
                
                progress_bar.progress(0.8)
            
            # Combine predictions
            with st.spinner("Generating final predictions..."):
                rnn_length = len(rnn_predictions)
                prophet_length = len(prophet_predictions)
                min_length = min(rnn_length, prophet_length)
                
                # Ensemble prediction (weighted average)
                rnn_weight = 0.6  # Give slightly more weight to RNN
                prophet_weight = 0.4
                
                combined_predictions = (
                    rnn_weight * rnn_predictions[-min_length:] + 
                    prophet_weight * prophet_predictions[-min_length:]
                )
                
                # Cache results
                st.session_state['predictions_cache'][cache_key] = {
                    'historical_data': stock_data[-120:],
                    'combined_predictions': combined_predictions,
                    'rnn_predictions': rnn_predictions,
                    'prophet_predictions': prophet_predictions
                }
                
                progress_bar.progress(1.0)
            
            # Display results
            plot_predictions(
                stock_data[-120:], 
                combined_predictions, 
                symbol, 
                company_name, 
                False
            )
            
            # Show model performance info
            with st.expander("Model Performance Details"):
                st.write(f"**Model Used**: {model_type}")
                st.write(f"**RNN Weight**: {rnn_weight}")
                st.write(f"**Prophet Weight**: {prophet_weight}")
                st.write(f"**Prediction Length**: {len(combined_predictions)} days")
                
            # Clear old cache entries to manage memory
            if len(st.session_state['predictions_cache']) > 5:
                oldest_key = list(st.session_state['predictions_cache'].keys())[0]
                del st.session_state['predictions_cache'][oldest_key]
                
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in display_stock_prediction: {e}")

# Main app entry point
if __name__ == "__main__":
    st.title("Stock Price Prediction")
    st.markdown("### Fast AI-Powered Stock Forecasting")
    
    display_stock_prediction()
    
    # Add sidebar with performance tips
    with st.sidebar:
        st.header("Performance Tips")
        st.info("""
        **GRU vs LSTM:**
        - GRU: Faster training, good accuracy
        - LSTM: Traditional choice, slightly more parameters
        
        **Caching:**
        - Results are cached for faster repeat predictions
        - Cache automatically manages memory
        """)

import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import concurrent.futures
import time
from scipy.ndimage import gaussian_filter1d
if 'predictions_cache' not in st.session_state:
    st.session_state['predictions_cache'] = {}


def get_symbol_from_name(company_name):
    try:
        search_result = yq.search(company_name)
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote and (quote['symbol'].endswith(".NS")):
                    return quote['symbol'], quote.get('exchange', 'Unknown')
        st.error("This stock is not listed in NSE(National Stock Exchange).")
        return -1, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prophet_prediction(stock_data):
    df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data['Close']})
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=120)
    forecast = prophet_model.predict(future)
    return forecast['yhat'].tail(120).values

def arima_prediction(stock_data):
    arima_model = ARIMA(stock_data['Close'], order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    forecast = arima_model_fit.forecast(steps=120)
    return forecast.values
def lstm_prediction(scaled_data, scaler, training_data_len):
    stop_training = st.session_state.get('stop_training', False)  
    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    
    for i in range(120, len(train_data)):
        stop_training = st.session_state.get('stop_training', False)
        if stop_training:
            return None  
        x_train.append(train_data[i-120:i])
        y_train.append(train_data[i, 0])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    lstm_model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
    
 
    early_stopping = EarlyStopping(monitor='loss', patience=15)
    lstm_model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, callbacks=[early_stopping])
    
    test_data = scaled_data[training_data_len - 120:]
    x_test = [test_data[i-120:i] for i in range(120, len(test_data))]
    x_test = np.array(x_test).reshape(-1, 120, test_data.shape[1])
    
    predictions = lstm_model.predict(x_test)
    return scaler.inverse_transform(predictions).flatten()
def detect_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data < lower_bound) | (data > upper_bound)]

def display_stock_prediction():
    if 'predictions_cache' not in st.session_state:
        st.session_state['predictions_cache'] = {}
    
    if 'stop_training' not in st.session_state:
        st.session_state['stop_training'] = False 
    company_name = st.text_input("Enter Company Name:", value=st.session_state.get('company_name', ''))
    
    
    
    if st.button("Predict"):
        st.session_state.company_name = company_name
        symbol, exchange = get_symbol_from_name(company_name)
        stop_button = st.button("Stop Training")
        if stop_button:
           st.session_state['stop_training'] = True  
           st.warning("Training stopped!")
           display_stock_prediction()

        if not symbol or symbol == -1:
            return

        if company_name in st.session_state['predictions_cache']:
            st.success("Using cached predictions.")
            cached_data = st.session_state['predictions_cache'][company_name]
            plot_predictions(cached_data['historical_data'], cached_data['combined_predictions'], symbol)
            return

        start_date = "2023-01-01"
        end_date = "2025-12-12"
        
        progress_bar = st.progress(0)
        
        with st.spinner("Downloading data..."):
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            if len(stock_data) < 120:
               st.write("Not enough data available for this symbol. At least 120 rows are required.")
               return
            progress_bar.progress(0.25)
        if stock_data.isnull().values.any():
            st.warning("Data contains null values. Performing data cleaning...")
            stock_data = stock_data.dropna()
            if stock_data.empty:
                st.error("After cleaning, no data is available for this symbol.")
                return
        
        outliers = detect_outliers(stock_data['Close'])
        if not outliers.empty:
            st.warning("Outliers detected in closing prices:")
            stock_data = stock_data[~stock_data['Close'].isin(outliers)]
        

        # Feature scaling
        data = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_data_len = int(np.ceil(0.8 * len(scaled_data)))

        with st.spinner("Training models..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                lstm_future = executor.submit(lstm_prediction, scaled_data, scaler, training_data_len)
                prophet_future = executor.submit(prophet_prediction, stock_data)
                arima_future = executor.submit(arima_prediction, stock_data)

                # Prepare data for XGBoost
                xgb_train_data = scaled_data[:training_data_len]
                xgb_target_data = scaled_data[120:training_data_len + 120]  # Adjust according to your target variable

                # Add XGBoost prediction to the executor
                xgb_future = executor.submit(xgboost_prediction, xgb_train_data, xgb_target_data)

                lstm_predictions = lstm_future.result()
                prophet_predictions = prophet_future.result()
                arima_predictions = arima_future.result()
                xgb_predictions = xgb_future.result()

            # Combine predictions (modify this as needed)
            combined_predictions = np.mean([lstm_predictions[-min_length:],
                                            prophet_predictions[-min_length:],
                                            arima_predictions[-min_length:],
                                            xgb_predictions[-min_length:]], axis=0)

            st.session_state['predictions_cache'][company_name] = {
                'historical_data': stock_data[-120:],
                'combined_predictions': combined_predictions
            }

            plot_predictions(stock_data[-120:], combined_predictions, symbol)
        
        progress_bar.progress(1.0)
        hide_button_style = """
    <style>
    button[aria-label="Stop Training"] {
        display: none;
    }
    </style>
"""
        st.markdown(hide_button_style, unsafe_allow_html=True)



def smooth_predictions(predictions, sigma=2):
    """Smooth the predictions using a Gaussian filter."""
    return gaussian_filter1d(predictions, sigma=sigma)

def plot_predictions(historical_data, combined_predictions, symbol):
    transition_days = 1  # Define the number of days for the transition state
    extended_historical_prices = historical_data['Close'].values[-transition_days:]  # Historical prices for transition
    transition_predictions = combined_predictions[:transition_days]
    smooth_transition = np.concatenate([extended_historical_prices, transition_predictions])
    smoothed_combined_predictions = smooth_predictions(combined_predictions)

    # Generate future dates starting from the last date in historical data
    future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(combined_predictions), freq='B')
    
    # Combine historical and future dates for plotting
    merged_future_dates = np.concatenate([historical_data.index[-transition_days:], future_dates])
    merged_smoothed_predictions = np.concatenate([smooth_transition, smoothed_combined_predictions[transition_days:]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'].values,
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2),
        marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=merged_future_dates,
        y=merged_smoothed_predictions,
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Stock Price Prediction for {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(tickangle=-45),
        hovermode="x unified",
        width=1500,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    display_stock_prediction()
import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from prophet import Prophet
import concurrent.futures
import time
from data_cleaning import data_cleaning
import logging
from utils import check_internet_connection
from plot_prediction import plot_predictions
import os

if not os.path.exists('log'):
    os.makedirs('log')
logging.basicConfig(filename='log/stock_prediction.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
if 'predictions_cache' not in st.session_state:
    st.session_state['predictions_cache'] = {}

def get_symbol_from_name(company_name):
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

def build_lstm_model(input_shape):
    try:
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
    except Exception as e:
        logging.error(f"Error in build_lstm_model: {e}")
        raise

def prophet_prediction(stock_data):
    try:
        if 'Close' not in stock_data.columns:
            raise ValueError("Stock data must contain a 'Close' column")
        stock_data.index = pd.to_datetime(stock_data.index)
        df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data['Close'].iloc[:, 0]})
        prophet_model = Prophet()
        prophet_model.fit(df)
        future = prophet_model.make_future_dataframe(periods=120)
        forecast = prophet_model.predict(future)
        return forecast['yhat'].tail(120).values
    except Exception as e:
        logging.error(f"Error in prophet_prediction: {e}")
        raise

def lstm_prediction(scaled_data, scaler, training_data_len):
    try:
        train_data = scaled_data[:training_data_len]
        x_train, y_train = [], []
        for i in range(120, len(train_data)):
            x_train.append(train_data[i-120:i])
            y_train.append(train_data[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if x_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty. Ensure there is enough data for training.")
        lstm_model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=15)
        lstm_model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, callbacks=[early_stopping])
        test_data = scaled_data[training_data_len - 120:]
        x_test = [test_data[i-120:i] for i in range(120, len(test_data))]
        x_test = np.array(x_test).reshape(-1, 120, test_data.shape[1])
        predictions = lstm_model.predict(x_test[-120:])
        return scaler.inverse_transform(predictions).flatten()
    except Exception as e:
        logging.error(f"Error in lstm_prediction: {e}")
        raise

def display_stock_prediction():
    try:
        if 'predictions_cache' not in st.session_state:
            st.session_state['predictions_cache'] = {}
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
        st.session_state.input_text = ""
        company_name = st.text_input("Enter Company Name:", value=st.session_state.get('company_name', ''))
        if st.button("Predict") or company_name:
            if not check_internet_connection():
                st.error("No internet connection. Please check your connection and try again.")
                return
            st.session_state.company_name = company_name
            symbol = get_symbol_from_name(company_name)
            st.write(symbol)
            if not symbol or symbol == -1:
                return
            if company_name in st.session_state['predictions_cache']:
                st.success("Using cached predictions.")
                cached_data = st.session_state['predictions_cache'][company_name]
                plot_predictions(cached_data['historical_data'], cached_data['combined_predictions'], symbol, company_name, False)
                return
            st.write(f"The company is listed on the NSE, and its stock symbol is '{symbol}'.")
            progress_bar = st.progress(0)
            with st.spinner("Downloading data..."):
                stock_data = yf.download(symbol, start="2023-01-01", end="2025-12-12")
                progress_bar.progress(0.25)
            with st.spinner("Cleaning data..."):
                time.sleep(1)
                scaled_data, scaler, training_data_len = data_cleaning(stock_data)
                progress_bar.progress(0.35)
            with st.spinner("Training models..."):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    lstm_future = executor.submit(lstm_prediction, scaled_data, scaler, training_data_len)
                    prophet_future = executor.submit(prophet_prediction, stock_data)
                    lstm_predictions = lstm_future.result()
                    prophet_predictions = prophet_future.result()
                progress_bar.progress(0.75)
            with st.spinner("Plotting Graph..."):
                lstm_length = len(lstm_predictions)
                prophet_length = len(prophet_predictions)
                time.sleep(2)
                min_length = min(lstm_length, prophet_length)
                combined_predictions = np.mean([
                    lstm_predictions[-min_length:],
                    prophet_predictions[-min_length:],
                ], axis=0)
                st.session_state['predictions_cache'][company_name] = {
                    'historical_data': stock_data[-120:],
                    'combined_predictions': combined_predictions
                }
            plot_predictions(stock_data[-120:], combined_predictions, symbol, company_name, False)
            progress_bar.progress(1.0)
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in display_stock_prediction: {e}")

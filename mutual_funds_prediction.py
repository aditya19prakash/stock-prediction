import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import concurrent.futures
from datetime import datetime

if 'predictions_cache' not in st.session_state:
    st.session_state['predictions_cache'] = {}

def get_symbol_from_name(company_name):
    try:
        search_result = yq.search(company_name)
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote and (quote['symbol'].endswith(".BO")):
                    return quote['symbol'], quote.get('exchange', 'Unknown')
        st.error("No Indian stock symbol found for the given company.")
        return -1, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def build_lstm_model(input_shape):
    model = Sequential([
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

def lstm_prediction(scaled_data, scaler, training_data_len):
    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(120, len(train_data)):
        x_train.append(train_data[i-120:i])
        y_train.append(train_data[i, 0])  
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2])) 
    lstm_model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
    lstm_model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1)
    test_data = scaled_data[training_data_len - 120:]
    x_test = [test_data[i-120:i] for i in range(120, len(test_data))]
    x_test = np.array(x_test).reshape(-1, 120, test_data.shape[1])  
    predictions = lstm_model.predict(x_test)
    return scaler.inverse_transform(predictions).flatten()

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
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]
def display_mutual_funds_prediction():
    if 'predictions_cache' not in st.session_state:
      st.session_state['predictions_cache'] = {}
    company_name = st.text_input("Enter Company Name:", value=st.session_state.get('company_name', ''))

    if st.button("Predict"):
        st.session_state.company_name = company_name
        symbol, exchange = get_symbol_from_name(company_name)

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
            progress_bar.progress(0.25)
        if stock_data.empty:
            st.write("No data available for this symbol.")
            return
        

        stock_data.dropna(inplace=True)
        data = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_data_len = int(np.ceil(0.8 * len(scaled_data)))

        with st.spinner("Training models..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                lstm_future = executor.submit(lstm_prediction, scaled_data, scaler, training_data_len)
                prophet_future = executor.submit(prophet_prediction, stock_data)
                arima_future = executor.submit(arima_prediction, stock_data)

                lstm_predictions = lstm_future.result()
                prophet_predictions = prophet_future.result()
                arima_predictions = arima_future.result()
                progress_bar.progress(0.75)

        min_length = min(len(lstm_predictions), len(prophet_predictions), len(arima_predictions))
        combined_predictions = np.mean([
            lstm_predictions[-min_length:],
            prophet_predictions[-min_length:],
            arima_predictions[-min_length:]
        ], axis=0)
        
     
        st.session_state['predictions_cache'][company_name] = {
            'historical_data': stock_data[-120:],
            'combined_predictions': combined_predictions
        }

        plot_predictions(stock_data[-120:], combined_predictions, symbol)
        progress_bar.progress(1.0)
        sip_calculator(symbol)

def plot_predictions(historical_data, combined_predictions, symbol):
   
    transition_days = 7  #
    extended_historical_prices = historical_data['Close'].values[-transition_days:]
    transition_predictions = combined_predictions[:transition_days]
    smooth_transition = np.concatenate([extended_historical_prices, transition_predictions])
    future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(combined_predictions), freq='B')
    transition_dates = np.concatenate([historical_data.index[-transition_days:], future_dates[:transition_days]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'].values,
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=transition_dates,
        y=smooth_transition,
        mode='lines',
        name='Transition (Historical + Predicted)',
        line=dict(color='green', dash='dash', width=2)
    ))

  
    remaining_future_dates = future_dates[transition_days:]
    remaining_predictions = combined_predictions[transition_days:]

    fig.add_trace(go.Scatter(
        x=remaining_future_dates,
        y=remaining_predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='orange', width=2,),
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

def sip_calculator(symbol):
    st.subheader("SIP Calculator")
    start_date = "2023-01-01"
    end_date = "2024-11-30"
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    sip_data = stock_data.iloc[-252:]

    monthly_investment = st.number_input(
        "Monthly Investment Amount (₹):", 
        min_value=1000, 
        step=500, 
        value=10000, 
        key=f"sip_calculator_{symbol}"
    )

    years = 3
    initial_price = sip_data['Close'].iloc[0]
    final_price = sip_data['Close'].iloc[-1]
    annual_return = ((final_price - initial_price) / initial_price) * 100
    st.write(f"Estimated Annual Return: {annual_return:.2f}%")

    P = float(monthly_investment)
    R = float(annual_return)
    T = float(years)
    r = R / (12 * 100)
    n = T * 12
    FV = P * (((1 + r)**n - 1) / r) * (1 + r)
    st.write(f"Future Value after {years} years based on predicted prices: ₹{FV:.2f}")

if __name__ == "__main__":
    display_mutual_funds_prediction()
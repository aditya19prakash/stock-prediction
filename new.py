import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import concurrent.futures
from datetime import datetime

# Function to get symbol from company name
def get_symbol_from_name(company_name):
    try:
        # Search for the company
        search_result = yq.search(company_name)

        # Check for valid search results
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote:
                    symbol = quote['symbol']
                    exchange = quote.get('exchange', '')
                    
                    # Only return if the symbol belongs to Indian stock markets
                    if symbol.endswith(".NS") or symbol.endswith(".BO"):
                        return symbol, exchange

        # Raise error if no valid Indian stock symbol is found
        st.error("No Indian stock symbol found for the given company.")
        return None, None
    except Exception as e:
        # Handle errors gracefully
        st.error(f"Error: {e}")
        return None, None

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Detect outliers
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

# LSTM Model Prediction
def lstm_prediction(scaled_data, scaler, training_data_len):
    train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(120, len(train_data)):
        x_train.append(train_data[i-120:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    lstm_model = build_lstm_model((x_train.shape[1], 1))
    lstm_model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)  # Reduced epochs

    test_data = scaled_data[training_data_len - 120:]
    x_test = np.array([test_data[i-120:i, 0] for i in range(120, len(test_data))]).reshape(-1, 120, 1)
    predictions = lstm_model.predict(x_test)
    return scaler.inverse_transform(predictions).flatten()

# Prophet Model Prediction
def prophet_prediction(stock_data):
    stock_data.index = pd.to_datetime(stock_data.index)
    df = pd.DataFrame({'ds': stock_data.index, 'y': stock_data['Close']})
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=120)
    forecast = prophet_model.predict(future)
    return forecast['yhat'].tail(120).values

# ARIMA Model Prediction
def arima_prediction(stock_data):
    arima_model = ARIMA(stock_data['Close'], order=(5, 1, 0))
    arima_model_fit = arima_model.fit()
    return arima_model_fit.forecast(steps=120).values

# Display Predictions
def display_predictions():
    if 'predicted_prices' not in st.session_state:
        st.session_state.predicted_prices = None

    company_name = st.text_input("Enter Company Name:", value=st.session_state.get('company_name', ''))

    if st.button("Predict"):
        st.session_state.company_name = company_name
        symbol, exchange = get_symbol_from_name(company_name)

        if not symbol:
            st.warning("Could not find a valid stock symbol for the entered company name.")
            return

        st.subheader(f"Prediction for {company_name}")
        st.write(f"The symbol of the company is {symbol} and it's listed on {exchange}")

        start_date = "2023-01-01"
        end_date = "2024-11-30"
        
        progress_bar = st.progress(0)

        with st.spinner("Downloading data..."):
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            progress_bar.progress(0.25)

        if stock_data.empty:
            st.write("No data available for this symbol.")
            return

        if stock_data.isnull().values.any():
            st.warning("Data contains null values. Performing data cleaning...")
            stock_data = stock_data.dropna()
            if stock_data.empty:
                st.error("After cleaning, no data is available for this symbol.")
                return
        
        outliers = detect_outliers(stock_data['Close'])
        if not outliers.empty:
            stock_data = stock_data[~stock_data['Close'].isin(outliers)]

        data = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_data_len = int(np.ceil(0.8 * len(scaled_data)))
        if len(scaled_data) < 120:
            st.error("Not enough data points for training the model.")
            return

        # Parallel training
        with st.spinner("Training models..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                lstm_future = executor.submit(lstm_prediction, scaled_data, scaler, training_data_len)
                prophet_future = executor.submit(prophet_prediction, stock_data)
                arima_future = executor.submit(arima_prediction, stock_data)

                lstm_predictions = lstm_future.result()
                prophet_predictions = prophet_future.result()
                arima_predictions = arima_future.result()
                progress_bar.progress(0.75)

        # Combine Predictions
        combined_predictions = np.mean([lstm_predictions[-120:], prophet_predictions, arima_predictions], axis=0)

        # Plot Combined Predictions
        fig = go.Figure()
        last_60_days_data = stock_data[-120:]

        fig.add_trace(go.Scatter(
            x=last_60_days_data.index,
            y=last_60_days_data['Close'].values,
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2),
        ))
        today = datetime.now().date()
        future_dates = pd.date_range(start=today, periods=120, freq='B')
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=combined_predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='orange', width=2),
            marker=dict(size=6),
        ))

        fig.update_layout(
            title=f"Stock Price Prediction for {symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis=dict(tickangle=-45),
            yaxis_tickformat='.2f',
            hovermode="x unified",
            width=1500,
            height=700
        )
        fig.update_traces(
            hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>",
        )

        st.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(1.0)

        sip_calculator(stock_data, symbol)
def combine_predictions(lstm_predictions, prophet_predictions, arima_predictions):
    # Ensure all predictions are of the same length (120)
    predictions_len = 120
    
    # Trim predictions if necessary
    lstm_predictions = lstm_predictions[-predictions_len:]
    prophet_predictions = prophet_predictions[-predictions_len:]
    arima_predictions = arima_predictions[-predictions_len:]

    # Combine predictions by averaging them
    combined_predictions = np.mean([lstm_predictions, prophet_predictions, arima_predictions], axis=0)
    return combined_predictions

def sip_calculator(stock_data, symbol):
    st.subheader("SIP Calculator")
    start_date = "2023-01-01"
    end_date = "2024-11-30"
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    sip_data = stock_data.iloc[-252:]
    monthly_investment = st.number_input("Monthly Investment Amount (₹):", min_value=1000, step=500, value=10000)
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
    display_predictions()
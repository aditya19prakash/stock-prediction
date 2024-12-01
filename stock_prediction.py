import streamlit as st
import yfinance as yf
import yahooquery as yq
import numpy as np
import pandas as pd
from prophet import Prophet
import time
import plotly.graph_objects as go
import datetime
from yahoo_api_handler import get_symbol_from_name
def fetch_stock_news(symbol):
    try:
        ticker = yq.Ticker(symbol)
        news = ticker.news(limit=5)  # Fetch the latest 5 news articles
        return news
    except Exception as e:
        return []
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

# Function to display stock prediction
def display_stock_prediction():
    if 'predicted_prices' not in st.session_state:
      st.session_state.predicted_prices = None

    company_name = st.text_input("Enter Company Name:")

    if st.button("Predict"):
        st.session_state.company_name = company_name
        symbol, exchange = get_symbol_from_name(company_name)
        if not symbol:
            st.warning("Could not find a valid stock symbol for the entered company name.")
            return

        st.subheader(f"Prediction for {company_name}")
        st.write(f"The symbol of the company is {symbol} and it's listed in {exchange}")
        st.subheader("Latest News")
        news = fetch_stock_news(symbol)
        if news:
            for article in news:
                st.markdown(f"**[{article['title']}]({article['link']})**")
                st.write(f"*Source: {article['publisher']} | Published: {datetime.datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}*")
                st.write("---")
        else:
            st.write("No news articles found for this stock.")
        start_date = "2021-06-01"
        end_date = "2024-11-30"
       

        progress_bar = st.progress(0)

        # Downloading data progress
        with st.spinner("Downloading data..."):
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            time.sleep(1)
            progress_bar.progress(0.25)
            print(stock_data)

        if stock_data.empty:
            st.write("No data available for this symbol.")
            return

      
        # Data Cleaning: Check for null values
        if stock_data.isnull().values.any():
            st.warning("Data contains null values. Performing data cleaning...")
            stock_data = stock_data.dropna()
            if stock_data.empty:
                st.error("After cleaning, no data is available for this symbol.")
                return

        outliers = detect_outliers(stock_data['Close'])
        if not outliers.empty:
            st.warning(f"Outliers detected in closing prices: {outliers.values}")
            stock_data = stock_data[~stock_data['Close'].isin(outliers)]

        # Prepare data for Prophet
        stock_data = stock_data.reset_index()[['Date', 'Close']]
        stock_data.columns = ['ds', 'y']

        # Initialize and train the Prophet model
        model = Prophet(daily_seasonality=False)
        with st.spinner("Training the model..."):
            model.fit(stock_data)
            progress_bar.progress(0.5)

        # Generate future predictions for 60 business days
        future = model.make_future_dataframe(periods=60, freq='B')
        forecast = model.predict(future)
        future_forecast = forecast.tail(60)  
        progress_bar.progress(0.75)
        
        stock_data = stock_data.tail(60)
        
        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data['ds'],
            y=stock_data['y'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', width=2,dash='dot'),
            marker=dict(size=6),
        ))

        fig.update_layout(
            title=f"Stock Price Prediction for {symbol} (Next 60 Days)",
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

        st.plotly_chart(fig, use_container_width=True, key="stock_chart")
        progress_bar.progress(1.0)
   


if __name__ == "__main__":
    st.title("Stock Price Prediction App")
    display_stock_prediction()
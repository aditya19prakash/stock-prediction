import numpy as np
import streamlit as st
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(filename='data_cleaning.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def detect_outliers(data):
    try:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data < lower_bound) | (data > upper_bound)]
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in detect_outliers: {e}")
        return pd.Series([])

def data_cleaning(stock_data: pd.DataFrame):
    try:
        if len(stock_data) < 120:
            st.error("Not enough data available for this symbol. At least 120 data points are required.")
            return None, None, None
        if stock_data.isnull().values.any():          
            stock_data = stock_data.dropna()
        if stock_data.empty:
            st.error("After cleaning, no data is available for this symbol.")
            return None, None, None
        outliers = detect_outliers(stock_data['Close'])
        if not outliers.empty:
            stock_data = stock_data[~stock_data['Close'].isin(outliers)]
        data = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        training_data_len = int(np.ceil(0.8 * len(scaled_data)))
        return scaled_data, scaler, training_data_len
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in data_cleaning: {e}")
        return None, None, None
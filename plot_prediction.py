import logging
from scipy.ndimage import gaussian_filter1d
from summary import summaryprint
from datetime import datetime
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
import os

# Ensure the log directory exists
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'error.log'), level=logging.ERROR)

def smooth_predictions(predictions, sigma=2):
    try:
        return gaussian_filter1d(predictions, sigma=sigma)
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in smooth_predictions: {e}")
        return predictions

def adjust_predictions(historical_prices, predictions, blend_days=5):
    try:
        if len(historical_prices) == 0 or len(predictions) == 0:
            return predictions
        adjustment = historical_prices[-1] - predictions[0]

        adjusted_predictions = predictions + adjustment

        weights = np.linspace(1, 0, blend_days)
        blended_predictions = adjusted_predictions.copy()
        blended_predictions[:blend_days] = (
            weights * historical_prices[-blend_days:] + (1 - weights) * adjusted_predictions[:blend_days]
        )
        return blended_predictions
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in adjust_predictions: {e}")
        return predictions

def plot_predictions(historical_data, combined_predictions, symbol, company_name, signal):
    try:
        historical_prices = historical_data['Close'].values[-1:].reshape(-1) if historical_data is not None else []
        smoothed_combined_predictions = adjust_predictions(historical_prices, combined_predictions)
        if np.std(combined_predictions) > 0.05:
            smoothed_combined_predictions = smooth_predictions(smoothed_combined_predictions)
        smoothed_combined_predictions = np.round(smoothed_combined_predictions, 2)
        if historical_data is not None:
            historical_120_days = historical_data[-120:]
        else:
            historical_120_days = None
        future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1),
                                     periods=len(combined_predictions), freq='B') if historical_data is not None else pd.date_range(start=datetime.datetime.today(), periods=len(combined_predictions), freq='B')
        merged_future_dates = np.concatenate([historical_data.index[-1:], future_dates]) if historical_data is not None else future_dates
        merged_smoothed_predictions = np.concatenate([historical_prices, smoothed_combined_predictions])
        if len(merged_future_dates) != len(merged_smoothed_predictions):
            st.warning("Length mismatch: Future dates vs. Predictions")
            return
        fig = go.Figure()
        historical_120_days["Close"] = historical_120_days["Close"].astype(int)
        
        merged_smoothed_predictions=merged_smoothed_predictions.astype(int)
        closing_prices = historical_120_days['Close'].squeeze()
        closing_prices=closing_prices.astype(int)
        print(closing_prices)
        if historical_120_days is not None:
            for i in range(1, len(historical_120_days)):
                color = 'red' if closing_prices[i] < closing_prices[i - 1] else 'green'
                fig.add_trace(go.Scatter(
                    x=historical_120_days.index[i-1:i+1],
                    y=closing_prices[i-1:i+1],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        for i in range(1, len(merged_smoothed_predictions)):
            color = 'red' if merged_smoothed_predictions[i] < merged_smoothed_predictions[i - 1] else 'green'
            fig.add_trace(go.Scatter(
                x=merged_future_dates[i-1:i+1],
                y=merged_smoothed_predictions[i-1:i+1],
                mode='lines',
                line=dict(color=color, width=3, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        fig.add_trace(go.Scatter(
            x=historical_120_days.index,
            y=closing_prices,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='── Historical Price (120 Days)',
            hovertemplate="<b>────────────</b><br><b>Historical Price:</b> %{y}<extra></extra>",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=merged_future_dates,
            y=merged_smoothed_predictions,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name=f'------ Predicted Price ({len(merged_smoothed_predictions)} Days)',
            hovertemplate="<b>--------------------------</b><br><b>Predicted Price:</b> %{y}<extra></extra>",
            showlegend=True
        ))
        fig.update_layout(
            title=f"Stock Price Prediction for {symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis=dict(tickangle=-45, title_font=dict(size=24), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
            hovermode="x unified",
            width=1500,
            height=700,
            font=dict(size=24),
            title_font=dict(size=24)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("\n\n")
        summaryprint(company_name,  merged_smoothed_predictions, symbol, signal)
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in plot_predictions: {e}")
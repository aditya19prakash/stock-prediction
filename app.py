import streamlit as st
from stock_prediction import display_stock_prediction
from mutual_funds_prediction import display_mutual_funds_prediction



st.set_page_config(
    page_title="Stock and Mutual Prediction",
    layout="wide",  
    page_icon="logo1.png"  
)


menu = st.sidebar.selectbox("Select Page", ["Stock Prediction", "Mutual Funds Prediction"])


if menu == "Stock Prediction":
    st.markdown("<h1 style='font-size: 42px;'>Stock Prediction</h1>", unsafe_allow_html=True)
    # Use a div to apply specific styles
    st.markdown('<div class="stock-prediction">', unsafe_allow_html=True)
    display_stock_prediction()  # Calls the function from stock_prediction.py
    st.markdown('</div>', unsafe_allow_html=True)
elif menu == "Mutual Funds Prediction":
    st.title("Mutual Funds Prediction")
    # Use a div to apply specific styles
    st.markdown('<div class="mutual-funds-prediction">', unsafe_allow_html=True)
    display_mutual_funds_prediction()  # Calls the function from mutual_funds_prediction.py
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import logging
from stock_prediction import display_stock_prediction
from mutual_funds_prediction import display_mutual_funds_prediction
from utils import check_internet_connection

logging.basicConfig(filename='app.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

st.set_page_config(
    page_title="Predictify",
    layout="wide",
    page_icon="project_logo.png"
)

st.markdown(
    """
    <style>
    body {
        background-color: black !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background-color: #333333 !important;  /* Blackish grey */
    }
    @media print {
        .sidebar .sidebar-content {
            background-color: #333333 !important;  /* Blackish grey */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    menu = st.sidebar.selectbox("Select Page", ["Stock Prediction", "Mutual Funds Prediction"])
except Exception as e:
    st.error(f"Error: {e}")
    logging.error(f"Error in sidebar selectbox: {e}")

try:
    if menu == "Stock Prediction":
        st.markdown("<h1 style='font-size: 42px;'>Stock Prediction</h1>", unsafe_allow_html=True)
        st.markdown('<div class="stock-prediction">', unsafe_allow_html=True)
        try:
            if not check_internet_connection():
             st.error("No internet connection. Please check your connection and try again.")
            else:
             display_stock_prediction() 
        except Exception as e:
            st.error(f"Error: {e}")
            logging.error(f"Error in display_stock_prediction: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif menu == "Mutual Funds Prediction":
        st.title("Mutual Funds Prediction")
        st.markdown('<div class="mutual-funds-prediction">', unsafe_allow_html=True)
        try:
            if not check_internet_connection():
              st.error("No internet connection. Please check your connection and try again.")
            else:
             display_mutual_funds_prediction()
        except Exception as e:
            st.error(f"Error: {e}")
            logging.error(f"Error in display_mutual_funds_prediction: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error: {e}")
    logging.error(f"Error in main menu selection: {e}")

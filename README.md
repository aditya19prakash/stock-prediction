# Stock Price Prediction using LSTM & Prophet

## 🚀 Overview
This project is a **Stock Prediction System** that forecasts future stock prices using **Long Short-Term Memory (LSTM)** and **Prophet (Time-Series Forecasting)** models. The application fetches real-time stock data from **Yahoo Finance** and provides users with interactive predictions through a **Streamlit-powered web app**.

## 🔥 Features
- 📊 Fetches real-time stock data from **Yahoo Finance API**
- 🔮 Predicts stock prices using **LSTM (Deep Learning)** and **Prophet (Time-Series Analysis)**
- 🚀 Interactive **Streamlit UI** for user-friendly experience
- ⚡ Efficient **caching** for faster predictions
- 🛠️ **Robust error handling** and logging for reliability

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, TensorFlow, Yahoo Finance API)
- **Machine Learning** (LSTM, Prophet)
- **Web App** (Streamlit)
- **Concurrency** (ThreadPoolExecutor for parallel processing)

## 📥 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/stock-prediction.git
   cd stock-prediction
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   streamlit run app.py
   ```

## 🔍 Usage
1. Enter the company name in the **Streamlit UI**.
2. The system fetches stock data from Yahoo Finance.
3. Predictions are generated using **LSTM & Prophet** models.
4. Visualized predictions are displayed on an interactive graph.

## 📌 Example Output
![Stock Prediction Chart](assets/stock_prediction.png)

## ⚠️ Troubleshooting
- If **Yahoo Finance API** is not working, ensure you have an active internet connection.
- If stock data is missing, check if the stock is listed on NSE (National Stock Exchange).
- Logging is enabled in `log/stock_prediction.log` for debugging.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Connect with Me
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

**Star ⭐ the repo if you found it helpful!** 🚀


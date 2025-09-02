
📈 Stock Price Trend Prediction with LSTM
🔹 Objective

The goal of this project is to predict future stock prices based on historical data using a Long Short-Term Memory (LSTM) network. We enhance the forecasting performance by integrating technical indicators such as Moving Averages and RSI.

🔹 Tools & Libraries

Python

Pandas, Numpy → data manipulation

yfinance → stock data fetching

Matplotlib, Seaborn → visualization

Scikit-learn → preprocessing, metrics

Keras (TensorFlow backend) → LSTM modeling

Streamlit → interactive dashboard

📂 Project Structure
lstm-stock-price-forecasting/
│── lstm_stock_forecasting.ipynb   # Jupyter notebook (model training & testing)
│── streamlit_app.py               # Streamlit dashboard
│── requirements.txt               # Dependencies
│── README.md                      # Project documentation
│── model/                         # (Optional) Saved LSTM model weights

🔹 Workflow
1️⃣ Data Collection

Historical stock data fetched via Yahoo Finance API (yfinance).

User can select any ticker (e.g., AAPL, MSFT, TSLA, TCS.NS).

2️⃣ Data Preprocessing

Closing price normalized (MinMaxScaler).

Time-series sequences prepared with lookback window.

Train-test split applied.

3️⃣ Feature Engineering

Added Technical Indicators:

Simple Moving Average (SMA)

Relative Strength Index (RSI)

4️⃣ Model Building

LSTM network with:

Input → LSTM(50 units) → Dropout → Dense → Output layer

Optimizer: Adam

Loss: MSE

5️⃣ Training & Validation

Hyperparameters configurable:

Lookback window (30–120 days)

Train-test ratio

Epochs

Batch size

6️⃣ Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

7️⃣ Visualization

Plot of Actual vs Predicted Close Price

Metrics displayed in Streamlit app

🔹 Results

Example with AAPL (Apple Inc.):

RMSE: ~20.28

MAE: ~17.51

MAPE: ~8.06%

📊 The predictions follow the stock price trend, though with smoother curves (due to lookback window & noise reduction).

▶️ Usage
Run Jupyter Notebook (for training & evaluation)
jupyter notebook lstm_stock_forecasting.ipynb

Run Streamlit Dashboard
streamlit run streamlit_app.py

📊 Features

✅ Fetches historical stock prices from Yahoo Finance

✅ Preprocesses & normalizes data for training

✅ Builds and trains an LSTM model

✅ Plots Actual vs Predicted stock prices

✅ Includes Moving Average & RSI indicators

✅ Interactive dashboard using Streamlit

🔹 Deliverables

Jupyter Notebook → lstm_stock_forecasting.ipynb

Trained Model Weights → stored in artifacts/

Prediction Graphs → stored in graphs/

Streamlit Dashboard → streamlit_app.py

README.md → project documentation

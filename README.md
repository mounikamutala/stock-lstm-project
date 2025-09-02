
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

# Stock Price Trend Prediction with LSTM

End-to-end project to forecast stock closing prices using Keras LSTM with indicators (SMA, RSI).

## Project Structure
```
lstm_lstm_stock_project/
├─ lstm_stock_forecasting.ipynb   # Step-by-step notebook
├─ streamlit_app.py               # Optional Streamlit dashboard
├─ artifacts/                     # Saved model, weights, predictions (created at runtime)
├─ graphs/                        # Saved plots (created at runtime)
└─ requirements.txt
```

## Quickstart (Local)
1. **Create & activate a virtual env (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   - Open `lstm_stock_forecasting.ipynb` in Jupyter/VS Code.
   - Set your `TICKER` (e.g., `AAPL` or `TCS.NS`) and run all cells.
   - Artifacts will be saved under `artifacts/` and plots under `graphs/`.

4. **Run the Streamlit dashboard (optional)**
   ```bash
   streamlit run streamlit_app.py
   ```


## Notes
- The model predicts **next-day close** using a rolling window (default 60 days) of features.
- Try increasing `EPOCHS` and `LOOKBACK` for improved results.
- RSI and SMAs are included as features to provide trend & momentum context.
- Make sure your ticker is correct (e.g., NSE tickers like `TCS.NS`).

## Requirements
See `requirements.txt` for exact packages/versions.


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

import streamlit as st
from datetime import date
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('INFY', 'TCS', 'TECHM')  # Updated stocks list
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = pd.date_range(start=TODAY, periods=n_years*252, freq=BDay()).size

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Open'], label="stock_open")
    plt.plot(data['Date'], data['Close'], label="stock_close")
    plt.title('Time Series data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot()

plot_raw_data()

# Forecast with ARIMA model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = ARIMA(df_train['y'], order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.predict(start=len(df_train), end=len(df_train) + period - 1)

# Show and plot forecast
st.subheader('Forecast data')

start_date = df_train['ds'].max()
if pd.isna(start_date):
    start_date = TODAY

forecast_data = pd.DataFrame({'Date': pd.date_range(start=start_date, periods=period, freq=BDay()), 'Forecast': forecast})
st.write(forecast_data.tail())

st.write(f'Forecast plot for {n_years} years')
plt.figure(figsize=(12, 6))
plt.plot(df_train['ds'], df_train['y'], label='Actual')
plt.plot(forecast_data['Date'], forecast_data['Forecast'], label='Forecast')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot()

st.write("Note: ARIMA model assumes stationary data and may not capture certain stock price patterns as effectively as other methods.")

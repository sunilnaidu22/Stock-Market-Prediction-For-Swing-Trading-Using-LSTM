# Stock Market Prediction using LSTM

## Overview
This project leverages **Long Short-Term Memory (LSTM)**, a type of Recurrent Neural Network (RNN), to predict stock prices. It's tailored for **swing traders and day traders**, utilizing historical price data to predict the next day’s stock price movement. Additionally, the **Put-Call Ratio (PCR)** is calculated from NSE option chain data to enhance market trend predictions.

## Features
- **LSTM-based Stock Price Prediction**: Uses a 15-day window of historical stock prices (open/close) to predict the next day's closing price.
- **Market Sentiment Analysis**: Integrates the Put-Call Ratio (PCR) derived from option chain data to predict market direction.
- **Supports Swing and Day Traders**: Provides insights for short-term trading strategies.

## Dataset
- **Stock Price Data**: Historical stock data (15-day open and close prices).
- **Option Chain Data**: Put-Call Ratio calculated from NSE option chain data.
  
### Data Sources
- **Stock Data**: [Source e.g., Yahoo Finance]
- **Option Chain Data**: [NSE website or any relevant source]

## Methodology
1. **Data Preprocessing**:
    - Normalize stock prices and transform them into 15-day sequences for model input.
    - Calculate PCR from the NSE option chain to represent market sentiment.
  
2. **LSTM Model**:
    - Designed to predict stock prices based on time-series data.
    - Input: 15 days of stock price data.
    - Output: Next day’s predicted closing price.
  
3. **Model Training**:
    - Split the data into 80% training and 20% testing.
    - Trained using the **Adam optimizer** with **Mean Squared Error (MSE)** as the loss function.

4. **Performance Evaluation**:
    - Metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**.
    - Directional accuracy of market prediction was enhanced using PCR data.

## Results
- The LSTM model accurately predicted the next day's stock price.
- Incorporating PCR data improved the model’s ability to forecast market direction, benefiting short-term traders.

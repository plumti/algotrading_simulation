# price_prediction_simulation
This is a quick python tutorial on how to create a stock price price simulation.

**Disclaimer: This tutorial is for educational purposes only and should not be interpreted as trading advice. The provided code, timestamps, and datasets are used for visualisation and experimentation purposes and may not represent a realistic trading model.**

By following the steps below, you'll be able to understand the purpose of the code and get started with your stock price analysis project.

## Installation
Assuming you have python (installed)
To begin, you'll need to install the Prophet library. You can find installation instructions for Prophet in the official documentation [here](https://facebook.github.io/prophet/docs/installation.html#python).
Once you have Prophet installed (assuming you have an IDE), you're ready to get started!

## Quick Start
I recommend reading through the quick start guide and trying out the provided code examples. You can find the guide [here](https://facebook.github.io/prophet/docs/quick_start.html#python-api).

## Understanding the Code
The reason why simulation of past-data (non-live) trading is very useful for data analysis is because very quickly you can set up and see your model's accuracy.
If I hadn't done a simulation of past-data, live trading could take days to indicate whether my model is accurate (in terms of predicting whether the price will go up or down).
In this example, the model used is within "prophet_predictions(self, data):" and it is used to predict the price of the next candlestick's price, the "calculate_accuracy" is used to check whether the prediction was correct in terms of signals (e.g going up or down).
When the code has finished executing, you should be able to see all the datasets there.
Have a look at 
[Basic example](./example.ipynb) (for an example).
```bash
import yfinance as yf
import pandas as pd
from prophet import Prophet
import numpy as np

# Use the context manager to redirect stdout and stderr to the null device
class MinuteDataPredictor:

    def __init__(self, symbol, file_path):
        self.symbol = symbol
        self.file_path = file_path
        self.minute_data = None
        self.correct_predictions=0
        self.wrong_predictions=0

    
    def fetch_minute_data(self):
        try:
            # Fetch historical data for the last x amount of days with x amount of minute intervals
            data = yf.download(self.symbol, period='2d', interval='60m')

            # Select only the 'Close' prices and reset the index to include 'Timestamp'
            self.minute_data = data[['Close']].reset_index()

            # Rename columns to match 'Timestamp' and 'Price'
            self.minute_data.rename(columns={'Datetime': 'Timestamp', 'Close': 'Price'}, inplace=True)

            # Format the timestamp column
            self.minute_data['Timestamp'] = self.minute_data['Timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Add empty columns for 'Prediction' and 'Accuracy'
            self.minute_data['Prediction_Prophet'] = None
            self.minute_data['Accuracy_Prophet'] = None
        
        except Exception as e:
            print(f"Error fetching minute data: {e}")

    def prophet_predictions(self, data):

            data = data[['Timestamp', 'Price']].copy()
            data.rename(columns={'Timestamp': 'ds', 'Price': 'y'}, inplace=True)
            model = Prophet()
            model.fit(data)
            future = model.make_future_dataframe(periods=1, freq='min')
            forecast = model.predict(future)
            next_minute_prediction = forecast['yhat'].iloc[-1]
            #print(next_minute_prediction)
            return next_minute_prediction

# Set the logging level to suppress informational messages

        
    def save_minute_data_with_predictions(self):
        try:
            self.fetch_minute_data()

            for i in range(1, len(self.minute_data) - 1):
                previous_data = self.minute_data.iloc[:i + 1].copy()
                
                # Make predictions using Prophet model
                next_minute_prediction_prophet = self.prophet_predictions(previous_data)
                self.minute_data.at[i, 'Prediction_Prophet'] = next_minute_prediction_prophet
                
                
                actual_price = self.minute_data.loc[i, 'Price']
                actual_change = self.minute_data.loc[i + 1, 'Price'] - actual_price
                predicted_change = next_minute_prediction_prophet - actual_price
                
                if (predicted_change > 0 and actual_change > 0 ):
                    self.correct_predictions += 1

                elif (predicted_change < 0 and actual_change < 0 ):
                    self.correct_predictions += 1
                    
                else:
                    self.wrong_predictions+=1
                total = self.correct_predictions+self.wrong_predictions
                if total !=0:
                    
                    accuracy = (self.correct_predictions / total) * 100
                else:
                    accuracy = 0

                self.minute_data.at[i, 'Accuracy_Prophet'] = accuracy
                                   
            self.minute_data.to_csv(self.file_path, index=False)
           # print(f"Minute data with predictions and accuracy based on model saved to {self.file_path}")

        except Exception as e:
            print(f"Error: {e}")
predictor = MinuteDataPredictor(symbol='AAPL', file_path='newminute_data_aapl_with_prediction_oop.csv')

predictor.save_minute_data_with_predictions()
Stock = pd.read_csv('newminute_data_aapl_with_prediction_oop.csv',  index_col=0)
df_Stock = Stock
df_Stock.head(10)
```bash

## Not accurate enough?
If you think these predictions aren't precise enough, try adjust the timestamp and dataset settings, add new models for your price predictions. 
# Sources:
https://facebook.github.io/prophet/
https://algotrading101.com/learn/yfinance-guide/
https://chat.openai.com/chat (GPT-4)
## Enquiries
Contact me via LinkedIn
https://www.linkedin.com/in/eliott-derville/

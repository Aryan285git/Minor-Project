# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pymongo import MongoClient

# Load data into Pandas DataFrame
nifty_data = pd.read_csv('nifty_data.csv', parse_dates=['Date'], index_col='Date')
sensex_data = pd.read_csv('sensex_data.csv', parse_dates=['Date'], index_col='Date')
usd_inr_data = pd.read_csv('usd_inr_data.csv', parse_dates=['Date'], index_col='Date')

# Combine the data into a single DataFrame
combined_data = pd.concat([nifty_data['Close'], sensex_data['Close'], usd_inr_data['Close']], axis=1)
combined_data.columns = ['NIFTY', 'SENSEX', 'USD_INR']

# Check for missing values and fill them if necessary
combined_data = combined_data.ffill()

# Convert columns to numeric
combined_data = combined_data.apply(pd.to_numeric, errors='coerce')

# ARIMA modeling function with additional checks
def arima_model(data, column_name, order):
    try:
        model = ARIMA(data[column_name], order=order)
        results = model.fit(maxiter=1000)  # Set maxiter here
        return results
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError in modeling {column_name}: {e}")
        return None
    except Exception as e:
        print(f"Error in modeling {column_name}: {e}")
        return None

# Perform ARIMA modeling for each column
nifty_results = arima_model(combined_data, 'NIFTY', order=(5, 1, 0))
sensex_results = arima_model(combined_data, 'SENSEX', order=(5, 1, 0))
usd_inr_results = arima_model(combined_data, 'USD_INR', order=(5, 1, 0))

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['market_analysis']
collection = db['arima_results']

# Store ARIMA results in MongoDB
def store_results(results, index_name):
    if results is not None:
        try:
            predictions = results.predict(start='2022-11-01', end='2023-11-01')
            collection.insert_one({
                'index': index_name,
                'params': results.params.tolist() if results.params is not None else None,
                'predicted_values': predictions.tolist()
            })
        except KeyError as e:
            print(f"KeyError during prediction for {index_name}: {e}")

# Store ARIMA results in MongoDB for NIFTY, SENSEX, and USD_INR
store_results(nifty_results, 'NIFTY')
store_results(sensex_results, 'SENSEX')
store_results(usd_inr_results, 'USD_INR')

# Close MongoDB connection
client.close()

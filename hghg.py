import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from pymongo import MongoClient

nifty_data = pd.read_csv('nifty_data.csv')
sensex_data = pd.read_csv('sensex_data.csv')
usd_inr_data = pd.read_csv('usd_inr_data.csv')

#columns_to_convert = ['Open_y', 'High_y', 'Low_y']
col=['Open','High','Low','Close']
#nifty_data[col] = nifty_data[col].apply(pd.to_numeric, errors='coerce')
#sensex_data[col] = sensex_data[col].apply(pd.to_numeric, errors='coerce')

nifty_inr=usd_inr_data.merge(nifty_data, how='outer' , on='Date')
sensex_inr=usd_inr_data.merge(sensex_data, how='outer', on='Date')

nifty_inr.dropna(inplace=True)
sensex_inr.dropna(inplace=True)

print(nifty_inr.info())

nifty_inr['Close_N'] = nifty_inr['Close_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Open_N'] = nifty_inr['Open_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Low_N'] = nifty_inr['Low_N'].replace(',', '', regex=True).astype(float)
nifty_inr['High_N'] = nifty_inr['High_N'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'])

# Set 'Date' as the index
nifty_inr.set_index('Date', inplace=True)

#print(nifty_inr.info())

# Assuming 'nifty_inr' is the merged dataset containing NIFTY and USD/INR rates
# 'date' column represents the common date column

# Splitting the data into training and testing sets
train_size = int(len(nifty_inr) * 0.8)  # 80% for training
#print(train_size)
train_data = nifty_inr.iloc[:train_size]
test_data = nifty_inr.iloc[train_size:]

# Separate endogenous (NIFTY) and exogenous (USD/INR) variables for training and testing
train_endog = train_data[['Open_N', 'High_N', 'Low_N', 'Close_N']]
train_exog = train_data[['Open', 'High', 'Low', 'Close']]

test_endog = test_data[['Open_N', 'High_N', 'Low_N', 'Close_N']]
test_exog = test_data[['Open', 'High', 'Low', 'Close']]


# Convert columns to numeric format
train_endog = train_endog.apply(pd.to_numeric, errors='coerce')
train_exog = train_exog.apply(pd.to_numeric, errors='coerce')
test_endog = test_endog.apply(pd.to_numeric, errors='coerce')
test_exog = test_exog.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
train_endog.dropna(inplace=True)
train_exog.dropna(inplace=True)
test_endog.dropna(inplace=True)
test_exog.dropna(inplace=True)

print(train_endog.shape, train_exog.shape, test_endog.shape, test_exog.shape)

# Print a sample of the data
print(train_endog.head())
print(train_exog.head())
print(test_endog.head())
print(test_exog.head())


endog_column = train_endog.iloc[:, 0]  # Replace 0 with the index of the desired column
model = ARIMA(endog_column, exog=train_exog, order=(1, 0, 0))

# Fit the ARIMAX model
#model = ARIMA(train_endog, exog=train_exog, order=(1, 0, 1))  # Adjust p, d, q values as needed
results = model.fit()

# Make predictions on the testing set
forecast = results.get_forecast(steps=len(test_endog), exog=test_exog)
predicted_values = forecast.predicted_mean

# Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(test_endog['Close_N'], predicted_values)
print(f"Mean Absolute Error: {mae}")

x=nifty_data.iloc[train_size:]

### Visualize the predicted values along with the actual values
plt.figure(figsize=(10, 6))
plt.plot(x['Date'], x['Close_N'], label='Actual NIFTY Close')
plt.plot(x['Date'], predicted_values, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print(sensex_inr.info())

sensex_inr['Close_S'] = sensex_inr['Close_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Open_S'] = sensex_inr['Open_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Low_S'] = sensex_inr['Low_S'].replace(',', '', regex=True).astype(float)
sensex_inr['High_S'] = sensex_inr['High_S'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
sensex_inr['Date'] = pd.to_datetime(sensex_inr['Date'])

# Set 'Date' as the indey
sensex_inr.set_index('Date', inplace=True)

#print(sensex_inr.info())

# Assuming 'sensex_inr' is the merged dataset containing NIFTY and USD/INR rates
# 'date' column represents the common date column

# Splitting the data into training and testing sets
train_size_sensex = int(len(sensex_inr) * 0.8)  # 80% for training
#print(train_size_sensex)
train_data_sensex = sensex_inr.iloc[:train_size_sensex]
test_data_sensex = sensex_inr.iloc[train_size_sensex:]

# Separate endogenous (NIFTY) and exogenous (USD/INR) variables for training and testing
train_endog_sensex = train_data_sensex[['Open_S', 'High_S', 'Low_S', 'Close_S']]
train_exog_sensex = train_data_sensex[['Open', 'High', 'Low', 'Close']]

test_endog_sensex = test_data_sensex[['Open_S', 'High_S', 'Low_S', 'Close_S']]
test_exog_sensex = test_data_sensex[['Open', 'High', 'Low', 'Close']]


# Convert columns to numeric format
train_endog_sensex = train_endog_sensex.apply(pd.to_numeric, errors='coerce')
train_exog_sensex = train_exog_sensex.apply(pd.to_numeric, errors='coerce')
test_endog_sensex = test_endog_sensex.apply(pd.to_numeric, errors='coerce')
test_exog_sensex = test_exog_sensex.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
train_endog_sensex.dropna(inplace=True)
train_exog_sensex.dropna(inplace=True)
test_endog_sensex.dropna(inplace=True)
test_exog_sensex.dropna(inplace=True)

print(train_endog_sensex.shape, train_exog_sensex.shape, test_endog_sensex.shape, test_exog_sensex.shape)

# Print a sample of the data
print(train_endog_sensex.head())
print(train_exog_sensex.head())
print(test_endog_sensex.head())
print(test_exog_sensex.head())


endog_column_sensex = train_endog_sensex.iloc[:, 0]  # Replace 0 with the indey of the desired column
model = ARIMA(endog_column_sensex, exog=train_exog_sensex, order=(5, 1, 0))

# Fit the ARIMAy model
#model = ARIMA(train_endog_sensex, exog=train_exog_sensex, order=(1, 0, 1))  # Adjust p, d, q values as needed
results = model.fit()

# Make predictions on the testing set
forecast_sensex = results.get_forecast(steps=len(test_endog_sensex), exog=test_exog_sensex)
predicted_values_sensex = forecast_sensex.predicted_mean

# Evaluate the model using Mean Absolute Error
mae_sensex = mean_absolute_error(test_endog_sensex['Close_S'], predicted_values_sensex)
print(f"Mean Absolute Error of Sensex: {mae_sensex}")

y=sensex_data.iloc[train_size_sensex:]

### Visualize the predicted values along with the actual values
plt.figure(figsize=(10, 6))
plt.plot(y['Date'], y['Close_S'], label='Actual Sensex Close')
plt.plot(y['Date'], predicted_values_sensex, label='Predicted Sensex Close', linestyle='--')
plt.ylabel('Date')
plt.ylabel('Sensex Close')
plt.title('Actual vs Predicted Sensex Close')
plt.legend()
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import itertools

# Define the range of p, d, q values to search
p_values = range(5)
d_values = range(5)
q_values = range(5)

# Load data
nifty_data = pd.read_csv('nifty_data.csv')
sensex_data = pd.read_csv('sensex_data.csv')
usd_inr_data = pd.read_csv('usd_inr_data.csv')

# Merge dataframes
nifty_inr = usd_inr_data.merge(nifty_data, how='outer', on='Date')
sensex_inr = usd_inr_data.merge(sensex_data, how='outer', on='Date')

# Drop NaN values
nifty_inr.dropna(inplace=True)
sensex_inr.dropna(inplace=True)

# Replace ',' and convert to float for NIFTY DataFrame
nifty_inr['Close_N'] = nifty_inr['Close_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Open_N'] = nifty_inr['Open_N'].replace(',', '', regex=True).astype(float)
nifty_inr['Low_N'] = nifty_inr['Low_N'].replace(',', '', regex=True).astype(float)
nifty_inr['High_N'] = nifty_inr['High_N'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'])

# Set 'Date' as the index
nifty_inr.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size = int(len(nifty_inr) *0.15)
train_data = nifty_inr.iloc[train_size:]
test_data = nifty_inr.iloc[:train_size]

# Separate endogenous and exogenous variables for training and testing
train_endog = train_data[['Close_N']]
train_exog = train_data[['Open', 'High', 'Low', 'Close','Open_N', 'High_N', 'Low_N']]

test_endog = test_data[['Close_N']]
test_exog = test_data[['Open', 'High', 'Low', 'Close','Open_N', 'High_N', 'Low_N' ]]

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

# Grid search for the best ARIMA order for NIFTY
best_mae_nifty = float('inf')  # Initialize with a large value
best_order_nifty = None

# Try different values for p, d, q
endog_column_nifty = train_endog.iloc[:, 0]

model_nifty = ARIMA(endog_column_nifty, exog=train_exog, order=(0, 0, 4))
results_nifty = model_nifty.fit()

forecast_nifty = results_nifty.get_forecast(steps=len(test_endog), exog=test_exog)
predicted_values_nifty = forecast_nifty.predicted_mean
mae_nifty = mean_absolute_error(test_endog['Close_N'], predicted_values_nifty)             
best_order_nifty = (1,3,3)

print(f"Best ARIMA Order for NIFTY: {best_order_nifty}")
print(f"Mean Absolute Error of NIFTY: {mae_nifty}")

# Visualize NIFTY predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close_N'], label='Actual NIFTY Close')
plt.plot(test_data.index, predicted_values_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print information about sensex_inr DataFrame
print(sensex_inr.info())

# Replace ',' and convert to float for Sensex DataFrame
sensex_inr['Close_S'] = sensex_inr['Close_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Open_S'] = sensex_inr['Open_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Low_S'] = sensex_inr['Low_S'].replace(',', '', regex=True).astype(float)
sensex_inr['High_S'] = sensex_inr['High_S'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
sensex_inr['Date'] = pd.to_datetime(sensex_inr['Date'])

# Set 'Date' as the index
sensex_inr.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size_sensex = int(len(sensex_inr) * 0.15)
train_data_sensex = sensex_inr.iloc[train_size_sensex:]
test_data_sensex = sensex_inr.iloc[:train_size_sensex]

# Separate endogenous and exogenous variables for training and testing
train_endog_sensex = train_data_sensex[[ 'Close_S']]
train_exog_sensex = train_data_sensex[['Open', 'High', 'Low', 'Close','Open_S', 'High_S','Low_S']]

test_endog_sensex = test_data_sensex[[ 'Close_S']]
test_exog_sensex = test_data_sensex[['Open', 'High', 'Low', 'Close','Open_S', 'High_S','Low_S']]

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

# Grid search for the best ARIMA order for Sensex
best_mae_sensex = float('inf')  # Initialize with a large value

# Try different values for p, d, q

endog_column_sensex = train_endog_sensex.iloc[:, 0]
model_sensex = ARIMA(endog_column_sensex, exog=train_exog_sensex, order=(0, 0, 3))
results_sensex = model_sensex.fit()


forecast_sensex = results_sensex.get_forecast(steps=len(test_endog_sensex), exog=test_exog_sensex)
predicted_values_sensex = forecast_sensex.predicted_mean
mae_sensex = mean_absolute_error(test_endog_sensex['Close_S'], predicted_values_sensex)


best_order_sensex = (0, 0, 3)
print(f"Best ARIMA Order for Sensex: {best_order_sensex}")
print(f"Mean Absolute Error of Sensex: {mae_sensex}")

# Visualize Sensex predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(test_data_sensex.index, test_data_sensex['Close_S'], label='Actual Sensex Close')
plt.plot(test_data_sensex.index, predicted_values_sensex, label='Predicted Sensex Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sensex Close')
plt.title('Actual vs Predicted Sensex Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#################

"""

# Input date from the user
input_date = '2023-12-15'  # Replace this with user input logic

# Generating future dates from the last available date to the inputted date
last_available_date = nifty_inr.index[-1]  # Assuming the last available date in the dataset
future_dates = pd.date_range(start=last_available_date, end=input_date)

endog_data_nifty_p=

endog_column_nifty_p = train_endog.iloc[:, 0]

model_nifty_p = ARIMA(endog_column_nifty_p, exog=train_exog_nifty_p, order=(0, 0, 4))
results_nifty = model_nifty.fit()
forecast_nifty = results_nifty.get_forecast(steps=len(test_endog), exog=test_exog)
predicted_values_nifty = forecast_nifty.predicted_mean
mae_nifty = mean_absolute_error(test_endog['Close_N'], predicted_values_nifty)             
best_order_nifty = (1,3,3)

print(f"Best ARIMA Order for NIFTY: {best_order_nifty}")
print(f"Mean Absolute Error of NIFTY: {mae_nifty}")

# Visualize NIFTY predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close_N'], label='Actual NIFTY Close')
plt.plot(test_data.index, predicted_values_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

#, exog=exog_sensex.loc[future_dates]

# ...
endog_data_nifty_p = nifty_inr["Close_N"]
train_data_nifty_p_exo = nifty_inr[['Open', 'High', 'Low', 'Close','Open_N', 'High_N', 'Low_N']]
endog_col_nifty_p = nifty_inr.iloc[:, 0]

model_nifty_p = ARIMA(endog_col_nifty_p, exog=train_data_nifty_p_exo, order=(0, 0, 4))
result_nifty_p = model_nifty_p.fit()
forecast_nifty_p = result_nifty_p.get_forecast(steps=len(endog_data_nifty_p), exog=train_data_nifty_p_exo)

# Visualize Nifty predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(forecast_nifty_p.predicted_mean,forecast_nifty_p, label='Predicted Nifty Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Nifty Close')
plt.title('Predicted Nifty Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
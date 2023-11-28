import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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
train_size = int(len(nifty_inr) * 0.85)
train_data = nifty_inr.iloc[:train_size]
test_data = nifty_inr.iloc[train_size:]

# Separate endogenous and exogenous variables for training and testing
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

# Grid search for the best ARIMA order for NIFTY
best_mae_nifty = float('inf')  # Initialize with a large value

# Try different values for p, d, q
endog_column_nifty = train_endog.iloc[:, 0]
model_nifty = ARIMA(endog_column_nifty, exog=train_exog, order=(1, 3, 3))
results_nifty = model_nifty.fit()
forecast_nifty = results_nifty.get_forecast(steps=len(test_endog), exog=test_exog)
predicted_values_nifty = forecast_nifty.predicted_mean
mae_nifty = mean_absolute_error(test_endog['Close_N'], predicted_values_nifty)
best_order_nifty = (1, 3, 3)
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

# Train the NIFTY ARIMA model on the full dataset
endog_column_nifty_full = nifty_inr['Close_N']
model_nifty_full = ARIMA(endog_column_nifty_full, exog=nifty_inr[['Open', 'High', 'Low', 'Close']], order=best_order_nifty)
results_nifty_full = model_nifty_full.fit()

# Forecast future values for NIFTY
steps_to_forecast_nifty = 10  # Adjust the number of steps as needed

# Extend exogenous variables for forecasting
# Extend exogenous variables for forecasting
future_exog_nifty = pd.DataFrame(index=pd.date_range(test_data.index[-1], periods=steps_to_forecast_nifty + 1, freq='B')[1:])
future_exog_nifty[['Open', 'High', 'Low', 'Close']] = test_exog.iloc[-1].values[None, :]

# Forecast future values for NIFTY
forecast_nifty_future = results_nifty_full.get_forecast(steps=steps_to_forecast_nifty, exog=future_exog_nifty)
predicted_values_nifty_future = forecast_nifty_future.predicted_mean

# Visualize NIFTY predictions with future values
plt.figure(figsize=(10, 6))
plt.plot(nifty_inr.index, nifty_inr['Close_N'], label='Actual NIFTY Close')
plt.plot(test_data.index, predicted_values_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.plot(future_exog_nifty.index, predicted_values_nifty_future, label='Future Predictions', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close with Future Predictions')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


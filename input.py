import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
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

# Replace ',' and convert to float for SENSEX DataFrame
sensex_inr['Close_S'] = sensex_inr['Close_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Open_S'] = sensex_inr['Open_S'].replace(',', '', regex=True).astype(float)
sensex_inr['Low_S'] = sensex_inr['Low_S'].replace(',', '', regex=True).astype(float)
sensex_inr['High_S'] = sensex_inr['High_S'].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'])
sensex_inr['Date'] = pd.to_datetime(sensex_inr['Date'])

# Set 'Date' as the index
nifty_inr.set_index('Date', inplace=True)
sensex_inr.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size = int(len(nifty_inr) * 0.15)
train_data_nifty = nifty_inr.iloc[train_size:]
test_data_nifty = nifty_inr.iloc[:train_size]
print(train_data_nifty)
print(test_data_nifty)

train_size_sensex = int(len(sensex_inr) * 0.85)
train_data_sensex = sensex_inr.iloc[:train_size_sensex]
test_data_sensex = sensex_inr.iloc[train_size_sensex:]

# Separate endogenous and exogenous variables for training and testing for NIFTY
train_endog_nifty = train_data_nifty[['Open_N', 'High_N', 'Low_N', 'Close_N']]
test_endog_nifty = test_data_nifty[['Open_N', 'High_N', 'Low_N', 'Close_N']]

# Separate endogenous and exogenous variables for training and testing for SENSEX
train_endog_sensex = train_data_sensex[['Open_S', 'High_S', 'Low_S', 'Close_S']]
test_endog_sensex = test_data_sensex[['Open_S', 'High_S', 'Low_S', 'Close_S']]

# Fitting VAR model for NIFTY
model_nifty = VAR(train_endog_nifty)

# Fitting VAR model for SENSEX
model_sensex = VAR(train_endog_sensex)

# Fitting VAR model for NIFTY
best_order_nifty = model_nifty.select_order()
lag_order_nifty = best_order_nifty.selected_orders['aic']

results_nifty = model_nifty.fit(lag_order_nifty)

# Fitting VAR model for SENSEX
best_order_sensex = model_sensex.select_order()
lag_order_sensex = best_order_sensex.selected_orders['aic']

results_sensex = model_sensex.fit(lag_order_sensex)

# Forecasting using VAR model for NIFTY
predicted_values_nifty = results_nifty.forecast(train_endog_nifty.values, steps=len(test_endog_nifty))
predicted_close_nifty = predicted_values_nifty[:, 3]  # Close_N column prediction

# Forecasting using VAR model for SENSEX
predicted_values_sensex = results_sensex.forecast(train_endog_sensex.values, steps=len(test_endog_sensex))
predicted_close_sensex = predicted_values_sensex[:, 3]  # Close_S column prediction

# Creating date range for test_data_nifty and test_data_sensex
date_range_nifty = pd.date_range(start=test_data_nifty.index[0], periods=len(test_endog_nifty), freq='B')
date_range_sensex = pd.date_range(start=test_data_sensex.index[0], periods=len(test_endog_sensex), freq='B')

# Visualizing NIFTY predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(date_range_nifty, test_data_nifty['Close_N'], label='Actual NIFTY Close')
plt.plot(date_range_nifty, predicted_close_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualizing SENSEX predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(date_range_sensex, test_data_sensex['Close_S'], label='Actual SENSEX Close')
plt.plot(date_range_sensex, predicted_close_sensex, label='Predicted SENSEX Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('SENSEX Close')
plt.title('Actual vs Predicted SENSEX Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

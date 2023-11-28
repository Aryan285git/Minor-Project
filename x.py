import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Specify a small regularization parameter (alpha)
alpha = 1e-6


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
nifty_inr[['Close_N', 'Open_N', 'Low_N', 'High_N']] = nifty_inr[['Close_N', 'Open_N', 'Low_N', 'High_N']].replace(',', '', regex=True).astype(float)

# Convert 'Date' to datetime
nifty_inr['Date'] = pd.to_datetime(nifty_inr['Date'], format='%m/%d/%y')

# Set 'Date' as the index
nifty_inr.set_index('Date', inplace=True)

# Split data into training and testing sets
train_size = int(len(nifty_inr) * 0.85)
train_data = nifty_inr.iloc[:train_size]
test_data = nifty_inr.iloc[train_size:]

# Separate endogenous and exogenous variables for training and testing in VARIMA for NIFTY
train_endog_varima_nifty = train_data[['Close_N', 'Open_N', 'High_N', 'Low_N']]
train_exog_varima_nifty = train_data[['Open_N', 'High_N', 'Low_N', 'Close_N']]

test_endog_varima_nifty = test_data[['Close_N', 'Open_N', 'High_N', 'Low_N']]
test_exog_varima_nifty = test_data[['Open_N', 'High_N', 'Low_N', 'Close_N']]

# Convert columns to numeric format
train_endog_varima_nifty = train_endog_varima_nifty.apply(pd.to_numeric, errors='coerce')
train_exog_varima_nifty = train_exog_varima_nifty.apply(pd.to_numeric, errors='coerce')
test_endog_varima_nifty = test_endog_varima_nifty.apply(pd.to_numeric, errors='coerce')
test_exog_varima_nifty = test_exog_varima_nifty.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values after conversion
train_endog_varima_nifty.dropna(inplace=True)
train_exog_varima_nifty.dropna(inplace=True)
test_endog_varima_nifty.dropna(inplace=True)
test_exog_varima_nifty.dropna(inplace=True)

# Grid search for the best VARIMA order for NIFTY
best_mae_varima_nifty = float('inf')  # Initialize with a large value
model_varima_nifty = VARMAX(train_endog_varima_nifty, exog=train_exog_varima_nifty, order=(1, 1), trend='c', enforce_stationarity=False, enforce_invertibility=False)
results_varima_nifty = model_varima_nifty.fit(disp=False, start_params=None, method='lbfgs', alpha=alpha)

# Try different values for p, d, q, and P, D, Q (order and seasonal order)
forecast_varima_nifty = results_varima_nifty.get_forecast(steps=len(test_endog_varima_nifty), exog=test_exog_varima_nifty)
predicted_values_varima_nifty = forecast_varima_nifty.predicted_mean
mae_varima_nifty = mean_absolute_error(test_endog_varima_nifty['Close_N'], predicted_values_varima_nifty)
best_order_varima_nifty = (1, 1)

print(f"Best VARIMA Order for NIFTY: {best_order_varima_nifty}")
print(f"Mean Absolute Error of VARIMA for NIFTY: {mae_varima_nifty}")

# Visualize VARIMA predictions with the best order for NIFTY
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['Close_N'], label='Actual NIFTY Close')
plt.plot(test_data.index, predicted_values_varima_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Actual vs Predicted NIFTY Close using VARIMA')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

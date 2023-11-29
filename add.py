
# Input date from the user
input_date = '2023-12-15'  # Replace this with user input logic

# Generating future dates from the last available date to the inputted date
last_available_date = nifty_inr.index[-1]  # Assuming the last available date in the dataset
future_dates = pd.date_range(start=last_available_date, end=input_date)

# Forecasting future values for NIFTY
endog_column_nifty = nifty_inr['Close_N']
exog_nifty = nifty_inr[['Open', 'High', 'Low', 'Close','Open_N', 'High_N', 'Low_N']]

model_nifty = ARIMA(endog_column_nifty, order=(1, 3, 3))
results_nifty = model_nifty.fit()

nn=exog_nifty

# Forecast future values for NIFTY
forecast_nifty = results_nifty.get_forecast(steps=len(future_dates), exog=nn)
predicted_values_nifty = forecast_nifty.predicted_mean

# Plotting the predicted NIFTY values for future dates
plt.figure(figsize=(10, 6))
plt.plot(future_dates, predicted_values_nifty, label='Predicted NIFTY Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('NIFTY Close')
plt.title('Predicted NIFTY Close from {} to {}'.format(last_available_date, input_date))
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ...
endog_data_nifty_p = nifty_inr["Close_N"]
train_data_nifty_p_exo = nifty_inr[['Open', 'High', 'Low', 'Close','Open_N', 'High_N', 'Low_N']]
endog_col_nifty_p = nifty_inr.iloc[:, 0]

model_nifty_p = ARIMA(endog_col_nifty_p, exog=train_data_nifty_p_exo, order=(0, 0, 4))
result_nifty_p = model_nifty_p.fit()
forecast_nifty_p = result_nifty_p.get_forecast(steps=len(endog_data_nifty_p), exog=train_data_nifty_p_exo)

# Visualize Nifty predictions with the best order
plt.figure(figsize=(10, 6))
plt.plot(forecast_nifty_p.predicted_mean, label='Predicted Nifty Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Nifty Close')
plt.title('Predicted Nifty Close')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#, exog=exog_sensex.loc[future_dates]
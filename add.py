
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


# Forecasting future values for Sensex
endog_column_sensex = sensex_inr['Close_S']
exog_sensex = sensex_inr[['Open', 'High', 'Low', 'Close','Open_S', 'High_S','Low_S']]

model_sensex = ARIMA(endog_column_sensex, order=(0, 0, 3))
results_sensex = model_sensex.fit()

# Forecast future values for Sensex
forecast_sensex = results_sensex.get_forecast(steps=len(future_dates))
predicted_values_sensex = forecast_sensex.predicted_mean

# Plotting the predicted Sensex values for future dates
plt.figure(figsize=(10, 6))
plt.plot(future_dates, predicted_values_sensex, label='Predicted Sensex Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sensex Close')
plt.title('Predicted Sensex Close from {} to {}'.format(last_available_date, input_date))
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#, exog=exog_sensex.loc[future_dates]
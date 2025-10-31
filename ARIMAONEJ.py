
def moving_average(N):
    CN = [0] * len(N)
    for i in range(3, len(N)):
        CN[i] = (CN[i - 3] + N[i - 2] + N[i - 1] + N[i]) / 3
    return CN

# Autoregressive model
def autoregressive_model(N, n_forecast):
    a_value = (N[-1] - N[-2]) / N[-2]  # Autoregressive coefficient (change in last value)
    forecast = [N[-1] + a_value * (i + 1) for i in range(n_forecast)]
    return forecast

# Compute differences of a series
def difference(data, interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff

# Integrate differences into the original series
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# ARIMA model
def arima_model(N, n_forecast, p, d, q):
    # Difference data if necessary
    if d > 0:
        N = difference(N, d)
    
    # Fit autoregressive model
    residuals = autoregressive_model(N, len(N) + n_forecast - 1)
    
    # Integrate back the differences
    if d > 0:
        history = N[len(N) - d:]
        for i in range(n_forecast):
            yhat = residuals[i] + history[-1]
            history.append(yhat)
            yhat = inverse_difference(history, yhat, d)
            residuals[i] = yhat
    
    return residuals[-n_forecast:]

# Example data
N = [45, 100, 60, 243, 17, 66]

# Compute CN values using moving average
CN = moving_average(N)

# Print CN values
print("Time\tN\tCN\t% Error/Accuracy")
for i in range(len(N)):
    percentage = 0 if i < 3 else (CN[i] / N[i - 1]) * 100
    print(f"{i}\t{N[i]}\t{CN[i]:.2f}\t{percentage:.2f}%")

# ARIMA model parameters
p = 1  # Autoregressive order
d = 0  # Differencing order
q = 0  # Moving average order
n_forecast = 10  # Number of forecasted values

# Forecast using ARIMA model
forecast_y = arima_model(N, n_forecast, p, d, q)

# Print forecasted values
print("\nForecasted Values:")
for i, forecast in enumerate(forecast_y):
    print(f"Forecast for next day {i+1}: {forecast:.2f}")

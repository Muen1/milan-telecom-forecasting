import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def sarima_forecast(train, test, order=(1,1,1)):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    start = len(train)
    end = len(train) + len(test) - 1
    pred = fitted.predict(start=start, end=end, dynamic=False)
    return np.asarray(pred), fitted

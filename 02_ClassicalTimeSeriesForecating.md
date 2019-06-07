



<span style = "font-family: Calibri; font-size:4em;"> Classical Time Series Forecasting</span>

<span style = "font-family: Calibri; font-size:16;">The Statistical Methods covered are:</span>

1) Auto Regression Moving Average (ARMA)





All the code assumes that a pandas dataframe **furniture_sales_weekly['Sales']** has been created as per the home page.

## 1. Auto Regression Moving Average (ARMA)

Using the functions from statsmodels is straightforward:

```python
from statsmodels.tsa.arima_model import ARMA
model = ARMA(furniture_sales_weekly['Sales'], order = (0,1))
model_fit = model.fit(disp=False)
y_pred = model_fit.predict(start = '2017-03-19')
```

The start date prediction has been set to match the Machine Learning split so that like-for-like comparisons can be made. This produced the following output:

![ARMA Sales Forecast](images\ARMAResult.png)



## 2. Auto Regressive Integrated Moving Averages (ARIMA)

```python
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(furniture_sales_weekly['Sales'], order = (1,1,1))
model_fit = model.fit(disp=False)
y_pred = model_fit.predict(start = '2017-03-19')
```

This produces the following results:

![ARIMA Sales Forecast](images\ARIMAResult.png)

The results are much worse than the ARMA forecast but this does not take the seasonality into account.



## 3. Seasonal Auto Regressive Integrated Moving Averages with eXogenous regressors

The code below determines the best order and seasonal order combination (in a similar manner to grid search). The measure is the AIC (Akaike Information Criteria)

```python
lowest_aic = 999999999
strlowest = ''

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(furniture_sales_weekly,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            
            if lowest_aic >  results.aic:
                strlowest = 'SARIMA{}x{}52 - AIC:{}'.format(param, param_seasonal, results.aic)
                lowest_aic = results.aic
        except:
            continue

print (strlowest)
```

This gives the following as the best model:

```python
SARIMA(1, 1, 1)x(1, 1, 0, 52)52 - AIC:1889.7557324050163
```

```python
mod = sm.tsa.statespace.SARIMAX(furniture_sales_weekly,
                                order=(0,1,1),
                                seasonal_order=(1,1,0,52),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
```

The best model is plugged into the SARIMAX function.

```python
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.L1         -0.9843      0.172     -5.736      0.000      -1.321      -0.648
ar.S.L52      -0.4989      0.142     -3.517      0.000      -0.777      -0.221
sigma2      9.498e+06   2.03e+06      4.677      0.000    5.52e+06    1.35e+07
==============================================================================
```



Visually these shows an improvement to the weekly sales forecasts and the error is an improvement on ARMA forecasting:

![SARIMAX Forecast](images\SARIMAX.png)

SARIMAX has a useful diagnostic print:

```python
results.plot_diagnostics(figsize=(16,8))
```

![SARIMAX diagnostics](images\SARIMAdiagnostics.png)
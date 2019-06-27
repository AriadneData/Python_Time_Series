# DeepLearning Time Series Forecasting

[Code for this page](https://nbviewer.jupyter.org/github/AriadneData/Python_Time_Series/blob/master/TimeSeries_Keras.ipynb)

The Deep Learning  networks use Keras:

1. MLP

2. Vanilla LSTM



All methods assume the dataframe has been created as in the ReadMe

### Further dataframe manipulation

___

As in the previous section for Machine Learning the Pandas **shift()**  function is used to create previous timesteps as features for the row.

MinMax Scalar was used to standardise the data set.

```python
from sklearn.preprocessing import MinMaxScaler

def MinMax(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df))
    df_scaled.rename(columns = {0:'Sales'}, inplace = True)
    return df_scaled, scaler
```
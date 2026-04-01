# Time Series Forecasting for Retail Sales with MLP, ARIMA, KAN, and Chronos Retail forecasting is high-stakes. Misses mean stockouts, markdowns, or
investor panic. ARIMA is the legacy tool: fast, interpretable...

### Time Series Forecasting for Retail Sales with MLP, ARIMA, KAN, and Chronos
Retail forecasting is high-stakes. Misses mean stockouts, markdowns, or
investor panic. ARIMA is the legacy tool: fast, interpretable,
consistent. Newer models like LSTMs and KANs claim to capture
nonlinearity. Transformers promise everything.

We tested all of them using U.S. retail sales and updated the results
through April 2025. The outcomes reflect real-world constraints: fixed
forecast window, same input features, same evaluation horizon.

We pulled seasonally adjusted U.S. Retail and Food Services Sales from
the FRED API:


Data is in millions of U.S. dollars. We normalize it for neural networks
but leave it raw for ARIMA. Our prediction window is 12 months. All
models start their forecast at the same point: April 2024.

We use a lag window of 24 months. That's long enough to capture seasonal
trends, short enough to avoid structural breaks.

### Model Summary
**ARIMA(5,1,0)** uses autoregressive lags and first differencing. It's
robust for linear dynamics.

**MLP** is a single hidden layer with 64 neurons and dropout. No fancy
layers. No attention.


**KAN** uses spline-based layers, which approximate continuous
functions. It's based on the Kolmogorov-Arnold representation theorem.
Ours had 3 grid segments and order-2 polynomials. Training took \~87
seconds.

**LSTM** uses a single layer with 32 hidden units. Forecasts are
generated from the final hidden state.

**Chronos T5** is AWS's pretrained Transformer for time series, loaded
from Hugging Face and used in inference mode.

### Evaluation Metrics
- **RMSE**: Root mean square error in millions USD
- **MAE**: Mean absolute error in millions USD
- **MAPE**: Mean absolute percentage error
- **Training Time**: Seconds

All models predict the same 12 months (Apr 2024--Mar 2025). We
inverse-transform predictions to compare in real units.


### Visual Comparison
We plotted the last 24 months of sales. Each model's forecast starts at
April 2024. A vertical line marks the forecast boundary.


- LSTM nailed it.
- MLP was close.
- KAN drifted.
- ARIMA flattened out.
- Chronos T5 trailed MLP but was better than ARIMA.

### What This Shows
Neural networks adapt. Even simple ones. With a basic feature window, a
small LSTM beat everything else.

KANs are expressive but slow. Without regularization and
validation-based stopping, they're hard to trust in production.

Chronos is viable for inference pipelines where latency matters and
retraining isn't an option.

ARIMA remains good for sanity checks, backtesting, and quick
benchmarks --- but it doesn't scale to today's volatility.


This same structure can be applied to other data series. I looked at
[personal savings
rates](https://fred.stlouisfed.org/series/PSAVERT) and it is slightly different. The trend
still holds that KAN is computationally expensive.

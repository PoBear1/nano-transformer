# Cross-sectional predictions of correlated time series with latent variables
This is an attempt (!) to combine a hidden state space model (SSM) with external inputs that include global variables affecting all time series to some extent, for example in the case of equities return it could be interest rates or new geopolitical information encoded using an LLM with a second model decoding the SSM state along with historical data to generate new distributions for time series forecasting. We aim to use a variational approach for the second model by estimating distribution parameters. 

This might fail, this might work, this might exceed benchmarks, who knows?

We're going to set this up as part of a point prediction model for a portfolio of equities. 
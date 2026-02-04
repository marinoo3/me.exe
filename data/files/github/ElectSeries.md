# ElecSeries

Weighted nearest neighbours for time series forecasting

R6 class implementing a weighted nearest neighbours (WNN) algorithm for univariate time series forecasting. The model uses the last **p** observations as a window, searches for the **K** most similar historical windows, and predicts future values as a distance-weighted combination of the neighbours' subsequent observations.

## Instalation
```bash
# install remotes if not already installed
install.packages("remotes")
# install directly from GitHub
remotes::install_github("rsquaredata/mmrClustVar")
# import library
library(mmrClustVar)
```

## Usage

Import package
```R
library(Elecseries)
```

Fit / predict
```R
model <- WNN$new(K = 3, p = 12)
model$fit(data.ts)
pred.ts <- model$predict(h = 96)
```

## TODO: credit article for part of code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.eval_measures import rmse, aic


class VectorAutoRegressor:
    def __init__(self, housing_index, sentiment_index, housing_column, date_format=None):
        # Import housing & sentiment data

        self.housing_column = housing_column
        self.housing_data = pd.read_csv(housing_index, parse_dates=['Period'], index_col='Period',
                                        usecols=['Period', self.housing_column], thousands=',')
        if date_format is not None:
            self.housing_data.index = pd.to_datetime(self.housing_data.index, format=date_format)
        self.housing_data.index = self.housing_data.index.to_period("M")
        self.housing_data = self.housing_data.sort_index()
        self.housing_data[housing_column] = self.housing_data[housing_column].astype(float)

        self.sentiment_data = sentiment_index

        # Merge and process data. But only merge data if sentiment index will be used
        if self.sentiment_data is None:
            self.data = self.housing_data
        else:
            self.data = pd.merge(self.housing_data, self.sentiment_data, left_index=True, right_index=True)

        self.data = self.data.dropna()
        self.n_differenced = 0  # n_differenced keeps track of how many times the series got differenced

        self.data_stationary = self.make_stationary(self.data)
        #self.data_stationary = self.data

        # Split data
        n_test = round(len(self.data) * 0.2)
        self.training_data, self.test_data = self.data_stationary[0:-n_test], self.data_stationary[-n_test:]

        # Construct model
        self.model = VAR(self.training_data)
        self.fit = self.model.fit(maxlags=4, ic='aic')

    def grangers_causation_matrix(self, test='ssr_chi2test', verbose=False, maxlag=12):
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table
        are the P-Values. P-Values lesser than the significance level (0.05), implies
        the Null Hypothesis that the coefficients of the corresponding past values is
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        data = self.data
        variables = list(self.data.columns)

        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
                if verbose: print('Y = {}, X = {}, P Values = {}'.format(r, c, p_values))
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        return df

    def cointegration_test(self, alpha=0.05):
        """Perform Johanson's Cointegration Test and Report Summary"""
        df = self.data
        out = coint_johansen(df, -1, 5)
        d = {'0.90': 0, '0.95': 1, '0.99': 2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1 - alpha)]]

        def adjust(val, length=6): return str(val).ljust(length)

        # Summary
        print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
        for col, trace, cvt in zip(df.columns, traces, cvts):
            print(adjust(col), ':: ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)

    def test_lags(self, lags=range(1, 10)):
        for lag in lags:
            result = self.model.fit(lag)
            print('Lag Oder = ', lag)
            print('AIC : ', result.aic)
            print('BIC : ', result.bic)
            print('FPE : ', result.fpe)
            print('HQIC: ', result.hqic, '\n')

    def make_stationary(self, data):
        for column in data.columns:
            stationary = adfuller_test(data[column], name=column)
            if not stationary:
                differenced_data = data.diff().dropna()
                self.n_differenced += 1
                return self.make_stationary(differenced_data)
        return data

    def fit_model(self):
        """Fits the model to to a specified lag order"""
        return self.fit

    def forecast_period(self, period, plot=True, ):
        """Forecasts a period using the trained VAR model
        Period: Timeperiod to be forecasted  (Pandas PeriodIndex)
        """

        actual_column, forecast_column = self.housing_column + '_actual', self.housing_column + '_forecast'

        forecast = pd.DataFrame(index=period, columns=[actual_column, forecast_column])
        forecast[actual_column] = self.data_stationary[self.housing_column][period]
        fit = self.fit
        lag_order = fit.k_ar

        for month in period:
            forecast_input = self.data_stationary[:month][-(lag_order + 1):-1].values
            forecasted_values = fit.forecast(y=forecast_input, steps=1)
            forecasted_housing_value = forecasted_values[0][0]
            forecast.at[month, forecast_column] = forecasted_housing_value

        if plot:
            plot_forecast(forecast)

        return forecast


def plot_forecast(forecast):
    """Plots the actual values and forecasted values of the housing data in one graph"""
    ax = plt.gca()
    for column in forecast.columns:
        if column.endswith("_forecast"):
            color = 'red'
        else:
            color = 'gray'
        forecast.plot(kind='line', y=column, color=color, ax=ax)
    plt.show()


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series, print report and return if series is stationary"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    if verbose:
        print('    Augmented Dickey-Fuller Test on "{}"'.format(name), "\n   ", '-' * 47)
        print(' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(' Significance Level    = {}'.format(signif))
        print(' Test Statistic        = {}'.format(output["test_statistic"]))
        print(' No. Lags Chosen       = {}'.format(output["n_lags"]))

        for key, val in r[4].items():
            print(' Critical value {} = {}'.format(adjust(key), round(val, 3)))

    if p_value <= signif:
        if verbose:
            print(" => P-Value = {}. Rejecting Null Hypothesis.".format(p_value))
            print(" => Series is Stationary.")
        return True
    else:
        if verbose:
            print(" => P-Value = {}. Weak evidence to reject the Null Hypothesis.".format(p_value))
            print(" => Series is Non-Stationary.")
        return False


def invert_transformation(self, df_forecast):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = self.data.columns
    for col in columns:
        # Roll back 1st Diff
        if self.n_differenced != 1:
            print("ALERT: REVERTING 1 DIFFERENCE, BUT SERIES WAS DIFFERENCED MORE OR LESS THAN ONCE")
        df_fc[str(col) + '_forecast_normal'] = self.data[col].iloc[-1] + df_fc[str(col) + '_forecast'].cumsum()
    return df_fc


def run_var_test(housing_path, sentiment_index, housing_column, lag_order, test_lags=False):
    """Helper function to use in Jupyter notebooks to quickly run a VAR on a housing index and a sentiment index"""
    var = VectorAutoRegressor(housing_index=housing_path, sentiment_index=sentiment_index,
                              housing_column=housing_column)
    if test_lags:
        var.test_lags()

    var.fit_model(lag_order=lag_order)

    # Forecast of whole dataset
    var.forecast_period(var.data.index[lag_order:])

    # Forecast of test set
    var.forecast_period(var.test_data.index)

    var.fit.summary()

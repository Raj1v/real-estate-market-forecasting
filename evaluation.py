from vector_auto_regression import VectorAutoRegressor
from models.model1 import Model1
import articles as art
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from latex_tables import table_printer


class Evaluator:
    """
    This class gives:
        - Forecast evaluation
        - VAR results
            - Coefficients
            - Fit
    """

    def __init__(self, model, articles, housing_index, market_data_column="House price index", sentiment=None,
                 date_format=None):
        self.housing_index = housing_index
        self.market_data_column = market_data_column
        self.date_format = date_format

        if sentiment is None:
            years = [article.year for article in articles]
            self.sentiment = model.timespan_sentiment(min(years), max(years))
        else:
            self.sentiment = sentiment

        self.print_evaluation()

    def print_evaluation(self):
        sentiment = self.sentiment
        negative_sentiment = sentiment[["Negative Scores"]]
        positive_sentiment = sentiment[["Positive Scores"]]
        print("§" * 100)
        print("POSITIVE SENTIMENT")
        self.positive_var, self.positive_forecast = self.var_results(positive_sentiment)
        self.calculate_mape(self.positive_forecast)
        self.up_down_prediction(self.positive_forecast)
        print("§" * 100)
        print("§" * 100)
        print("NEGATIVE SENTIMENT")
        self.negative_var, self.negative_forecast = self.var_results(negative_sentiment)
        self.calculate_mape(self.negative_forecast)
        self.up_down_prediction(self.negative_forecast)
        print("§" * 100)

    def var_results(self, sentiment_index):
        var = VectorAutoRegressor(housing_index=self.housing_index, sentiment_index=sentiment_index,
                                  housing_column=self.market_data_column, date_format=self.date_format)

        forecast = var.forecast_period(var.test_data.index)
        print(var.fit.summary())
        self.print_confidence_intervals(var)
        return var, forecast

    def print_confidence_intervals(self, var):
        results = var.fit
        n = results.nobs
        print("Confidence intervals:")
        for label, coefficient in results.params[self.market_data_column].items():
            se = results.stderr[self.market_data_column][label]
            h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
            print(label, "interval: [ ", coefficient - h, ",", coefficient + h, " ]")

    def calculate_mape(self, forecast):
        predictions = list(forecast[self.market_data_column + '_forecast'])
        actuals = list(forecast[self.market_data_column + '_actual'])
        mape = 0
        for i in range(len(predictions)):
            actual = actuals[i]
            predicted = predictions[i]
            mape += abs((actual - predicted) / actual)

        mape = mape / len(predictions)
        print('MAPE: %.3f' % mape)
        return mape

    def up_down_prediction(self, forecast):
        predictions = np.sign(list(forecast[self.market_data_column + '_forecast']))
        actuals = np.sign(list(forecast[self.market_data_column + '_actual']))
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions)
        confusion = confusion_matrix(actuals, predictions)
        print("Accuracy:", accuracy)
        print("Precision", precision)
        print("Confusion matrix", confusion)

    def print_latex_table(self, sentiment, housing_label, sentiment_label):
        if sentiment == "positive":
            var = self.positive_var
            sentiment_index = "Positive Scores"
        elif sentiment == "negative":
            var = self.negative_var
            sentiment_index = "Negative Scores"
        else:
            raise ValueError("Seniment parameter should be exactly equal to either 'positive' or 'negative'")

        printer = table_printer(var.fit, self.market_data_column, sentiment_index, housing_label, sentiment_label)
        printer.print_table()

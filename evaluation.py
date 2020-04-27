from vector_auto_regression import VectorAutoRegressor
from model1 import Model1
import articles as art


class Evaluator:
    """
    This class gives:
        - Forecast evaluation
        - VAR results
            - Coefficients
            - Fit
    """

    def __init__(self, model, articles, housing_index):
        self.housing_index = housing_index

        years = [article.year for article in articles]

        self.sentiment = model.timespan_sentiment(min(years), max(years))

        self.print_evaluation()

    def print_evaluation(self):
        sentiment = self.sentiment
        negative_sentiment = sentiment[["Negative Scores"]]
        positive_sentiment = sentiment[["Positive Scores"]]
        print("§" * 100)
        print("POSITIVE SENTIMENT")
        self.var_results(positive_sentiment)
        print("§" * 100)
        print("§" * 100)
        print("NEGATIVE SENTIMENT")
        self.var_results(negative_sentiment)
        print("§" * 100)

    def var_results(self, sentiment_index):
        var = VectorAutoRegressor(housing_index=self.housing_index, sentiment_index=sentiment_index,
                                  housing_column="House price index")

        var.forecast_period(var.test_data.index)
        print(var.fit.summary())

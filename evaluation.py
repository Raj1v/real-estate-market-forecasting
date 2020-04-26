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

        sentiment = model.timespan_sentiment(min(years), max(years))

        negative_sentiment = sentiment[["Negative Scores"]]
        positive_sentiment = sentiment[["Positive Scores"]]

        self.var_results(positive_sentiment)
        self.var_results(negative_sentiment)

    def var_results(self, sentiment_index):
        var = VectorAutoRegressor(housing_index=self.housing_index, sentiment_index=sentiment_index,
                                  housing_column="House price index")
        print(var.fit.summary())

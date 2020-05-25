from abc import ABC, abstractmethod
import pandas as pd
import articles as art


class ModelBase(ABC):
    def __init__(self, articles):
        self.articles = articles
        pass

    def timespan_sentiment(self, start_year, end_year):
        """Returns the sentiment scores of a specific timespan within a set of articles"""
        articles = self.articles
        positive_scores, negative_scores = [], []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                month_articles = art.months_articles(articles, month, year)
                month_sentiment_pos, month_sentiment_neg = self.articles_sentiment(month_articles)

                positive_scores.append(month_sentiment_pos)
                negative_scores.append(month_sentiment_neg)

        dates = pd.period_range(start="%d-1" % start_year, end="%d-12" % end_year, freq='M')
        sentiment_scores = list(zip(positive_scores, negative_scores))
        dataframe = pd.DataFrame(sentiment_scores, index=dates, columns=['Positive Scores', 'Negative Scores'])

        return dataframe

    def articles_sentiment(self, articles):
        """Returns the positive and negative sentiment scores for a set of articles"""

        if len(articles) == 0:
            return 0, 0

        positivity, negativity, neutrality = 0, 0, 0
        for article in articles:
            article_positivity, article_negativity, article_neutrality = self.article_sentiment(article)
            positivity += article_positivity
            negativity += article_negativity
            neutrality += article_neutrality

        positivity_indicator = positivity / len(articles)
        negativity_indicator = negativity / len(articles)

        return positivity_indicator, negativity_indicator

    @abstractmethod
    def article_sentiment(self, article):
        pass

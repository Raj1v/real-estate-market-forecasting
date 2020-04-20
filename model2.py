import pandas as pd
import articles as art


def classify_article(article, classifier, vectorizer):
    review_vector = vectorizer.transform([article.body])
    classification = classifier.predict(review_vector)[0]
    return classification


def articles_sentiment(articles, classifier, vectorizer):
    if len(articles) == 0:
        return 0, 0

    positive_articles, negative_articles, neutral_articles = 0, 0, 0
    for article in articles:
        classifcation = classify_article(article, classifier, vectorizer)
        if classifcation == '1':
            positive_articles += 1
        elif classifcation == '-1':
            negative_articles += 1
        else:
            neutral_articles += 1

    positivity_indicator = positive_articles / len(articles)
    negativity_indicator = negative_articles / len(articles)

    return positivity_indicator, negativity_indicator


def timespan_sentiment(articles, start_year, end_year, classifier, vectorizer):
    """Returns the sentiment scores of a specific timespan within a set of articles"""
    positive_scores, negative_scores = [], []

    for year in range(start_year, end_year + 1):
        print(year)
        for month in range(1, 13):
            month_articles = art.months_articles(articles, month, year)
            month_sentiment_pos, month_sentiment_neg = articles_sentiment(month_articles, classifier, vectorizer)

            positive_scores.append(month_sentiment_pos)
            negative_scores.append(month_sentiment_neg)

    dates = pd.period_range(start="%d-1" % start_year, end="%d-12" % end_year, freq='M')
    sentiment_scores = list(zip(positive_scores, negative_scores))
    dataframe = pd.DataFrame(sentiment_scores, index=dates, columns=['Positive Scores', 'Negative Scores'])

    return dataframe

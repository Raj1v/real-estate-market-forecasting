import csv
import articles as art

LM_NEGATIVE_WORDS, LM_POSITIVE_WORDS = [], []

with open('data/lm-sentiment-wordlist/lm_negative.csv') as f:
    reader = csv.reader(f)
    next(reader)
    LM_NEGATIVE_WORDS = [word[0] for word in list(reader)]

with open('data/lm-sentiment-wordlist/lm_positive.csv') as f:
    reader = csv.reader(f)
    next(reader)
    LM_POSITIVE_WORDS = [word[0] for word in list(reader)]


def tokens_sentiment(tokens):
    """Returns the positive and negative sentiment scores of a list of tokens as a tuple"""
    positive_tokens, negative_tokens = 0, 0

    for token in tokens:
        token = token.upper()
        if token in LM_POSITIVE_WORDS:
            positive_tokens += 1
        elif token in LM_NEGATIVE_WORDS:
            negative_tokens += 1

    positivity = positive_tokens / len(tokens)
    negativity = negative_tokens / len(tokens)
    return positivity, negativity


def articles_sentiment(articles):
    """Returns the positive and negative sentiment scores for a set of articles"""

    if len(articles) == 0:
        return 0, 0

    positivity, negativity = 0, 0
    for article in articles:
        headline_positivity, headline_negativity = tokens_sentiment(article.body_tokenized())
        positivity += headline_positivity
        negativity += headline_negativity

    positivity = positivity / len(articles)
    negativity = negativity / len(articles)

    return positivity, negativity


def timespan_sentiment(articles, start_year, end_year):
    """Returns the sentiment scores of a specific timespan within a set of articles"""
    positive_scores, negative_scores = [], []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_articles = art.months_articles(articles, month, year)
            month_sentiment_pos, month_sentiment_neg = articles_sentiment(month_articles)

            positive_scores.append(month_sentiment_pos)
            negative_scores.append(month_sentiment_neg)

    return positive_scores, negative_scores

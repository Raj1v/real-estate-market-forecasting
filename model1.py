import csv
import articles as art
import pandas as pd
from model import Model

LM_NEGATIVE_WORDS, LM_POSITIVE_WORDS = [], []

with open('data/lm-sentiment-wordlist/lm_negative.csv') as f:
    reader = csv.reader(f)
    next(reader)
    LM_NEGATIVE_WORDS = [word[0] for word in list(reader)]

with open('data/lm-sentiment-wordlist/lm_positive.csv') as f:
    reader = csv.reader(f)
    next(reader)
    LM_POSITIVE_WORDS = [word[0] for word in list(reader)]


class Model1(Model):
    def article_sentiment(self, article):
        """Returns the positive and negative sentiment scores of a list of tokens as a tuple"""
        tokens = article.body_tokenized()
        positive_tokens, negative_tokens, neutral_tokens = 0, 0, 0

        for token in tokens:
            token = token.upper()
            if token in LM_POSITIVE_WORDS:
                positive_tokens += 1
            elif token in LM_NEGATIVE_WORDS:
                negative_tokens += 1
            else:
                neutral_tokens += 1

        positivity = positive_tokens / len(tokens)
        negativity = negative_tokens / len(tokens)
        neutrality = neutral_tokens / len(tokens)
        return positivity, negativity, neutrality

import csv
import data_importer
from models.model_base import ModelBase

LM_POSITIVE_WORDS, LM_NEGATIVE_WORDS = data_importer.import_lm_dictionary()


class Model(ModelBase):
    def article_sentiment(self, article):
        """Returns the positive and negative sentiment scores of a list of tokens as a tuple"""
        tokens = article.body_tokenized()
        if len(tokens) == 0:
            return 0, 0, 0
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

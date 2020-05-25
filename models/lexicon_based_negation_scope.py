import data_importer
from models.model_base import ModelBase

LM_POSITIVE_WORDS, LM_NEGATIVE_WORDS = data_importer.import_lm_dictionary()
NEGATE = {
        "aint",
        "arent",
        "cannot",
        "cant",
        "couldnt",
        "darent",
        "didnt",
        "doesnt",
        "ain't",
        "aren't",
        "can't",
        "couldn't",
        "daren't",
        "didn't",
        "doesn't",
        "dont",
        "hadnt",
        "hasnt",
        "havent",
        "isnt",
        "mightnt",
        "mustnt",
        "neither",
        "don't",
        "hadn't",
        "hasn't",
        "haven't",
        "isn't",
        "mightn't",
        "mustn't",
        "neednt",
        "needn't",
        "never",
        "none",
        "nope",
        "nor",
        "not",
        "nothing",
        "nowhere",
        "oughtnt",
        "shant",
        "shouldnt",
        "uhuh",
        "wasnt",
        "werent",
        "oughtn't",
        "shan't",
        "shouldn't",
        "uh-uh",
        "wasn't",
        "weren't",
        "without",
        "wont",
        "wouldnt",
        "won't",
        "wouldn't",
        "rarely",
        "seldom",
        "despite",
        "n't"
    }


def check_negation(token_index, tokens, n=3):
    """Check if the n tokens before the a specific token is a negation token"""
    for prepending_token in range(1, n+1):
        i = token_index - prepending_token
        if i >= 0 and tokens[i] in NEGATE:
            return True

    return False


class Model(ModelBase):
    def article_sentiment(self, article):
        """Returns the positive and negative sentiment scores of a list of tokens as a tuple"""
        tokens = article.body_tokenized()
        positive_tokens, negative_tokens, neutral_tokens = 0, 0, 0

        for index, token in enumerate(tokens):
            token = token.upper()
            negation = check_negation(index, tokens)
            positive = token in LM_POSITIVE_WORDS
            negative = token in LM_NEGATIVE_WORDS
            if positive or (negative and negation):
                positive_tokens += 1
            elif negative or (positive and negation):
                negative_tokens += 1
            else:
                neutral_tokens += 1

        positivity = positive_tokens / len(tokens)
        negativity = negative_tokens / len(tokens)
        neutrality = neutral_tokens / len(tokens)
        return positivity, negativity, neutrality

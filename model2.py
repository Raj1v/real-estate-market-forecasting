import pandas as pd
import articles as art
from model import Model


class Model2(Model):

    def classify_article(self, article):
        """Classifies an article using SVM"""
        review_vector = self.vectorizer.transform([article.body])
        classification = self.classifier.predict(review_vector)[0]
        return classification

    def article_sentiment(self, article):
        positive, negative, neutral = False, False, False
        classifcation = self.classify_article(article)
        if classifcation == '1':
            positive = True
        elif classifcation == '-1':
            negative = True
        else:
            neutral = True

        return int(positive), int(negative), int(neutral)

    def __init__(self, articles, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.articles = articles

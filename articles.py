from bs4 import BeautifulSoup
import nltk
import csv
import sys
import os
import pickle

csv.field_size_limit(sys.maxsize)


class Article:
    def __init__(self, headline, body, month, year, source):
        self.year = int(year)
        self.month = int(month)
        self.headline = headline
        self.body = BeautifulSoup(body, 'html.parser').get_text()  # Clean HTML tags from body
        self.source = source

    def __repr__(self):
        return '<Article {headline: \'' + self.headline + '\', date: ' + str(self.month) + "-" + str(
            self.year) + ", source: " + self.source + '}>'

    def headline_tokenized(self):
        """Returns a tokenized version of headline with stopwords removed as a list"""
        # TODO: Remove stopwords
        tokens = nltk.word_tokenize(self.headline)
        return tokens

    def body_tokenized(self):
        """Returns a tokenized version of body with stopwords removed as a list"""
        # TODO: Remove stopwords
        tokens = nltk.word_tokenize(self.body)
        return tokens


def load_articles_ft(path="data/articles_ft.csv"):
    articles = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            date, body, headline = row[-1], row[-2], row[-3]
            month, year = date.split('-')
            article = Article(headline, body, month, year, source="ft")
            articles.append(article)
    return articles


def load_articles_guardian(path="data/articles_guardian.csv"):
    articles = []
    with open(path) as f:
        reader = csv.reader(f)
        csv_headers = next(reader)
        headline_index = csv_headers.index('headline')
        body_index = csv_headers.index('trailText')
        date_index = csv_headers.index('webPublicationDate')
        for row in reader:
            date, headline, body = row[date_index], row[headline_index], row[body_index]
            year, month = date.split('-')[0:2]
            article = Article(headline, body, month, year, source="guardian")
            articles.append(article)
    return articles


def load_articles(use_cache=True):
    pickle_path = "pickles/articles"
    if os.path.exists(pickle_path) and use_cache:
        infile = open(pickle_path, 'rb')
        articles = pickle.load(infile)
        infile.close()
    else:
        articles_ft = load_articles_ft()
        articles_guardian = load_articles_guardian()
        articles = articles_ft + articles_guardian
        outfile = open(pickle_path, 'wb')
        pickle.dump(articles, outfile)
        outfile.close()
    return articles


def months_articles(articles, month, year):
    results = []
    for article in articles:
        if article.month == month and article.year == year:
            results.append(article)
    return results

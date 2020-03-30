import model1
import articles as art
import random
import csv

articles_guardian = art.load_articles_guardian()
articles_ft = art.load_articles_ft()
articles = articles_ft + articles_guardian

# Write sentiment scores to csv file
sent_pos, sent_neg = model1.timespan_sentiment(articles, 2000, 2020)
with open('sent_normalized.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    i = 0
    for year in range(2000, 2020 + 1):
        for month in range(1, 13):
            wr.writerow([str(year) + "-" + str(month), sent_pos[i], sent_neg[i]])
            i += 1

# Print some tokenized articles from both newspapers for inspection
print("GUARDIAN:")
for article in random.sample(articles_guardian, 10):
    print(article.body_tokenized())
print("FT:")
for article in random.sample(articles_ft, 10):
    print(article.body_tokenized())

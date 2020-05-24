import pandas as pd
import os.path
import csv


def import_nonfarm_data():
    """Imports sentiment tagged sentences from Levenberg Non-farm paper data"""
    sources = ["AP", "DJ", "LB", "MA", "WS", "z_OT"]
    years = range(2000, 2013)
    basepath = "data/nonfarm/sources"

    def get_paths():
        """Returns lists of all folders from which to extract data"""
        paths = []
        for source in sources:
            for year in years:
                year = str(year)
                for month in range(1, 13):
                    month = '%02d' % month  # Pad zero for single digit months
                    path = basepath + "/" + source + "/" + year + "/" + month
                    paths.append(path)
        return paths

    paths = get_paths()
    contents = []
    labels = []
    confidences = []

    for path in paths:
        sentences_path = path + "/raw.txt.tok.sntcs"
        features_path = path + "/raw.txt.tok.feats"

        if not os.path.exists(sentences_path) or not os.path.exists(features_path):
            continue

        with open(features_path) as features:
            for feature in features:
                label = feature.split(" ")[0]

                if label == '\n':
                    confidence = None
                else:
                    confidence = feature.split(" ")[4]

                labels.append(label)
                confidences.append(confidence)

        with open(sentences_path) as sentences:
            for sentence in sentences:
                contents.append(sentence)

    data = pd.DataFrame({'Sentence': contents, 'Label': labels, 'Confidence': confidences})
    data['Confidence'] = pd.to_numeric(data['Confidence'])
    return data


def import_lm_dictionary():
    """Imports the positive and negative wordlists from the Loughran-McDonald sentiment lexicon"""
    with open('data/lm-sentiment-wordlist/lm_negative.csv') as f:
        reader = csv.reader(f)
        next(reader)
        LM_NEGATIVE_WORDS = [word[0] for word in list(reader)]

    with open('data/lm-sentiment-wordlist/lm_positive.csv') as f:
        reader = csv.reader(f)
        next(reader)
        LM_POSITIVE_WORDS = [word[0] for word in list(reader)]

    return LM_POSITIVE_WORDS, LM_NEGATIVE_WORDS

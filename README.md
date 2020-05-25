# Influence of sentiment on real estate markets
This repository offers an modular and extendable framework for analyzing real estate markets using sentiment models

# About the project
## Motivation
This project exists as part of a bachelor's thesis to graduate from the Amsterdam University College with a degree in Liberal Arts & Sciences. The title of the thesis is `THE INFLUENCE OF MEDIA SENTIMENT ON THE BRITISH REAL
ESTATE MARKET` and utilizes the code in this repository to analyze the British Real Estate market using sentiment analysis on articles from The Guardian and Financial Times newspapers.

## Features
+ Machine learning (SVM) sentiment analysis model
+ Lexicon-based sentiment analysis model
+ Vector autoregression to correlate sentiment with real estate data
+ Exporting model results as LaTex tables

## Highlights
+ Easy implementation of aditional sentiment models using inheritance from a base class
+ Easy to change retrain SVM models with different parameters or training data
+ Caching to speed up common tasks (e.g. article loading, model training, etc.)
+ Well documented code (docstrings for classes and functions, annotated notebooks, comments where needed)


## Built with
+ Python 3
+ Jupyter Notebooks

# Getting started
## Downloading data
The first thing to do before using this tool is to provide some data. For the original research, the following data was used:
+ News articles from The Guardian and The Financial Times
+ Sentiment annotated text from several financial news sources (Training data for machine learning model)
+ Consumer confidence data
+ Loughran-McDonald financial sentiment lexicon (For the lexicon-based sentiment model)
+ Real estate data
    + Direct real estate market (UK National Statistics)
    + Securitized real estate market (FTSE EPRA NAREIT UK)

The paper associated with this project provides more detail on the used dataset.
To get started, it is highly advisable to run and test this project with the same data, which can be downloaded here.
The entire zipfile should be extracted in the `data` folder

## Running and evaluating models
The most straightforward way to run and evaluate models is through executing the `Evaluation` Jupyter notebook in the `notebooks` folder. 

Alternatively, the code belows shows how a Python file can evaluate models.

    import articles as art
    from evaluation import Evaluator
    import models.lexicon_based

    # Load Financial Times & The Guardian articles
    articles = art.load_articles()

    # Initialize the sentiment model
    model = models.lexicon_based.Model(articles)

    # Evaluate the model on a given real estate index
    real_estate_index = 'data/real_estate_data/direct/england.csv'
    ev = Evaluator(model, articles, real_estate_index)
This code will run the lexicon-based sentiment model on the direct real estate market and output its results and performance metrics, such as the regression coefficients with confidence intervals, plots of its forecast results and its MAPE.

For more detailed information and an example of running the slightly more complex machine learning model, see the `Evaluation` notebook.

## Creating new models
It is easy to implement and analyze new sentiment models by inhereting from the base model class and implementing the `article_sentiment` method. The code below implements a simple new sentiment model.

    from models.model_base import ModelBase

    class Model(ModelBase):
        """
        Ultra-simple lexicon-based sentiment model
        """
        def article_sentiment(self, article):
            """Returns the positive and negative sentiment scores of a list of tokens as a tuple"""

            # Use the functions available through Article class
            tokens = article.body_tokenized()

            positivity, negativity, neutrality = 0,0,0
            if 'happy' in tokens:
                positivity = 1
            elif 'sad' in tokens:
                negativity = 1
            else:
                neutrality = 1

            return positivity, negativity, neutrality

If desired, `__init__` can be overwritten to add more parameters to a model, however, each `__init__` must set the `self.articles` attribute.

All models inheriting from the `ModelBase` class are automatically compatible with the evaluation scripts.

## Other common tasks
The Jupyter notebooks in the `notebooks` guide you through the following other tasks:
+ Complex evaluation & exporting results as LaTex tables
+ Training SVM models
+ Determing the classification confidence of SVM models
+ Working with sentiment data from an external model (in the example: Consumer Confidence)
+ Establishing a simple baseline autoregressive model on real estate indices

# Furter customization
Below are some quick tips on getting started with further customizing the framework

# File structure
    .
    ├── data                        # Data used by models (e.g.: news articles, sentiment lexicons, real estate data)
    ├── models                      # Sentiment models
    ├── notebooks                   # Jupyter notebooks for various tasks such as evaluating and training models
    ├── pickles                     # Stored pickle files to reuse the output of demanding tasks 
    ├── articles.py                 # Helper functions and class for dealing with news articles
    ├── data_importer.py            # Helper functions for various data loading purposes
    ├── evaluation.py               # Class for evaluating the performance of sentiment models
    ├── latex_tables.py             # Printing resutls of model evaluation as a latex table
    └── vector_auto_regression.py   # Implementation of vector autoregression on sentiment models

See the `README.MD` files within the `models`, `notebooks` and `pickles` folder for a rundown of their individual files.

# License
This work is licenced under the MIT Licence. 
# Credits


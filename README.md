# Influence of sentiment on real estate markets
This repository offers an modular and extendable framework for analyzing real estate markets using sentiment models

# About the project
## Motivation
This project exists as part of a bachelor's thesis to graduate from the Amsterdam University College with a degree in Liberal Arts & Sciences. The title of the thesis is `THE INFLUENCE OF MEDIA SENTIMENT ON THE BRITISH REAL
ESTATE MARKET` and utilizes the code in this repository to analyze the British Real Estate market using sentiment analysis on articles from The Guardian and Financial Times newspapers.

## Features & Highlights
+ Vector autoregression to 
+ Easy implementation of aditional sentiment models using inheritance from a base class
+ Use pickles as cache to reuse model results
+ Exporting model results as LaTex tables


## Results


## Built with
+ Python 3
+ Jupyter Notebooks

# Getting started
## Downloading data

##  

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


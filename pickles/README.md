# Pickles
This folder contains Python objects stored as pickled that can be used to skip expensive parts of the sentiment analysis process.

    .
    ├── articles              # All articles of the default dataset (The Guardian + Financial Times)
    ├── svm_classifier        # Trained SVM classifier on annotated financial news sentences provided by Levenberg et al. (2018)
    ├── svm_classifier_prob   # Same SVM classifier but constructed to give sentiment class probability estimates
    └── vectorizer            # Vectorizer used in both SVM classifiers

# Notebooks
Each notebook in this folder demonstrates one of the functionalities that can be implemented using this project.

+ **Baseline:** Implements and runs an autoregressive model on real estate indices to serve as a baseline to evaluate sentiment models
+ **Consumer Confidence Model:** Shows how to implement a model that uses sentiment from an external source such as a CSV file. In this case consumer confidence data is used.
+ **Evaluation:** Shows to run and evaluate sentiment models constructed within the framework of this project
+ **SVM Confidence:** Displays how to assess the classification confidence of SVM models
+ **Train SVM:** Shows how to train SVM models

Note that each notebook starts with:
```python
import os
os.chdir('..')
```
This is done to set the working directory to the main folder of this repository. 

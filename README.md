# Project 4 - Credit card approval - PD credit card

# 1. Project Motivation

The motivation of this project was to create a model to predict and identify the probability of default base in the clients that have information in the last 12 months also as a data scientist working in the banking business have the oportunity to work with this kind of problems and using varaibles that normal and do not use, it was challenging base that for reason of habeus data information as family size or something like that we are not able to use. This project have some work to do and using manchine learning model beside of normal credit score card.

# 2. File Descriptions

The information was taking from Kaggle the information of this are in the next link: [Kaggle - Credit Card Approval](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) in this link you will have the description of the variables that have and download.

When you download have two csv files:

- application_record.csv -  have the clients information.
- credit_record.csv -  the history of credit behavior.

# 3. How to Interact with your project

For interact with this project ypu have to download the databases descritive before, after follow the requiremnts below for working with the jupyter notebook and follow the process of analysis, clean, modeling, results and conclusions.

In the gitbub follow:
- Download zip file, extract the files
- jupyter notebook
- Blog with the more important findings

# 4. Requirements

For the development of this project we use diferent kind of libreries the list of those it is listing next:


- Numpy
- Pandas
- Skelearn
- Matplotlib
- Seaborn
- Stat models
- Imblearn

However for more information exactly which function are need it this is the list:

- import pandas as pd
- import numpy as np
- import seaborn as sns;sns.set_theme(style="darkgrid")
- import matplotlib.pyplot as plt
- from datetime import datetime
- import math
- from yellowbrick.model_selection import LearningCurve
- from yellowbrick.classifier import ROCAUC
- from sklearn.pipeline import Pipeline, FeatureUnion
- import statsmodels.api as sm
- from statsmodels.formula.api import ols
- from sklearn.model_selection import GridSearchCV,StratifiedKFold
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler, MinMaxScaler
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.linear_model import LogisticRegression
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.svm import SVC
- from sklearn.datasets import make_regression
- from sklearn.tree import export_graphviz
- from sklearn.inspection import permutation_importance
- from sklearn.preprocessing import RobustScaler
- from sklearn.base import BaseEstimator
- from sklearn.base import TransformerMixin
- from xgboost import XGBClassifier
- from datetime import timedelta
- import datetime
- from imblearn.over_sampling import SMOTE
- import shap
- from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix, classification_report, auc
- from sklearn.metrics import roc_curve, roc_auc_score,make_scorer
- from sklearn.model_selection import cross_val_score, KFold,cross_val_predict,StratifiedKFold,RandomizedSearchCV
- pd.set_option('display.max_columns',None)
- %matplotlib inline

Also this project where produce in python 3 and the version of the libreries was the newest of day write this readme.

# 5. Summary of the results
 
check out:

https://juliansufl.github.io/udacity_data_scientist_project1/

# 6. Acknowledgments

scikit-learn developers. (n.d.). sklearn.linear_model.LogisticRegression — scikit-learn 1.1.1 documentation. Retrieved May 12, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Pedregosa, F., Weiss, R., & Brucher, M. (2011). Scikit-learn : Machine Learning in Python. 12, 2825–2830.

Also in the notebook you find more information

# Thank you! Gracias totales!


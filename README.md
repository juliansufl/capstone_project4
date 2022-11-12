# Credit Card Approval - Calculation of PD

![credit-cards-4-steps](https://user-images.githubusercontent.com/60525865/201497267-55cbf1d0-89ca-404c-beaa-6541fe2da1eb.jpg)

# Introduction

In this project, the idea it is resolve a common situation that presents in the baking business, when they have to management credit risk which seeks for identify the possibility of get loses lead for non-payment or defaulf when a counterpart does not fulfill their obligations with the entity.

For that reason, this notebook wants to create a model to predict which kind of clients that apply for a credit card can get approval base in the historial of application and client information. It is commmon in the business base on new informartion of clients create models with new information and get better prediction base in the time horizon take it. Otherwise, in the banking a default it is define when a counterpart get 60 or highest days in non-payment that kind of moment it is considered a bad client this apply for credit cards which is the case of study.

In conclusion, we will implements some manchine learning all classifier as, Xgboosting, Decision Tree, Random Forest, Logistict Regression and SVM Classifier to make a prediction of which is the probability of bad clients that we can predict. To know which is the best model we will take the best F1-Score, this metrics are good for classifier problems bacause take into account the precision and recall of a prediction for better results. This kind of work can help me in the daily of my work as data scientist in the banking business.

The information is taking from [KAGGLE - CREDIT CARD APPROVAL](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) for look of the dictionary of each variable.

## Part 1: EDA



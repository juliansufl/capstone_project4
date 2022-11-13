# Credit Card Approval - Calculation of PD

![credit-cards-4-steps](https://user-images.githubusercontent.com/60525865/201497267-55cbf1d0-89ca-404c-beaa-6541fe2da1eb.jpg)

# Introduction

The idea it is resolve a common situation that presents in the baking business, when they have to management credit risk which seeks for identify the possibility of get loses lead for non-payment or defaulf when a counterpart does not fulfill their obligations with the entity. Nowadays this it is common a generated some financial problems for the banks for that reason it is need it the credit risk management with new model and new strategy to get good clients.

For that reason, this blog has the explanation of the creation of a model to predict which kind of clients that have a credit card approved base in the historial of clients application and credit balance information. It is commmon in the business get new clients or new application for the existing ones, for that new informartion of clients it is need it new models with that information to get better prediction base in the time horizon take in. Otherwise, for that reason in the banking a default it is define when a counterpart get 60 or highest days in non-payment that kind of moment it is considered a bad client in this case in period of the last 12 months this apply for credit cards which is the case of study.

We will implement a manchine learning classifier as Xgboosting to make a prediction of which is the probability of bad clients that we can predict adn get base in the information. To know which is the best model we will take the best F1-Score, this metrics are good for classifier problems bacause take into account the precision and recall of a prediction for better results also when we have a mayority class in the data. This kind of work can help in the daily of work as data scientist in the banking business.

## Part 1: EDA

As we mention in the introduction we have information in two resources clients application and credit balance. First, clients application has the information of income, income source, nucleus family, own car, own property, type of house and etc... In the credit balance we have the ID, balance month and status which how the payments have be do it for clients, for that reason we need to fin the clients that we have credit balance information and create the BGI(Bad and Good Indicator).

### Clients Application 
![application](https://user-images.githubusercontent.com/60525865/201498525-96cbf72b-8bd9-4204-96d6-4f69eafa2c81.png)

### Credit Balance

![credit](https://user-images.githubusercontent.com/60525865/201498544-039d01b2-cd58-420b-873a-4dbf399bf1c1.png)

With this we seek for the information that is mutual in the base and joining it have one base:

![df](https://user-images.githubusercontent.com/60525865/201541758-3414ca65-5ea4-4eb4-889a-97890757279f.png)

After this we looking for strage behavior in the information

![Nulls](https://user-images.githubusercontent.com/60525865/201498768-b64d8b64-3850-4f14-95ab-26c1a662a22f.png)

We just have null information in the occupation varible that is good.

Now us we said in the beginning we want to predict the PD of the clients that have information in the last 12 months for that reason we need to find those clients and get the information, for that we filter the clients that have 11 months of credit balance information and after look for them in the current month us 0 and that we know that we have the 12 months.

![clients_12](https://user-images.githubusercontent.com/60525865/201498946-419d3031-b3e2-4ce1-9a93-8af06e64676d.png)






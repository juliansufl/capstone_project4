# Credit Card Approval - Calculation of PD

![credit-cards-4-steps](https://user-images.githubusercontent.com/60525865/201497267-55cbf1d0-89ca-404c-beaa-6541fe2da1eb.jpg)

# Introduction

The idea of this blog, it is resolve a common situation that presents in the baking business, when they have to management credit risk which seeks for identify the possibility of get loses lead for non-payment or defaulf when a counterpart does not fulfill their obligations with the entity. Nowadays this it is common a generated some financial problems for the banks for that reason it is need it. The credit risk management improve with new model and new strategy to get good clients.

For that reason, this blog has the explanation of the creation of a model to predict which kind of clients that have a credit card approved are good, for that base in the historial of clients application and credit balance information, it is possible do an estimation. It is commmon in the business get new clients or new application for the existing ones, for that new informartion of clients it is need, with new models and new information to get better prediction base in the time horizon take in or to analyze. Otherwise, for that reason in the banking a default it is define when a counterpart get 60 or highest days in non-payment that kind of moment it is considered a bad client in this case in period of the last 12 months this apply for credit cards which is the case of study.

For identify the Probabilitu of Default(PD) we will implement a manchine learning classifier as Xgboosting to make a prediction of which is a bad clients that we can predict base in the information. To know which is the best model we will take the best F1-Score, this metrics are good for classifier problems bacause take into account the precision and recall of a prediction for better results also when we have a mayority class in the data. This kind of work can help in the daily of work as data scientist in the banking business.

## Part 1: EDA

As we mention in the introduction we have information in two resources clients application and credit balance. First, clients application has the information of income, income source, nucleus family, own car, own property, type of house and etc... In the credit balance we have the ID, balance month and status which how the payments have be do it for clients, for that reason we need to find the clients that we have credit balance information and create the BGI(Bad and Good Indicator).

### Clients Application 
![application](https://user-images.githubusercontent.com/60525865/201498525-96cbf72b-8bd9-4204-96d6-4f69eafa2c81.png)

### Credit Balance

![credit](https://user-images.githubusercontent.com/60525865/201498544-039d01b2-cd58-420b-873a-4dbf399bf1c1.png)

With this data bases we have to seek for the information that is mutual in the bases and joining it, and just have one base, because we have to take the clients that have history in the credit balance

![df](https://user-images.githubusercontent.com/60525865/201541758-3414ca65-5ea4-4eb4-889a-97890757279f.png)

As we see that it is the number of clients between the both bases and the number of variables available.

With this it is necessary look for null values to beging to create a plan to clean up the data and also create the dependent variable for find the PD. 

![Nulls](https://user-images.githubusercontent.com/60525865/201498768-b64d8b64-3850-4f14-95ab-26c1a662a22f.png)

We just have null information in the occupation varible that is good.

Now us we said in the beginning we want to predict the PD of the clients that have information in the last 12 months for that reason we need to find those clients and get the information, for that we filter the clients that have 11 months of credit balance information and after look for them in the current month us 0 and that we know that we have the 12 months. 

![clients](https://user-images.githubusercontent.com/60525865/201544485-602b75d2-0c75-4dcf-9f62-441c9808c658.png)

With the number of clients that have 12 months, we want to find the maxinum of the status as we mention, in the credit balance information have days past due. This variable need clean up for define the BGI, first the 'C' is pay that month and 'X' the client does not have any purchase, from 0 to 5 represent as going up as 30 days past due for each number. We clean the string find the maximum for the clients identify in the 12 months period and create the BGI. With this every client that have greater that 1 it is a bad clients else as 0.

![BGI_unbalanced](https://user-images.githubusercontent.com/60525865/201545050-55f002c6-c01d-4e7d-9086-4d5ae2d2355e.png)

With the dependet variable define we can look for more strange things in the rest of the dataset, although we can see that the BGI it is the unbalanced.

### Cleaning

As we see before the the occupation variable has NaN values, this for plotting and know the dimension of that we change the null for 'Other'

![ocuppation](https://user-images.githubusercontent.com/60525865/201545669-1adaa557-d5ee-4464-98aa-6cd7883046d8.png)

This variable has at least 30% of others but in the data base it is a variable that have the income source. 

![Income source](https://user-images.githubusercontent.com/60525865/201545760-a3c66814-02d2-47d9-a343-f58070a0b93a.png)

It is that the occupation variable can be drop because we have the information from the income source of the clients and that give us enough information for this matter.

In the data we have the days of employment and the days birth of the people for that matter we create a fucntion to fix that and have the age of employment and age for every client. We use a boxplot for this vairables

![age](https://user-images.githubusercontent.com/60525865/201548046-6a32c985-fb72-4358-836c-c1bc96d718f7.png)

![age_employ](https://user-images.githubusercontent.com/60525865/201548064-e240ddee-b1c7-42c4-8490-d381561f6fa4.png)

For the case age emplyment have outliers follow for other variables as income and children by nucleus family that information has been fix by a function using the Turkey method

    def tukey_method(data,varible):
    '''
    This function create a range interquartle for identify outliers
    
    INPUT: Database to clean, 
           Name the variable with outliers
    OUTPUT: DataFrame with the ID and the Value of the outliers
    '''

    outliers_age_employ = df_pd[['ID',varible]]
    #finding quantiles

    q1 = outliers_age_employ[varible].quantile(0.25)

    q3 = outliers_age_employ[varible].quantile(0.75)

    #calcule interquartile

    iqr = q3-q1

    #finding the min a max interquartile

    max_value = q3 + 1.5 * iqr

    min_value = q1 - 1.5 * iqr

    outliers = outliers_age_employ[(outliers_age_employ[varible] > max_value) | (outliers_age_employ[varible] < min_value)]
    return outliers

![age_employ2](https://user-images.githubusercontent.com/60525865/201548531-034cffbb-be4c-478d-8369-5634ca2a2960.png)

We this this method we remove the outliers in the variables that we mention and we see that it works. Now we have the data ready.

## Part 2: Methodology

### Preprocessing

Before the modeling process it is necessary to devide the data into train and test, the divide was take 70-30. With the train and test define, as pass adove it was create the BGI we identify that the variable is unbalanced for that reason we use a metodology call SMOTE, this method create in a synthetic way to recreate the more scenario of the minority class in this case the bad clients.

![bgi_balance](https://user-images.githubusercontent.com/60525865/201549305-83912a8f-bc20-4528-9585-4a4b109b6e81.png)

This balance just had done in the train bases because when we implement a model and need to do prediction base in new data we are not going to have that on balanced for that reason we do it this way, and get prediction base on that. Also create code that let us preprocess numerical features with robust RobustScaler and categorical features.

    ''' 
    # get the numerical variables names
    numerical_features = list(X.select_dtypes(["int64","float64"]).columns)

    #create the steps for the numerical pipeline
    numerical_steps = [('num_selector',FeatureSelector(numerical_features)),
                       ('std_scaler',RobustScaler())]

    #create two piples with the respective steps
    numerical_pipeline =  Pipeline(numerical_steps)

    #create a list of the pipeline
    pipeline_list = [('numerical_pipeline',numerical_pipeline)]
    #create the full pipeline
    pipeline_preprocessing = FeatureUnion(transformer_list=pipeline_list)
    
    '''

### XGBoost Classifier Implementation

For modeling we will use a the Extreme Gradient Boost Classifier (XGBoost). This method has emsembles from decision tree models, that added trees one by one to emsemble and fit the correct error of prediction that are made for each run by prior models. This kind of models as know as a boosting.

As we implement a model, this run by default parameters but this kind change by using a parameter tuning. This is the common parameters.

![Key-parameters-used-for-XGBoost-classification](https://user-images.githubusercontent.com/60525865/201549921-938db8b3-5d39-4388-b8d0-ad64a51e77f1.png)

[Take from](https://www.researchgate.net/figure/Key-parameters-used-for-XGBoost-classification_tbl1_322791200)

### Parameter tunning

This parameter can be change to get the best fit for prediction in the model for that we define this parameters

    '''
    param_grid={"learning_rate":[0.001],
                "colsample_bytree":[0.6,0.8],
                "max_depth":[2,4,6,8],
                "subsample":[0.6,0.8],
                "n_estimators":[100,200,300],
                "reg_lambda":[0.5,1,1.5,2],
                "gamma":[0,0.1,0.3],
                "random_state":[42]
                }
    '''
To get the best parameters we use the metric "error" and "auc" with this we secure that get the best

![image](https://user-images.githubusercontent.com/60525865/201550104-4c57eba0-86bd-44d9-bd87-84cf945b59ef.png)

When the XGBoost select between the pull of values that we put on each parameter, those help ud to reduce the error to get the best parameters. In the image above we can see a clear example of how the algorithm works. For the parameter 'gamma' we put the values of [0,0.1, 0.3] and the model found a optimization processes that using the value of 0.3 is better than others. That happen with every one of them that we difene in the grid.

## Part 3: Results

![results](https://user-images.githubusercontent.com/60525865/201550443-c20e8fc6-7108-4714-baf3-99abccae1570.png)

![accu](https://user-images.githubusercontent.com/60525865/201550461-097299a6-0183-4bad-858f-da9a08078d9c.png)

After we implement the model we can see that we are getting a F1 score the 14.29% for prediction for new bad clients, the importance of F1 score helps for taking into the count the precision to identify new cases and the recall for that new cases in the model. Also we can see that we the balance data tha we use and the test unbalance we are getting a good accurancy base in the model and the variables that we are using.

In the banking business the uses manchine learning models have generate doubt prefer not to use this type of models because they think that is a black box and it is complex. With the help of the SHAP package that have the shapley values methos it is posible to example how this black box works when we iterature with the variables.

In first place, the package hace plot for importance and Weight Feature, has Split Mean Gain and Sample Coverage. The meaning of each are:

- Feature Weight: The weight of each feature

- Split Mean Gain: Implies the relative contribution of the corresponding characteristic to the model. A higher value for this metric compared to another features means that it is more important for generating a prediction.

- Sample Coverage: Means the relative number of observations related to this characteristic.

![featuer](https://user-images.githubusercontent.com/60525865/201550960-6dc69832-16ec-4b63-b7b8-63cb274e8802.png)

In the graphic we can see that the 13 features uses in the model have some relevance in the model but some do not have the some much importance, for exemple 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'FLAG_EMAIL' and 'AGE_EMPLOY', that varaibles do no get the best for prediction of bad clients. However, 'STATE_PROPERTY', 'CARS','NAME_EDUCATION_TYPE' and 'NAME_INCOME_TYPE' are the variables with more importance for predict the bad clients.

Shap packages have also a plot with detail with blue how positive afect the variable and red for how bad afect the variable to predict for new cases.

![shap](https://user-images.githubusercontent.com/60525865/201551459-8f1175a3-f5af-4647-8b40-3ff8571fd59a.png)

As we can see this indicate us when the variable is in blue color that this range decreases the prediction of the probability that the client will have non-payment in the future. On the other hand, when the variable is in the red color, its default probability prediction begins to increase.

### Justification

The results that we obtain are two reasons. First, the XGBoost model reduce the regularization, that have method as L1 and L2, that give more complexity in the model. The creation of new trees that predict errors from previous trees that are then combined with previous trees. This is the reason why this model has become so popular and has won so many challenge contests for the performance for iteration.

The last one is that during the process we explained the paramater tunning, we did not use the default paramerters, we use a model with the option to select the best parameters.

## Part 4: Conclusion

![image](https://user-images.githubusercontent.com/60525865/201551763-4209dc10-bc89-4858-9184-c6a27b9634c1.png)

The propuse of this blog was to identify which variables are the best for predict the probability of default in clients that have credit cards aprovals, taking into the count clients that have information in the last 12 months and the max of non-payment in that period of time and getting the nowadays information.

Non-payment was define as 60 days or more without pay that give us the dependent variable and after looking for the clients that really have 12 month of information. When we implement the models knew that we have a unbalanced target we used a SMOTE for balanced.

With the implementation of diferents models and getting the best model as XGBoost Classifier that have the best F1 score base in the way that precision and recall relation for this metric and also when we have unbalanced target, also making a comparative with a random forest model other model base in tree has more prediction of bad clients in time as show the imagen above.

Use the black box machine learning models work quite well in the banking business. This can explain the behavior of all features, using analyze within the data before modeling, be able to using shapley velues to understand the iteration with the features with the model it is a huge advantage. It is important when we are going to implement a credit risk model for the default in the last 12 months for the information of each client and make sure that the clients have the period of time.

In conclusion the model select it is posible to determine 10.83% more of bad clients base in the information have, this means that in bank if you put as a credit 1 millon of dolars you are able to loss 100k, when before you are just viable to predict 2k of lossing in the long run that is a problem for the business of credit that you want to recolect the money that you lend and get the interest from that.



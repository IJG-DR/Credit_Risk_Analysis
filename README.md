# Credit Risk Analysis
Module 17 repository


## Overview

The purpose of this analysis is to apply machine learning to predict credit card risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, the analysis employs different techniques to train and evaluate models with unbalanced classes. For this purpose, the *imbalanced-learn* and *scikit-learn* libraries were utilized to build and evaluate several models using resampling techniques.

### Methodology

This analysis is based on a credit card dataset from *LendingClub*, a peer-to-peer lending services company. The data was provided as a *csv* file, *LoanStats_2019Q1.csv*, and read into a *jupyter notebook* using the *Pandas* library toolset. The dataset contains 68,817 card records. 

Before implementing the machine learning models, the data was inspected, and categorized fields that were selected to be used as features were encoded into numerical values using the *get_dummies()* method. 

The target field for our prediction was *loan_status*, as a proxy for credit risk.   We defined credit risk as a binary outcome, even though the data contained in the *loan_status* field had more granularity to it. 

Low-risk cards were defined as those with *loan_status* as current in their payment. Cards  with overdue payments, including those within the grace period (first 15 days), as well as those in the buckets comprised of 16-30 days overdue, 30-120 days overdue, and those in default (beyond 120 days), were defined as high-risk. With this split, the data counts reported 68,470 low-risk accounts and 347 high-risk accounts. 

The data was then split into training and testing sets using *sklearn*'s *train_test_split()* function.

Next, the following resampling models were applied, and a Logistic Regression classifier was used to make predictions:

* oversampling using *RandomOverSample*
* oversampling using *SMOTE*
* undersampling using *CLusterCentroids*

Then, a *SMOTEENN* combinatorial approach of over and undersampling was applied, and a Logistic Regression classifier was used to make further predictions.

Afterwards, utilizing the *imblearn.ensemble* toolkit library, random forest ensemble classifiers were applied with 100 estimators, using the following classifiers to make predictions:

* *BalancedRandomForestClassifier*
* *EasyEnsembleClassifier*

After each of the six classifiers were applied, an *accuracy*, *confusion matrix* and an *imbalanced classification report* were generated for the purpose of comparison.

The *Python* code utilized for the analysis is contained in two *ipynb* files included in this repository as *credit_risk_resampling.ipynb* and *credit_risk_ensemble.ipynb*.

## Results

The output reported after running each of the models is presented in the following code sections:

### * Random Oversample Classifier

![Random Oversample Classifier](Resources/images/random_oversampler.png)

### * SMOTE Classifier

![SMOTE Classifier](Resources/images/smote_oversampler.png)

### * Cluster Centroids Classifier

![Cluster Centroids Classifier](Resources/images/clustered_centroid_undersampler.png)

### * SMOTEENN Classifier

![SMOTEENN Classifier](Resources/images/smoteenn_over-undersampler.png)

### * Balanced Random Forest Classifier

![Balanced Random Forest Classifier](Resources/images/balanced_random_forest.png)

The report of feature importance is presented below, sorted in descending order (from most to least important feature), along with the feature score.

![Feature Importance](Resources/images/features_1.png)
![Feature Importance](Resources/images/features_2.png)
![Feature Importance](Resources/images/features_3.png)


### * Easy Ensemble Classifier

![Easy Ensemble Classifier](Resources/images/easy_ensemble.png)

## Summary

As can be established from the different summary statistics of the models, none of the models was very good at predicting high-risk cards, although most of them had good prediction of low-risk cards. This is evident from the low precision score for high-risk, which ranged between 0.01 and 0.09. The Random Forest and Ensemble models faired better in both respects than the different sampling-based models, presenting the highest precision score for high-risk cards. The Cluster Centroid Undersampling model faired the worst in accuracy, precision and recall for both low-risk and high-risk cards. 

In addition to theses models, we also performed a second run of each set. For this second run, we changed the original criteria for low and high risk. The second run considers that credit cards still within the grace period would also be considered low-risk, in an attempt to determine if any improvement on the model predictions could be achieved this way. 

After running our models again, we saw a very slight improvement in the accuracy statistic, but no significant improvement in the prediction statistic for high-risk. 

By redefining our risk criteria, this resulted in an even smaller number of high-risk training data points (68,641 low risk and 176 high risk, which is almost half the number in the original analysis). By making reducing the number of data points, there is a larger inbalance in the data,which presents a greater challenge when trying to predict outcomes using machine learning models. Ideally, we would like to conduct an alternate analysis with more high-risk data points.

The summary table below compares all of our findings.

![Summary Table for All Models](/Resources/images/Summary_tables_all_models.png)

Should we recommend any of these models? Based on the low prediction statistic for high-risk loans, it is unlikely we would recommend any of them, even though some performed better than others. The high prediction rate for low-risk cards also signals that our model may be overfitting for these cards.

The reason for not recommending them, aside from their low prediction score, is that we would need to first define a desired risk appetite (i.e. and acceptable margin of error). This definition of risk appetite would balance accurate prediction between high-risk cards and low-risk cards. Choosing a model that improves the prediction of high-risk cards could reduce the accuracy of predicting low-risk cards, and some good business would be sacrificed. Again, it is all a matter of defining an acceptable balance and desired cut-off points. This type of analysis would done by including a much larger sample of high-risk cards and reducing the sample of low-risk cards.

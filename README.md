# Deep-Learning-Challenge
------------
## Overview of the Analysis
### Goal
Asses the likelihood of success for a venture that Alphabet Soup is considering funding.

## Softwares Used
### Libraries
Numpy
Pandas
pathlib
    Path
sklearn.metrics
    balanced_accuracy_score
    confusion_matrix
    classification_report
sklearn.model_selection
    train_test_split
sklearn.linear_model
    LogisticRegression
imblearn.over_sampling
    RandomOverSampler

## Preprocessing the Data
-  Read in the charity_data.csv file into a dataframe with Pandas in Google Colab.
-  Dropped the EIN and NAME columns.
-  Used .nunique() to determine the number of unique values in each column.
-  Assessed the counts of values in APPLICATION_TYPE and CLASSIFICATION in order to bin the data.
-  Set cut off values for APPLICATION_TYPE and CLASSIFICATION and replaced values below the cutoff point with "Other".
-  Converted the categorical variables, 'APPLICATION_TYPE', 'INCOME_AMT','CLASSIFICATION','AFFILIATION','ORGANIZATION','USE_CASE' and 'SPECIAL_CONSIDERATIONS' to numeric values with pd.get_dummies.
-  Replaced the categorical variables with the encoded variables in the dataframe and dropped the original categorical columns.
-  Split the dataframe into two arrays: the target array (IS_SUCCESSFUL) and the features array, everything except the IS_SUCCESSFUL column.
-  Used train_test_split to split the data into the training and testing data.
-  Used StandardScaler to fit a scaler to the X_train data, and then apply the scaler to the X_train and X_test data with the transform function.

##  Compile, Train, and Evaluate the Model


## Random OverSampler


## Summary


------------
------------

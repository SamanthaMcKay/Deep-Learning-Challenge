# Deep-Learning-Challenge
------------
## Overview of the Analysis
### Goal
Assess the likelihood of success for a venture that Alphabet Soup is considering funding.

## Softwares Used
### Libraries
Pandas

sklearn.model_selection
    train_test_split
    
sklearn.preprocessing
    StandardScaler
    
TensorFlow

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
-   Defined the model with tf.keras.models.Sequential().
-   Added two hidden layers with 16 neurons each and using the relu activation function.
-   Added an output layer with one neuron using the sigmoid activation function.
-   Compiled the model.
-   Fit the model with 100 epochs.
-   Found the loss and accuracy and saved the results to a h5 file.

  ![Pre Optimization Model](https://github.com/SamanthaMcKay/Deep-Learning-Challenge/assets/132176159/6fda3510-ee87-4140-98b9-ced6cca07658)
  
## Optimize the Model
-   To optimize the model:
  - I dropped the SPECIAL_CONSIDERATIONS column.
  - I tried dropping the USE_CASE, ORGANIZATION, AFFLIATION, STATUS, INCOME_AMT, and ASK_AMT
    in various combinations but none of the combinations increased the accuracy.
  - I lowered the CLASSIFICATION cutoff value from 1000 to 200, which seemed to help a             little.
  - I added a third layer, changed all the hidden layer activation functions to tanh (since the
    scaled data was negative and tanh handles negative values better).
    - I tried various combinations of relu and tanh, but tanh for all three seemed to work best.
  - Each hidden layer has 74 nodes because that is twice the input dimensions.
    - There was no noticeable difference when I increased the nodes past 74.
  - The accuracy increased 0.23% and the loss increased a little from 0.5556 to 0.5565.
  - The file was saved with a unique name to an h5 file.

![Optimized Model](https://github.com/SamanthaMcKay/Deep-Learning-Challenge/assets/132176159/f52ce01e-fb81-40ac-81e6-0ae29b9392cc)

## Report on the Neural Network Model
### Overview
The purpose of this analysis was to train a model to predict the success of a venture in order to advise a non-profit on which ventures to invest funds in.

### Results
#### Data Preprocessing
1.The target variable of my model was the "IS_SUCCESSFUL" column.
2. The features for my model were 'APPLICATION_TYPE',
'INCOME_AMT','CLASSIFICATION','AFFILIATION','ORGANIZATION','USE_CASE' and 'SPECIAL_CONSIDERATIONS'.
3. The SPECIAL_CONSIDERATIONS variable was removed in my optimized model as it was insignificant to the the target.

#### Compiling, Training, and Evaluating the Model
4. For my optimized model, I used 74 neurons per hidden layer because 74 is 2x 37 which is the number of input dimensions. I used three layers and all three layers had the tanh activation function because it incorporates negative values easier.
5. I was unable to achieve the target model performance.
6. I tried removing the columns USE_CASE, ORGANIZATION, AFFLIATION, STATUS, INCOME_AMT, and ASK_AMT one by one, and then tried dropping the organization and affliation columns at the same time. I also tried dropping the income_amt and ask_amt columns at the same time. I did not notice a significant difference when dropping any of the columns. I ended up only dropping the special_considerations column because it seemed the least relevant to predicting the success of a venture. I tried increasing all of the neurons to 99 neurons per layer. It did not make much of a difference. I tried having the activation functions be (relu, tanh,tanh),(tanh,tanh,relu) and (relu,relu,tanh) before settling on having all three use tanh.

### Summary
The data had one inherently numeric variable, ASK_AMT. In my opinion, the model needed more inherently numeric information. I think that having too many categorical variables diminishes the success of a model. It was difficult to increase the accuracy. With more practice with machine learning, I will become more adept at understanding the strengths of different machine learning types, the important steps in preprocessing the data, and the different characteristics and variables of each model that I can adjust to optimize my machine learning models. I think that a random forest model could work better for this dataset.


------------
------------

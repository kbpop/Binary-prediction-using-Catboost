# Load the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# import the data
data_train = pd.read_csv('../input/spaceship-titanic/train.csv')
data_test = pd.read_csv('../input/spaceship-titanic/test.csv')
data_train.head()

# Gather some general info from the data
data_train.info()

# Get the general structure of the data
data_train

# Find the number of values for each column
data_train.count()

# Splitting the cabin column into its three components and create the training data for the model
data_train[['Deck', 'RoomNumber', 'RoomType']] = data_train['Cabin'].str.split('/', expand=True)
X = data_train[['HomePlanet','CryoSleep','Deck', 'RoomNumber', 'RoomType','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Transported']]
y = X.pop('Transported').astype(int)
X

# Split the data into training and validation
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
X_train_full

# Define the what columns are what kind of data for the preprocessing step
numerical_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_cols = ['HomePlanet','CryoSleep','Deck', 'RoomNumber', 'RoomType','Destination','VIP']

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



#Define the variables that are going to be inputed into the Catboost model
MODEL_MAX_DEPTH = 1
MODEL_RL = 0.025
MODEL_ESR = 25
MODEL_VERBOSE = 100
MODEL_ITERATIONS = 100

# Create a defintion of the catboost model to make optimization easier
def catboosting(MODEL_MAX_DEPTH,
                MODEL_RL,
                MODEL_ESR,
                MODEL_VERBOSE,
                MODEL_ITERATIONS):

    return CatBoostClassifier(
        verbose=MODEL_VERBOSE,
        early_stopping_rounds=MODEL_ESR,
        max_depth=MODEL_MAX_DEPTH,
        task_type='GPU',
        learning_rate=MODEL_RL,
        iterations=MODEL_ITERATIONS,
        loss_function='MultiClass',
        eval_metric= 'Accuracy')


# Define the model
model = catboosting(MODEL_MAX_DEPTH,
                MODEL_RL,
                MODEL_ESR,
                MODEL_VERBOSE,
                MODEL_ITERATIONS)

# Looking at X one last time
X

# Create the pipeline to make optimization easier and code more readable
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# train the model
my_pipeline.fit(X_train_full, y_train)

# predict the results of the testing set
preds = my_pipeline.predict(X_valid_full)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

'''
for optimization I would loop through the components below over a range of appropriate values for the componenets below 
and find the value with the Lowest Mean Absolute Error. The optimized components can then be found. For the sake of running 
the script I did not include this that the program wouldn't have a very long run-time. 

    MODEL_MAX_DEPTH = 1
    MODEL_RL = 0.025
    MODEL_ESR = 25
    MODEL_VERBOSE = 100
    MODEL_ITERATIONS = 100
'''

#convert the array from the prediction to a pandas dataframe, convert to an int32 and then make sure that the data values look right
pd.DataFrame(preds).astype('int32').astype(bool).value_counts()

# Submit the test set using the right PassengerID and the submission sample as a guide as a csv file
test_preds = pd.DataFrame(preds).astype('int32').astype(bool)
output = pd.DataFrame({'PassengerId': data_test.PassengerId,
                      'Transported':test_preds.iloc[:, 0]})
output.to_csv('submission.csv', index=False)

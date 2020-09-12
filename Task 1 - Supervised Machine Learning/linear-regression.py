# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os
import pickle

# Reading data
data = pd.read_csv('http://bit.ly/w-data')

X = data.iloc[:, :1].values
y = data.iloc[:, 1].values
 
# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Training the algorithm
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# Saving model to disk
pickle.dump(regressor, open(os.path.join(os.path.dirname(__file__),'.\SupervisedML.pkl'),'wb'))
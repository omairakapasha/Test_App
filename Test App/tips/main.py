#Prediction tip using tip dataset from seaborn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
tips = sns.load_dataset('tips')

# Display the first 5 rows of the dataset
print(tips.head())

# Display the information of the dataset
print(tips.info())

# Display the statistical summary of the dataset
# print(tips.describe())

# Create the feature matrix
X = tips[['total_bill', 'size']]
y = tips['tip']

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make prediction
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

# Export the model
import joblib
joblib.dump(model, 'tip_model.pkl')

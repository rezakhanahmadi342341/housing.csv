import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('final_data1.csv')

# Split the dataset into training and testing sets (80% training, 20% testing)
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target variable for the training set
train_features = train_set.drop(['median_house_value'], axis=1)  # Features (without the target)
train_labels = train_set['median_house_value'].copy()  # Target variable (median house value)

# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(train_features, train_labels)

# Make predictions on the training set
predicted_prices_train = lin_reg.predict(train_features)

# Calculate Mean Squared Error and Root Mean Squared Error for the training set
mse_train = mean_squared_error(train_labels, predicted_prices_train)
rmse_train = np.sqrt(mse_train)

# Output the Root Mean Squared Error for the training set
print("Training Set RMSE:", rmse_train)

# ==================================================================================================

# Evaluate the model on the test set

# Separate features and target variable for the test set
test_features = test_set.drop(['median_house_value'], axis=1)  # Features (without the target)
test_labels = test_set['median_house_value'].copy()  # Target variable (median house value)

# Make predictions on the test set
predicted_prices_test = lin_reg.predict(test_features)

# Calculate Mean Squared Error and Root Mean Squared Error for the test set
mse_test = mean_squared_error(test_labels, predicted_prices_test)
rmse_test = np.sqrt(mse_test)

# Output the Root Mean Squared Error for the test set
print("Test Set RMSE:", rmse_test)

# ==================================================================================================

# Optional: Calculate and output the R-squared score for both the training and test sets

# R-squared score for the training set
r2_train = r2_score(train_labels, predicted_prices_train)
print("Training Set R² Score:", r2_train)

# R-squared score for the test set
r2_test = r2_score(test_labels, predicted_prices_test)
print("Test Set R² Score:", r2_test)

# ==================================================================================================

# Optional: Making predictions with a new dataset
# Ensure the new dataset has the same features as the training set
# Load new data (replace 'new_data' with the actual dataset variable)
# new_data = pd.read_csv('new_data.csv')  # Uncomment this if loading from a file

# Make predictions with new data

predicted_prices_new = lin_reg.predict(new_data)  # Replace 'new_data' with the actual new dataset

# Output predicted prices from the new data
print("Predicted Prices for New Data:", predicted_prices_new)

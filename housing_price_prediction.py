import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('final_data1.csv')

# Split the dataset into training and testing sets (80% training, 20% testing)
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Separate features and target variable for the training set
train_features = train_set.drop(['median_house_value'], axis=1)
train_labels = train_set['median_house_value'].copy()

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor model
rf_reg = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit GridSearchCV on the training data
grid_search.fit(train_features, train_labels)

# Get the best parameters and estimator
best_rf_reg = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ==================================================================================================

# Evaluate the model on the test set

# Separate features and target variable for the test set
test_features = test_set.drop(['median_house_value'], axis=1)
test_labels = test_set['median_house_value'].copy()

# Make predictions on the test set using the best model from GridSearch
predicted_prices_test = best_rf_reg.predict(test_features)

# Calculate Mean Squared Error and Root Mean Squared Error for the test set
mse_test = mean_squared_error(test_labels, predicted_prices_test)
rmse_test = np.sqrt(mse_test)

# Output the Root Mean Squared Error for the test set
print("Test Set RMSE:", rmse_test)

# ==================================================================================================

# R-squared score for the test set
r2_test = r2_score(test_labels, predicted_prices_test)
print("Test Set R² Score:", r2_test)

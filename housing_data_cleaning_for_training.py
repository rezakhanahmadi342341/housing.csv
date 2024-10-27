import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the CSV file
housing = pd.read_csv('housing.csv')

# Split the data into train and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Create a copy of the test set for exploration
data = test_set.copy()

# Display info about the dataset
data.info()

# ================================================================================================================================

# Plot longitude vs. latitude with population size and house value
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, 
          s=data['population'] / 100, label='population', figsize=(10, 7), 
          c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()

# ================================================================================================================================

# Convert 'ocean_proximity' to numeric (will result in NaN for non-numeric values)
data['ocean_proximity'] = pd.to_numeric(data['ocean_proximity'], errors='coerce')

# Calculate correlation matrix and show correlation with 'median_house_value'
corr_matrix = data.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# ================================================================================================================================

# Plot scatter matrix for selected features
from pandas.plotting import scatter_matrix

features = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(data[features], figsize=(12, 8))

# ================================================================================================================================

# Plot the relationship between 'median_house_value' and 'median_income'
data.plot(kind='scatter', y='median_house_value', x='median_income', alpha=0.4, 
          s=data['population'] / 100, label='population', figsize=(10, 7), 
          c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()

# ================================================================================================================================

# Create new columns for better feature representation
data['total_rooms_per_households'] = data['total_rooms'] / data['households']
data['total_bedrooms_per_total_rooms'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_households'] = data['population'] / data['households']

# Show the first few rows with the new features
data.head()

# ================================================================================================================================

# Create a copy of the training set for data preparation
df = train_set.copy()

# Remove the categorical column 'ocean_proximity'
df_num = df.drop("ocean_proximity", axis=1)

# ================================================================================================================================

from sklearn.impute import SimpleImputer
import numpy as np

# Use SimpleImputer to fill missing values with the median
imputer = SimpleImputer(strategy='median')
imputer.fit(df_num)

# Transform the data using the fitted imputer
df_num_imputer_tr = pd.DataFrame(imputer.transform(df_num), columns=df_num.columns)

# Display info of the transformed and original datasets
df_num_imputer_tr.info()
df_num.info()

# ================================================================================================================================

# Custom transformer to add new combined attributes
from sklearn.base import BaseEstimator, TransformerMixin

# Indexes for the relevant columns
rooms_ix, bedrooms_ix, populations_ix, households_ix = 3, 4, 5, 6

class combinerAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        total_rooms_per_households = x[:, rooms_ix] / x[:, households_ix]
        total_bedrooms_per_total_rooms = x[:, bedrooms_ix] / x[:, rooms_ix]
        population_per_households = x[:, populations_ix] / x[:, households_ix]
        return np.c_[x, total_rooms_per_households, total_bedrooms_per_total_rooms, population_per_households]

# Apply the custom transformer
custom = combinerAttributesAdder()
data_custom_tr_tmp = custom.transform(df_num_imputer_tr.values)

# Convert the transformed data back into a DataFrame
data_custom_tr = pd.DataFrame(data_custom_tr_tmp)

# Update column names
columns = list(df_num_imputer_tr.columns)
columns += ["total_rooms_per_households", "total_bedrooms_per_total_rooms", "population_per_households"]
data_custom_tr.columns = columns

# Show the first 10 rows of the transformed data
data_custom_tr.head(10)

# ================================================================================================================================

# Scale the features using StandardScaler
from sklearn.preprocessing import StandardScaler

feature_scaler = StandardScaler()
data_num_scaled_tr = pd.DataFrame(feature_scaler.fit_transform(data_custom_tr.values), columns=data_custom_tr.columns)

# Show the first 10 rows of the scaled data
data_num_scaled_tr.head(10)

# ================================================================================================================================

# One-hot encode the 'ocean_proximity' categorical feature
from sklearn.preprocessing import OneHotEncoder

encoder_1hot = OneHotEncoder(sparse_output=False)

# Fit and transform the 'ocean_proximity' column
data_cat_1hot_tmp = encoder_1hot.fit_transform(df[["ocean_proximity"]])

# Convert to DataFrame with appropriate column names
data_cat_1hot = pd.DataFrame(data_cat_1hot_tmp, columns=encoder_1hot.get_feature_names_out(['ocean_proximity']))

# Concatenate numerical and one-hot encoded categorical data
final = pd.concat([data_num_scaled_tr, data_cat_1hot], axis=1)

# Display the first 10 rows of the final dataset
final.head(10)

# ================================================================================================================================

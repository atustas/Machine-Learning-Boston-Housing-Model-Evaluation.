import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# load the boston housing dataset
boston = load_boston()
boston_data = boston.data
boston_target = boston.target

# create a dataframe from data
boston_df = pd.DataFrame(data= np.c_[boston['data'], boston['target']],
                     columns= np.append(boston['feature_names'], ['target']))

# get summary statistics
print(boston_df.describe())

# create a scatter plot of target variable vs. number of rooms
plt.scatter(boston_df['RM'], boston_df['target'])
plt.xlabel('Number of rooms')
plt.ylabel('House price')
plt.show()

# preprocess the data
scaler = StandardScaler()
boston_data_scaled = scaler.fit_transform(boston_data)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(boston_data_scaled, boston_target, test_size=0.3)

# train and evaluate different regression models
models = {'Linear Regression': LinearRegression(),
          'Decision Tree Regression': DecisionTreeRegressor(),
          'Random Forest Regression': RandomForestRegressor()}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores)
    r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(name)
    print('RMSE:', rmse.mean())
    print('R^2:', r2.mean())
    
# use the best-performing model to make predictions on new data
best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Best Model')
print('MSE:', mse)
print('R^2:', r2)

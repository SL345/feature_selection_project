# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(data):
    np.random.seed(9)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    rf_selector = SelectFromModel(RandomForestClassifier())
    rf_selector.fit(X, y)
    rf_support = rf_selector.get_support()
    selected_variables = X.loc[:,rf_support].columns.tolist()
    return selected_variables


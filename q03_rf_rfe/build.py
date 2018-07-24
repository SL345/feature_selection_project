# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
from collections import OrderedDict

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    names = data.columns
    rfc = RandomForestClassifier()
    i = int(len(X.columns)/2)
    rfe = RFE(rfc,n_features_to_select= i  ,step=1)
    rfe.fit(X,y)
    d = OrderedDict(zip(names,rfe.ranking_))
    top_features = []
    for k,v in d.items():
        if v == 1:
            top_features.append(k)
    return top_features



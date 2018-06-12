# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    selector = SelectPercentile(score_func=f_regression, percentile=20)
    x_new = selector.fit_transform(X,y)
    dct = dict(zip(selector.scores_[selector.get_support(True)],selector.get_support(True))) 
    indices = [dct.get(num) for num in np.sort(selector.scores_[selector.get_support(True)])[::-1]]
    lst = list(X.columns[indices])
    return lst



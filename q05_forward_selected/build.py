# %load q05_forward_selected/build.py
# Default imports
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import OrderedDict

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

# Your solution code here
def forward_selected(data,model):
    X = data.iloc[:,:-1]
    y = data['SalePrice']
    model = LinearRegression()
    lst = []
    r_squared_list = []
    dct = {}
    column_list = []
    colname_list = []
    X1 = pd.DataFrame({'A' : [np.nan]})
    X2 = pd.DataFrame({'A' : [np.nan]})
    for j in range(len(X.columns)):
        for i in range(len(X.columns)):
            if not X1.empty:
                X1 = pd.concat([X1,X.iloc[:,i]],axis=1)
                if 'A' in X1:
                    del X1['A']
                if 'A' in X2:
                    del X2['A']
            model.fit(X1,y)
            y_pred = model.predict(X1)
            index = r2_score(y, y_pred)
            lst.append(index)
            dct[index] = X.iloc[:,i]
            if j== 0:
                X1 = pd.DataFrame({'A' : [np.nan]})
            else:
                X1=X2
        r_squared = np.sort(lst)[::-1][0]
        if r_squared_list:
            if r_squared > r_squared_list[-1] :
                r_squared_list.append(r_squared)
                col = dct.get(r_squared)
                colname_list.append(col.name)
                column_list.append(col)
                X2 = pd.DataFrame(OrderedDict(zip(colname_list,column_list)))
                del X[col.name]
        else:
            r_squared_list.append(r_squared)
            col = dct.get(r_squared)
            colname_list.append(col.name)
            column_list.append(col)
            X2 = pd.DataFrame(OrderedDict(zip(colname_list,column_list)))
            del X[col.name]
        lst = []
        dct={}
#    r_squared_array = ['{:.16f}'.format(num) for num in r_squared_list]
    return list(X2.columns), list(r_squared_list)


# --------------
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
## Load the data
df = pd.read_csv(path)

## Split the data and preprocess
train = df[df['source']=='train']

test = df[df['source']=='test']

## Baseline regression model
X=train[['Item_Weight','Item_MRP','Item_Visibility']]

Y= train['Item_Outlet_Sales']

reg = LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3, random_state = 6)

reg.fit(X_train,y_train)
print(reg.coef_)
y_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_pred,y_test))

r2 = r2_score(y_test,y_pred)
## Effect on R-square if you increase the number of predictors
X1 = train.drop(columns=['Item_Outlet_Sales','Item_Identifier','source'])

X1_train,X1_test,y_train,y_test = train_test_split(X1,Y,test_size = 0.3, random_state = 6)

reg.fit(X1_train,y_train)

y_pred = reg.predict(X1_test)

rmse1 = np.sqrt(mean_squared_error(y_pred,y_test))

r2_1 = r2_score(y_test,y_pred)
print(reg.coef_)
print(r2)
print(r2_1)
## Effect of decreasing feature from the previous model
X2 = train.drop(columns=['Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years','source'])

X2_train,X2_test,y_train,y_test = train_test_split(X2,Y,test_size = 0.2, random_state = 42)

reg.fit(X2_train,y_train)
print(reg.coef_)
y_pred = reg.predict(X2_test)

rmse2 = np.sqrt(mean_squared_error(y_pred,y_test))

r2_2 = r2_score(y_test,y_pred)

print(r2_2)

## Detecting hetroskedacity
plt.scatter(y_pred,(y_pred-y_test))

## Model coefficients
print(pd.DataFrame(X2.columns,reg.coef_))

## Ridge regression
ridge_model = Ridge()
ridge_model.fit(X2_train,y_train)

ridge_pred = ridge_model.predict(X2_test)

rmse_ridge = np.sqrt(mean_squared_error(ridge_pred,y_test))

r2_ridge = r2_score(y_test,ridge_pred)

print(r2_ridge)

## Lasso regression
lasso_model = Lasso()
lasso_model.fit(X2_train,y_train)

lasso_pred = lasso_model.predict(X2_test)

rmse_lasso = np.sqrt(mean_squared_error(lasso_pred,y_test))

r2_lasso = r2_score(y_test,lasso_pred)

print(r2_lasso)


## Cross vallidation
rmse_L1=-np.mean(cross_val_score(lasso_model, X2_train, y_train,cv=10))
print(rmse_L1)
# cross validation with L2
rmse_L2=-np.mean(cross_val_score(ridge_model, X2_train, y_train,cv=10))
print(rmse_L2)



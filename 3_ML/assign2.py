'''
Author : Saqeeb
Course : Applied Machine Learning

'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10
#plt.plot(x,y)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# ANSWER 1
# poly regression 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# 1
'''
# Poly
poly1 = PolynomialFeatures(degree=1)
polytrain1 = poly1.fit_transform(X_train.reshape(-1,1))

linreg1 = LinearRegression().fit(polytrain1 , y_train)

x_pred1 = poly1.transform(x_pred)
y_pred1 = linreg1.predict(x_pred1)
final_array = np.array(y_pred1, ndmin=2)

'''

# 6 9 
x_pred = np.linspace(0,10,100).reshape(-1,1)
final_array = np.empty(shape= (4,100))
for index,value in enumerate([1,3,6,9]):
    poly = PolynomialFeatures(degree = value)
    poly_train = poly.fit_transform(X_train.reshape(-1,1))
    linreg = LinearRegression().fit(poly_train , y_train)
    x_pred_poly = poly.transform(x_pred)
    y_pred = linreg.predict(x_pred_poly)
    final_array[index] = y_pred

plt.figure()
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.plot(x_pred , final_array[0], lw = 3)
plt.plot(x_pred , final_array[1],lw=3)
plt.plot(x_pred , final_array[2],lw=3)
plt.plot(x_pred , final_array[3],lw=3)
plt.legend(['a','b','c','d','e'])


# ANSWER 2

from sklearn.metrics.regression import r2_score
r2_test = np.empty(10)
r2_train = np.empty(10)
for i in range(0,10):
    poly = PolynomialFeatures(degree=i)
    poly_train = poly.fit_transform(X_train.reshape(-1,1))
    poly_test = poly.transform(X_test.reshape(-1,1))
    linreg = LinearRegression().fit(poly_train,y_train)
    r2_train[i] = linreg.score(poly_train,y_train)
    r2_test[i] = linreg.score(poly_test, y_test)


# ANSWER 3

plt.plot(range(0,10),r2_train,'r',
         range(0,10),r2_test,'b')

# ANSWER 4


from sklearn.linear_model import Lasso
poly = PolynomialFeatures(degree=12)
poly_train = poly.fit_transform(X_train.reshape(-1,1))
poly_test = poly.transform(X_test.reshape(-1,1))

linreg = LinearRegression().fit(poly_train,y_train)
lasso = Lasso(alpha=0.01, max_iter=10000).fit(poly_train, y_train)


linreg.score(poly_test,y_test)
lasso.score(poly_test,y_test)


# ANSWER 5

from sklearn.tree import DecisionTreeRegressor

dtc = DecisionTreeRegressor(random_state=0).fit(poly_train,y_train)

z = dtc.feature_importances_
z = pd.DataFrame(z, index = range(1,11),)
z = z.sort_values(0,ascending=False)
list(z.index[:5])

from adspy_shared_utilities import plot_feature_importances

plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf, iris.feature_names)
plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))

#ANSWER 6

from sklearn.svm import SVR
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
#X_subset and y_subset
X_train, X_test, y_train, y_test = train_test_split(X_subset,y_subset)

svc = SVC(kernel='rbf',C = 1, random_state=0)

svr = SVR(kernel='rbf',C=1)
z = validation_curve(svr,poly_train,y_train,'gamma', np.logspace(-4,1,6),scoring='accuracy' )
z[0] = z[0].mean(axis=1)
z
z = tuple([z[i].mean(axis=1) for i in [0,1]])




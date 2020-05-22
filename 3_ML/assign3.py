# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:22:37 2020

@author: sony
"""

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('readonly/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def answer_one():
    df = pd.read_csv('fraud_data.csv')
    prec = df['Class'].value_counts()[1]/len(df)
    return prec

'''
ANSWER 2
'''

from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

dummy_most_freq = DummyClassifier(strategy ='most_frequent').fit(X_train,y_train)
accu_dummy = dummy_most_freq.score(X_test,y_test)
pred_dummy = dummy_most_freq.predict(X_test)
recall_dummy = recall_score(y_test,pred_dummy)


'''
ANSWER 3 
'''
from sklearn.svm import SVC
from sklearn.metrics import recall_score,precision_score
svc = SVC().fit(X_train,y_train)
accu_svc = svc.score(X_test,y_test)
pred_svc = svc.predict(X_test)
recall_svc = recall_score(y_test,pred_svc)
precision_svc = precision_score(y_test, pred_svc)


'''
ANSWER 4 
'''
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
parameter = {'C': 1e9, 'gamma': 1e-07}
svc = SVC(C = 1e9,gamma=1e-07).fit(X_train,y_train)
df_svc = svc.decision_function(X_test) > -220
confusion_matrix(y_test, df_svc)


'''
ANSWER 5
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,roc_curve
import matplotlib.pyplot as plt
import numpy as np

logreg = LogisticRegression().fit(X_train,y_train)

precision , recall, threshold = precision_recall_curve(y_test, logreg.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
fig,ax = plt.subplots()
ax.plot(precision[close_zero], recall[close_zero],'o',markersize = 10,label='threshold zero',
         fillstyle="none",c='k',mew=2)
ax.plot(precision,recall, label='precision recall curve')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')

fpr,tpr,threshold = roc_curve(y_test,logreg.decision_function(X_test))
close_zero = np.argmin(np.abs(threshold))
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr[close_zero],tpr[close_zero],'o',markersize = 10, label='threshold', fillstyle='none', c='k',mew=2)
plt.legend(loc=4)


'''
ANSWER 6
'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'penalty': ['l1', 'l2'],'C':[0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(LogisticRegression(),param_grid = param_grid , scoring = 'recall',cv=3).fit(X_train,y_train)

z = grid.cv_results_
grid.cv_results_['mean_test_score'].reshape(5,2)

'''
ANSWER 7
'''


























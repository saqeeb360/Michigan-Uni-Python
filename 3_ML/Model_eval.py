# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:41:30 2020

@author: sony
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1234)
X1 = np.random.normal(50,40,size=450).astype('int')
X2 = np.random.normal(100,50,size=450).astype('int')
y = (X1>50) & (X2 > 100)

df = pd.DataFrame({0:X1,1:X2,2:y})

#X = pd.DataFrame(np.hstack((X1.reshape(-1,1),X2.reshape(-1,1))))

plt.scatter(df[0],df[1], c = df[2])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[[0,1]],df[2],random_state=0)

sum(y_train==True)
sum(y_train==False)

'''
Dummy Classifier : most freq 
'''

from sklearn.dummy import DummyClassifier

dummycls = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
pred_most_freq = dummycls.predict(X_test)
np.unique(pred_most_freq)
dummycls.score(X_test,y_test)

'''
Dummy Classifier random
'''
dummy_rand = DummyClassifier().fit(X_train,y_train)
pred_rand = dummycls.predict(X_test)
np.unique(pred_rand)
dummy_rand.score(X_test,y_test)

'''
Tree
'''

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
tree.score(X_test,y_test)
pred_tree = tree.predict(X_test)
'''
Logistics Regression
'''

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=12).fit(X_train,y_train)
logreg.score(X_train,y_train)
logreg.score(X_test,y_test)
pred_log = logreg.predict(X_test)
'''
Confusion matrix
'''
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred_most_freq)
confusion_matrix(y_test, pred_rand)
confusion_matrix(y_test,pred_tree)
confusion_matrix(y_test,pred_log)

'''
F1 score : harmonic mean of precision and recall
'''

from sklearn.metrics import f1_score

f1_score(y_test,pred_most_freq)
f1_score(y_test, pred_rand)
f1_score(y_test,pred_tree)
f1_score(y_test,pred_log)

'''
classification report
'''

from sklearn.metrics import classification_report

pd.DataFrame(classification_report(y_test,pred_most_freq,output_dict=True))
pd.DataFrame(classification_report(y_test, pred_rand,output_dict=True))
pd.DataFrame(classification_report(y_test,pred_tree,output_dict=True))
pd.DataFrame(classification_report(y_test,pred_log,output_dict=True))

classification_report(y_test,pred_log,output_dict=True)
classification_report(y_test,pred_log, target_names=["False","True"], output_dict=True)


'''
SVC
'''

from sklearn.svm import SVC

svc = SVC(gamma=0.002).fit(X_train,y_train)
pred_svc = svc.predict(X_test)
svc.score(X_test,y_test)
confusion_matrix(y_test,pred_svc)
pd.DataFrame(classification_report(y_test,pred_svc,output_dict=True))

'''
Decision Function on SVC
'''
pred_threshold_svc = svc.decision_function(X_test)> -0.9
confusion_matrix(y_test,pred_threshold_svc)
pd.DataFrame(classification_report(y_test,pred_threshold_svc,output_dict=True))

'''
Precisoion Recall Curve
'''

from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test,svc.decision_function(X_test))

close_zero = np.argmin(np.abs(threshold))
fig,ax = plt.subplots()
ax.plot(precision[close_zero], recall[close_zero],'o',markersize = 10,label='threshold zero',
         fillstyle="none",c='k',mew=2)
ax.plot(precision,recall, label='precision recall curve')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
#ax.set_aspect(1)
#ax.axis([0,1,0,1])

'''
ROC - Receiver operating characteristic
'''

from sklearn.metrics import roc_curve

fpr , tpr, threshold = roc_curve(y_test, logreg.decision_function(X_test),drop_intermediate=False )

plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')

close_zero = np.argmin(np.abs(threshold))
plt.plot(fpr[close_zero],tpr[close_zero],'o',markersize = 10, label='threshold', fillstyle='none', c='k',mew=2)
plt.legend(loc=4)

'''
AUC of ROC
'''

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, logreg.decision_function(X_test))
roc_auc_score(y_test, svc.decision_function(X_test))
deci_func_log = logreg.decision_function(X_test) 
'''
different value of gamma in svc
'''

plt.figure()
for gamma in [0.001,0.01,0.1,1,10]:
    svc = SVC(gamma= gamma).fit(X_train,y_train)
    accuracy = svc.score(X_test,y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test), drop_intermediate=False)
    print("gamma = {:.2f} accuracy = {:.2f} AUC = {:.2f}".format(gamma,accuracy,auc))
    plt.plot(fpr,tpr, label= "gamma={:.3f}".format(gamma))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.01,1)
plt.ylim(0, 1.02)
plt.legend(loc='best')


'''
Applying accuracy and auc - roc on whole df
'''



plt.figure()
for gamma in [0.001,0.01,0.1,1,10]:
    svc = SVC(gamma= gamma).fit(X_train,y_train)
    accuracy = svc.score(df.iloc[:,[0,1]],df.iloc[:,2])
    auc = roc_auc_score(df.iloc[:,2], svc.decision_function(df.iloc[:,[0,1]]))
    fpr, tpr, _ = roc_curve(df.iloc[:,2], svc.decision_function(df.iloc[:,[0,1]]), drop_intermediate=False)
    print("gamma = {:.3f} accuracy = {:.2f} AUC = {:.2f}".format(gamma,accuracy,auc))
    plt.plot(fpr,tpr, label= "gamma={:.3f}".format(gamma))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc='best')
plt.xlim(-0.01,1)
plt.ylim(0, 1.02)




'''
Cross validation and Gridsearch on model
'''

from sklearn.model_selection import cross_val_score
cross_val_score(SVC(),X_train,y_train)

from sklearn.model_selection import GridSearchCV
param_grid = {'gamma':[0.0001,0.001,0.01,0.1,1]}
grid = GridSearchCV(SVC(),param_grid= param_grid).fit(X_train,y_train)
grid.best_estimator_
grid.best_params_
grid.best_score_

grid2 = GridSearchCV(SVC(),param_grid = param_grid, scoring = 'roc_auc').fit(X_train,y_train)
grid2


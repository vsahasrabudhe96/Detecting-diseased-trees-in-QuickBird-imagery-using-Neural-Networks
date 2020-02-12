# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:13:23 2020

@author: vsaha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df_train = pd.read_csv('D:/Downloads/wilt/training.csv')
df_test = pd.read_csv('D:/Downloads/wilt/testing.csv')
df_train['class'] = df_train['class'].str.contains('w').astype(int)
df_test['class'] = df_test['class'].str.contains('w').astype(int)
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class = 'ovr',random_state= 0)))
models.append(('CART',RandomForestClassifier(n_estimators = 100,random_state= 0)))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NaiveBayes',GaussianNB()))
models.append(('SVM',SVC(gamma='auto',random_state= 0)))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#using RandomForestClassifier
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train,y_train)


# STarting with ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

'''
classifier = Sequential()
#Starting with the first layer and hidden layer
classifier.add(Dense(units = 3,kernel_initializer = 'uniform',activation = 'relu',input_dim = 5))

classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
#classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))


classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
#Compliling the NN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the data to the ANN
classifier.fit(X_train,y_train,batch_size = 32,epochs = 100)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
'''

def build_classifier():
    classifier = Sequential()
#Starting with the first layer and hidden layer
    classifier.add(Dense(units = 3,kernel_initializer = 'uniform',activation = 'relu',input_dim = 5))
    
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
    #classifier.add(Dense(units = 3, kernel_initializer = 'uniform',activation = 'relu'))
    
    
    classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
    #Compliling the NN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

    
#classifier = KerasClassifier(build_fn = build_classifier,batch_size = 32,epochs = 100)
accuracies = cross_val_score(classifier,X = X_train,y = y_train,cv = 10,n_jobs = -1)

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv = 10)
grid_search = grid_search.fit(X=X_train,y=y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

y_pred = classifier.predict(X_test)
y_mean = y_pred.mean()
y_pred = (y_pred>y_mean)


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
from sklearn.metrics import log_loss
loss = log_loss(y_test,y_pred)

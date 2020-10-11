#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('D_Train1.csv')
train_data =train_data.values

X = train_data[:,1:]
y = train_data[:,0]

test_data = pd.read_csv('D_Test1.csv')
test_data = test_data.values

X_test = test_data[:,1:]
y_test = test_data[:,0]


PCA
pca = PCA(n_components=2)
# pca = pca.fit(X)
# print(pca.explained_variance_ratio_)
X = pca.fit_transform(X)
X_test = pca.transform(X_test)

#Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


#Normalize(Better)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)


def VisualizeResult(X_test,y_test,classifier,title):
    x_set, y_set = X_test, y_test
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01), np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
    plt.figure()
    
    y = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    
    plt.contourf(x1, x2, y, alpha = 0.75, cmap = ListedColormap(('red', 'green','purple','black')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('orange', 'blue','red','yellow'))(i), label = j)
    
    plt.title(title)
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.legend()
    plt.show()




#NB #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    val_acc = clf.score(X_val,y_val)
    table.append(val_acc)


y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Naive Bayes:",round(100*acc,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm,"\n")

# VisualizeResult(X_test, y_test, clf,'Naive Bayes(Testing set)')




#SVM #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf2 = SVC(C=100)
    clf2.fit(X_train,y_train)
    val_acc = clf2.score(X_val,y_val)
    table.append(val_acc)

y_pred2 = clf2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
cm2 = confusion_matrix(y_test, y_pred2)
print("SVM:",round(100*acc2,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm2,"\n")

# VisualizeResult(X_test, y_test, clf2,'SVM(Testing set)' )



#Perceptron #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf3 = Perceptron()
    clf3.fit(X_train,y_train)
    val_acc = clf3.score(X_val,y_val)
    table.append(val_acc)

y_pred3 = clf3.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
cm3 = confusion_matrix(y_test, y_pred3)
print("Perceptron:",round(100*acc3,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm3,"\n")

# VisualizeResult(X_test, y_test, clf3,'Perceptron(Testing set)' )





#OVR #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf4 = OneVsRestClassifier(SVC())
    clf4.fit(X_train,y_train)
    val_acc = clf4.score(X_val,y_val)
    table.append(val_acc)

y_pred4 = clf4.predict(X_test)
acc4 = accuracy_score(y_test, y_pred4)
cm4 = confusion_matrix(y_test, y_pred4)
print("OVR:",round(100*acc4,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm4,"\n")


# VisualizeResult(X_test, y_test, clf4,'OneVsRest(Testing set)')






#KNN #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf5 = KNeighborsClassifier()
    clf5.fit(X_train,y_train)
    val_acc = clf5.score(X_val,y_val)
    table.append(val_acc)

y_pred5 = clf5.predict(X_test)
acc5 = accuracy_score(y_test, y_pred5)
cm5 = confusion_matrix(y_test, y_pred5)
print("KNN:",round(100*acc5,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm5,"\n")

# VisualizeResult(X_test, y_test, clf5,'KNN(Testing set)')






#DecisionTree #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf6 = DecisionTreeClassifier()
    clf6.fit(X_train,y_train)
    val_acc = clf6.score(X_val,y_val)
    table.append(val_acc)

y_pred6 = clf6.predict(X_test)
acc6 = accuracy_score(y_test, y_pred6)
cm6 = confusion_matrix(y_test, y_pred6)
print("DecisionTree:",round(100*acc6,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm6,"\n")

# VisualizeResult(X_test, y_test, clf6, 'Decision Tree(Testing set)')





#RandomForest #########################################################################
skf = StratifiedKFold(shuffle=True)
table =[]
for train_index, val_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf7 = RandomForestClassifier(max_depth=50)
    clf7.fit(X_train,y_train)
    val_acc = clf7.score(X_val,y_val)
    table.append(val_acc)

y_pred7 = clf7.predict(X_test)
acc7 = accuracy_score(y_test, y_pred7)
cm7 = confusion_matrix(y_test, y_pred7)
print("RandomForest:",round(100*acc7,2),"%")
print("cross_val_acc mean:",round(np.mean(table),3))
print("cross_val_acc std:",round(np.std(table),3))
print(cm7,"\n")

# VisualizeResult(X_test, y_test, clf7,'Random Forest(Testing set)')

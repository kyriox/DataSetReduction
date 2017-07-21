#!/usr/bin/env  python
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as accs
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import StratifiedShuffleSplit as kfold
from sklearn.metrics import roc_curve, auc
import math



# clfs={#"kNN k=3":[KNeighborsClassifier(n_neighbors=3),0,0,0,0],
#     #"KNN 5":[KNeighborsClassifier(n_neighbors=5),0,0,0,0],
#     "NN (kNN k=1)": [KNeighborsClassifier(n_neighbors=1),0,0,0,0],
#     "SVM - Linear kernel": [svm.SVC(kernel="linear",probability=True),0,0,0,0],
#     #"SVM Exponencial Cuadratico": [svm.SVC(kernel="rbf"),0,0,0,0],
#     #"SVM Polinimial": [svm.SVC(kernel="poly"),0,0,0,0],
#     #"SVM Sigmoide": [svm.SVC(kernel="sigmoid",probability=True),0,0,0,0],
#     "Naive Bayes": [GaussianNB(),0,0,0,0],
#     #"QDA":[ QuadraticDiscriminantAnalysis(),0,0,0,0],
#     #"Ada":[AdaBoostClassifier(),0,0,0,0],
#     #"RFC":[RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),0,0,0,0],
#     "ANN":[MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),0,0,0,0],
#}

def load(fname):
    data=np.loadtxt(fname,delimiter=",")
    X=data[:,:-1]
    y=data[:,-1]
    return X,y



#rcParams['font.sans-serif'] = ['Tahoma']


def test_classifiers(X,y,n=5,rname="results.txt"):        
    clfs={
        "NN (kNN k=1)": [KNeighborsClassifier(n_neighbors=1),[],[],[],[]],
        #"SVM - Linear kernel": [svm.SVC(kernel="linear",probability=True),[],[],[],[]],
        "Naive Bayes": [GaussianNB(),[],[],[],[]],
        #"SVM Exponencial Cuadratico": [svm.SVC(kernel="rbf"),[],[],[],[]],
        #"ANN":[MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),[],[],[],[]],
    }
    skf=kfold(y, n_iter=n, random_state=None,  train_size=0.7)
    output=open(rname,"w")
    for train,test in skf:
        Xt,Yt=X[train],y[train]
        Xv,Yv=X[test],y[test]
        for (k,v)  in  clfs.items():
            v[0].fit(Xt,Yt)
            #print(clfs[k])
            Yr=v[0].predict(Xv)
            #print(accs(Yv,Yr))
            v[1].append(accs(Yv,Yr))
            v[2].append(f1(Yv,Yr,average="macro"))
            v[3].append(recall(Yv,Yr,average="macro"))
            v[4].append(f1(Yv,Yr,average="micro"))
    for k,v in clfs.items():
        fm="%s | %s| %s | %s | %s\n"
        output.write(fm %(k,"Accuracy",np.mean(v[1]),min(v[1]),max(v[1])))
        output.write(fm  %(k,"F1",np.mean(v[2]),min(v[2]),max(v[2])))
        output.write(fm %(k,"Recall",np.mean(v[3]),min(v[3]),max(v[3])))

import sys
if __name__ == "__main__":
    fname=sys.argv[1]
    rname=sys.argv[2]
    X,y=load(fname)
    test_classifiers(X,y,5,rname)


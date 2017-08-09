#!/usr/bin/env  python
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as accs
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as recall
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import StratifiedShuffleSplit as kfold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import BaggingClassifier
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
    #X=data[:,:-1]
    #y=data[:,-1]
    return data



#rcParams['font.sans-serif'] = ['Tahoma']

#def predict(l):

def test_classifiers(X,y,n=7,rname="results.txt"):        
    clfs={
#        "Bagging KNN": [BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5),[],[],[],[]],
        "NN (kNN k=1)": [KNeighborsClassifier(n_neighbors=1),[],[],[],[],[]],
        "NN (kNN k=3)": [KNeighborsClassifier(n_neighbors=3),[],[],[],[],[]],
        "NN (kNN k=3 w)": [KNeighborsClassifier(n_neighbors=3, weights='distance'),[],[],[],[],[]],
        "NN (kNN k=5 w)": [KNeighborsClassifier(n_neighbors=5, weights='distance'),[],[],[],[],[]],
        #"NN (kNN k=7 w)": [KNeighborsClassifier(n_neighbors=7, weights='distance'),[],[],[],[]],
        #"SVM - Linear kernel": [svm.SVC(kernel="rbf",probability=True),[],[],[],[]],
 #       "Naive Bayes": [GaussianNB(),[],[],[],[]],
#        "SVM Sigmoide": [svm.SVC(kernel="sigmoid"),[],[],[],[]],
        #"ANN":[MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1),[],[],[],[]],
    }
    V=["Voting KNN",[None,[],[],[],[]]]
    skf=kfold(y, n_iter=n, random_state=None,  train_size=0.7)
    output=open(rname,"w")
    for train,test in skf:
        Xt,Yt=X[train],y[train]
        Xv,Yv=X[test],y[test]
        votes=[]
        for (k,v)  in  clfs.items():
            v[0].fit(Xt,Yt)
            #print(clfs[k])
            Yr=v[0].predict(Xv)
            #print(accs(Yv,Yr))
            v[1].append(accs(Yv,Yr))
            v[2].append(f1(Yv,Yr,average="macro"))
            v[3].append(recall(Yv,Yr,average="macro"))
            v[4].append(f1(Yv,Yr,average="micro"))
            v[5].append(kappa(Yv,Yr))
            #votes.append(Yr)
        #Yp=predict(votes)
    for k,v in clfs.items():
        fm="%s | %s| %s | %s | %s\n"
        output.write(fm %(k,"Accuracy",np.mean(v[1]),min(v[1]),max(v[1])))
        output.write(fm  %(k,"Kappa",np.mean(v[5]),min(v[5]),max(v[5])))
        #output.write(fm %(k,"Recall",np.mean(v[3]),min(v[3]),max(v[3])))


def make_precictions(X,y,T,yT=[],rname="predictions.txt"):        
    output=open(rname,"w")
    #sv=svm.SVC(kernel="linear",probability=True)
    #sv=KNeighborsClassifier(n_neighbors=1)
    sv=KNeighborsClassifier(n_neighbors=3, weights='distance')
    sv.fit(X,y)
    Yr=sv.predict(T)
    #print(len(Yr), Yr[0],Yr[4999])
    for y in Yr:
        print(int(y))
    #output.write("\n".join([str(int(y))  for y in Yr]))

import sys
if __name__ == "__main__":
    fname=sys.argv[1]
    tname=sys.argv[2]
    testing=sys.argv[3]
    #rname=sys.argv[4]
    data=load(fname)
    #X=data[:,:-1]
    X=data[:20000,:]
    y=load("labels.csv")
    #y=data[:,-1]    
    T=load(tname)
    #print(len(T))
    #print(len(X[1]))
    #print(len(T[1]))
    make_precictions(X,y,T)
    test_classifiers(X,y,7,testing)


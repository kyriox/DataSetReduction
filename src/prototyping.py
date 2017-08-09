#!/usr/bin/env python
import json
from math import sqrt
from SparseArray import SparseArray as sp
from  heapq import heappop, heappush
import numpy as np
from multiprocessing import Pool
import math
import itertools

def get_data(source):
    text=open(source).readlines()
    R=[json.loads(x) for x in text]
    del text
    X=[sp.index_data(r['vec'],r['vecsize']) for r in R]
    return R,X

def cosine_similarity(a,b):
    return a.dot(b)/(sqrt(a.mul(a).sum())*sqrt(b.mul(b).sum()))

def cosine_distance(a,b):
    return 1-a.dot(b)/(sqrt(a.mul(a).sum())*sqrt(b.mul(b).sum()))

def load(source):
    R,X=get_data(source)
    if 'klass' in R[0]:
        Y=[r['klass'] for r in R]
    else: Y=[]
    d=R[0]['vecsize']
    del R
    return X,Y,d


def centroid(X):
    N=len(X)
    t=X[0]
    for i in range(1,N):
        t=t.add(X[i])
    return t.mul2(1/N)


def max_cord(X):
    N=len(X)
    t=X[0]
    for i in range(1,N):
        t=t.max(X[i])
    return t

def center_point(X):
    N=len(X)
    R=[0 for e in range(N)]
    for i in range(N):
        r=0
        for j in range(N):
           r+=cosine_similarity(X[i],X[j])
        R[i]=(r,i,X[i])
    return max(R)[-1]

center_function={'centroid':centroid,
                  'center_point':center_point,
                 'max_cord':max_cord}


def rocchio(X,Y,c_func='centroid'):
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    Xr,Yr=[],[]
    for ind in indices:
        Xi=[X[i] for i in ind]
        Xr.append(center_function[c_func](Xi))
        Yr.append(Y[ind[0]])
    return Xr,Yr

def get_class(nn):
    klasses = [ n[1] for  n in nn]
    r=[(klasses.count(k),k) for k in klasses]
    r.sort(reverse=True)
    return r[0][1]

def KNN(C,Y,k,source,dest):
    R,T=get_data(source)
    S=[]
    for i in range(len(T)):
        q=T[i]
        f=[(cosine_similarity(q,c), y) for c,y in zip(C,Y)]
        f.sort(reverse=True)
        #print(f[:3])
        neighbors=f[:k]
        #print(neighbors)
        S.append({})
        S[-1]['text']=R[i]['text']
        if k==1:
            S[-1]['klass']=neighbors[0][1]
        else:
            S[-1]['klass']=get_class(neighbors)
        S[-1]['decision_function']=[n[0] for n in neighbors]
        #S[-1]['voc_affinity']=R[i]['voc_affinity']
    #with open(dest, 'w') as outfile:
    #        json.dump(json.dumps(R), outfile)
    open(dest,'w').write("%s" % "\n ".join(json.dumps(r) for r in S))

def homogeneous(partitions,Y,hg):
    G=set(partitions)
    r={}
    for g in G:
        if g not in hg:
            indices=[i for i,x in enumerate(partitions) if x==g]
            #print("indices:",indices)
            if all(y == Y[indices[0]] for y in [Y[i] for i in indices]):
                r[g]=True
                hg.add(g)
            else:
                r[g]=False
    return False if (False in r.values()) else True

def split(partitions,sim_matrix,i,j):
    g=partitions[i]
    g1=max(partitions)+1
    indices=[l for l,x in enumerate(partitions) if x==g]
    #print("indices %s" %g, indices)
    i,j=sorted([i,j])
    partitions[i]=g
    partitions[j]=g1
    for k in indices:
        #i,k=sorted([i,k])
        #j,k=sorted([j,k])
        if k!=j and k!=i:
            if sim_matrix[i,k]>sim_matrix[j,k]:
                partitions[k]=g
            else:
                partitions[k]=g1

def rsp3_centroids(X,Y,partitions,c_func='centroid'):
    C,Yc=[],[]
    for g in set(partitions):
        indices=[i for i,x in enumerate(partitions) if x==g]
        Xi=[X[i] for i in indices]
        #C.append(centroid(Xi))
        C.append(center_function[c_func](Xi))
        Yc.append(Y[indices[0]])
    return C,Yc

def distance_structs(X):
    h=[]
    N=len(X)
    sim_matrix=np.zeros((N,N),dtype=np.float)
    for i in range(N):
        for j in range(i,N):
            simij=cosine_similarity(X[i],X[j])
            heappush(h, (simij,[i,j]))
            sim_matrix[i,j]=simij
            sim_matrix[j,i]=simij
    return h,sim_matrix

def rsp3(X,Y,c_func='centroid'):
    N=len(X)
    partitions=[0 for i in range(N)]
    h,sim_matrix=distance_structs(X)
    hg=set()
    ht=False
    while h and (not ht):
        current=heappop(h)
        i,j=current[1]
        if partitions[i]==partitions[j] and (partitions[i] not in hg):
                split(partitions,sim_matrix,i,j)
                ht=homogeneous(partitions,Y,hg)
    return rsp3_centroids(X,Y,partitions,c_func)

import random

def choose_centers(X,k=2):
    C=[]
    c1=random.sample(X, 1)[0]
    C.append(c1)
    D=np.array([(1-cosine_similarity(x,c1))**2 for x in X])
    for i in range(1,k):
        print(i)
        cp=(D/D.sum()).cumsum()
        j = np.where(cp >= random.random())[0][0]
        C.append(X[j])
        D=np.array([min([(1-cosine_similarity(x,c))**2 for c in C]) for x in X])
    return C

def kmeans(X,k,max_iter=1000, c_func="centroid"):
    C=[sp.fromlist([]) for x in range(k)]
    Cn=choose_centers(X,k)
    N=len(X)
    it=0
    partitions=[0 for i in range(N)]
    while (False in [C[k].data==Cn[k].data for k in range(k)])and it<max_iter:
        partitions=[i for c,i  in [max([(cosine_similarity(x,Cn[i]),i) for i in range(k)]) for x in X]]
        C=Cn[:]
        for p in range(k):
            ind=[i for i,l in enumerate(partitions) if l==p]
            Xi=[X[i] for i in ind]
            Cn[p]=centroid(Xi)
        it+=1
    print(it)
    return Cn,partitions

def group_elements(X,Y,partitions):
    Xi,Yi=[],[]
    for i,g in enumerate(set(partitions)):
        ind=[j for j,l in enumerate(partitions) if l==g]
        if g!=-1:
            Xi.append([X[j] for j in ind])
            Yi.append([Y[j] for j in ind])
    return Xi,Yi

def _prepare(X,Y,c_func='centroid'):
    Xr,Yr=[],[]
    for Xt,Yt in zip(X,Y):
            #Xr.append(centroid(Xt))
            Xr.append(center_function[c_func](Xt))
            Yr.append(Yt[0])
    return Xr,Yr

def osc(X,Y,c_func='centroid'):
    Xr,Yr=[],[]
    for Xt,Yt in zip(X,Y):
        if all(y == Yt[0] for y in Yt):
            #Xr.append(centroid(Xt))
            Xr.append(center_function[c_func](Xt))
            Yr.append(Yt[0])
        else:
            added=set()
            count=[(Yt.count(y),y) for y in set(Yt)]
            count.sort(reverse=True)
            mayority=[i for i,y in enumerate(Yt) if  Yt[i]==count[0][1]]
            minority=[i for i,y in enumerate(Yt) if  Yt[i]!=count[0][1]]
            for i in minority:
                j=max([(cosine_similarity(Xt[i],Xt[m]),m) for m in mayority])[1]
                if j not in added:
                    Xr.append(Xt[j])
                    Yr.append(Yt[j])
                    added.update([j])
                o=max([(cosine_similarity(Xt[j],Xt[m]),m) for m in minority])[1]
                if o not in added:
                    Xr.append(Xt[o])
                    Yr.append(Yt[o])
                    added.update([o])
    return Xr,Yr

def basic_e_nets(X,Y,eps=0.01, per_class=True, c_func='centroid'):
    if per_class:
        indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    else:
        indices=[[i for i,x in enumerate(X)]]
    H,M=distance_structs(X)
    partitions=[-1 for x in X]
    for ind in indices:
        lista=ind[:]
        random.shuffle(ind)
        while ind:
            i=ind[0]
            partitions[i]=i
            for j in lista:
                if M[i][j] < eps and i!=j and j in ind:
                    partitions[j]=i
                    ind.remove(j)
            ind.remove(i)
    #if per_class:
    #    Xr,Yr=group_elements(X,Y,partitions)
    #else:
    Xg,Yg=group_elements(X,Y,partitions)
    return osc(Xg,Yg,c_func)

def distance_matrices(X,Y):
    M=[]
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    for ind in indices:
        H,Mi=distance_structs([X[i]  for i in ind])
        M.append(Mi)
    return M

def density_nets(X,Y,k,M, c_func='centroid'):
    partitions=[-1 for x in X]
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    #M= M if M else distance_matrices(X,Y)
    for ind  in indices:
        lind=ind[:]
        random.shuffle(lind)
        while lind:
            sk,i=0,lind.pop(0)
            li=ind.index(i)
            #print(ind,np.argsort(-Mi[li]), Mi[li])
            knn=[x for x in  np.argsort(-M[i]) if x in ind]
            for nn in knn:
                if partitions[nn]==-1:
                    partitions[nn]=i
                    sk=sk+1
                if sk==k+1:
                    break
    #print(partitions.count(-1))
    Xg,Yg=group_elements(X,Y,partitions)
    #print(partitions.count(-1))
    return _prepare(Xg,Yg,'centroid')

from collections import OrderedDict as sset
def cnn(X,Y,M,c_func='centroid'):
    partitions=[-1 for x in X]
    Xr,Yr,S,sl,=[],[],sset.fromkeys([]),0
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    for ind in indices:
        k=random.choice(ind)
        Xr.append(X[k])
        Yr.append(Y[k])
        S[k]=k
    while sl!=len(S):
        sl=len(S)
        for i in [i for i,l in enumerate(partitions) if l==-1]:
            #knn=[j for  j in ind[i] if j in S]
            Sl=[j for j in S.keys()]
            lknn=M[i,Sl].argsort()
            nn=S[Sl[lknn[0]]]
            if Y[i]==Y[nn]:
                partitions[i]=partitions[nn]
            else:
                partitions[i]=i
                Xr.append(X[i])
                Yr.append(Y[i])
                S[i]=i
    return Xr,Yr

def rnn(X,Y,M):
    S,partitions=cnn(X,Y,M)
    iknn=M.argsort(axis=1)
    sl=[i for i in S.keys()]
    print("Antes de rnn", len(S))
    for key in sl:
        p=S[key]
        del S[key]
        for i in range(len(X)):
            lknn=[j for j in iknn[i] if j in S.keys()]
            nn=S[lknn[0]]
            if Y[i]!=Y[nn]:
                #print(i,key)
                S[p]=p
                break
    Xr=[X[i] for i in S]
    Yr=[Y[i] for i in S]
    return Xr,Yr,S

def density_enets(X,Y,k,M=[], c_func='centroid'):
    partitions=[-1 for x in X]
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    #M = M if M else distance_matrices(X,Y)
    #print("Inicilizacion completa")
    for ind in indices:
        epsilons=[]
        lind=ind[:]
        random.shuffle(lind)
        while lind:
            sk,i=0,lind.pop(0)
            li=ind.index(i)
            print(li)
            knn=[x for x in  np.argsort(-M[li]) if x in ind]
            #print(knn)
            Xk=[X[i]]
            #partitions[i]=i
            for nn in knn:
                lj=ind.index(nn)
                if partitions[nn]==-1:
                    #print(nn,"->>",i)
                    partitions[nn]=i
                    sk=sk+1
                    Xk.append(X[nn])
                if sk==k+1 or ((1-M[li][lj])>sum(epsilons[-10:])/10 and len(epsilons)>10):
                    break
            if len(Xk)>1:
                ck=centroid(Xk)
                epsilons.append(max([cosine_distance(x,ck) for x in Xk]))
    #print(partitions)
    #print('yyyyy',Mi)
    Xg,Yg=group_elements(X,Y,partitions)
    return _prepare(Xg,Yg,c_func='centroid')

import math
from sklearn.metrics import f1_score

def test_epsnets(X,Y,source,eps):
    #lk=[x for x in range(2, math.ceil(math.sqrt(len(X)))+2,2)]
    base=source.split('.')[0]
    for i in range(30):
        Xr,Yr=epsilon_nets(X,Y,eps)
        test_svm(Xr,Yr,source,"enets/%s.k%st%s.predicte.json" %(base,eps,i))

def test_dnets(X,Y,source,k,n=50):
    #lk=[x for x in range(2, math.ceil(math.sqrt(len(X)))+2,2)]
    M=distance_matrices(X,Y)
    base=source.split('.')[0]
    #for k in lk:
    #print(k)
    for i in range(n):
        Xr,Yr=density_nets(X,Y,k,M)
        test_svm(Xr,Yr,source,"dnetsc/%s.k%st%s.predicte.json" %(base,k,i))

def test_denets(X,Y,source,k,n=50):
    #lk=[x for x in range(2, math.ceil(math.sqrt(len(X)))+2,2)]
    M=distance_matrices(X,Y)
    base=source.split('.')[0]
    #for k in lk:
    #print(k)
    for i in range(n):
        Xr,Yr=density_enets(X,Y,k,M)
        test_svm(Xr,Yr,source,"denetsc/%s.k%st%s.predicte.json" %(base,k,i))

def test_cnn(X,Y,source,n=50):
    #lk=[x for x in range(2, math.ceil(math.sqrt(len(X)))+2,2)]
    H,M=distance_structs(X)
    base=source.split('.')[0]
    #for k in lk:
    #print(k)
    for i in range(n):
        Xr,Yr=cnn(X,Y,M)
        test_svm(Xr,Yr,source,"cnn/%s.cnn1t%s.predicte.json" %(base,i))


from scipy.sparse import csr_matrix
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

def to_csr_matrix(X):
    data,row,col=[],[],[]
    for i in range(len(X)):
        c=list(X[i].index)
        row+=[i for e in c]
        col+=c
        data+=list(X[i].data)
    M=csr_matrix((data,(row,col)))
    return M


from scipy.sparse.linalg import svds as sparsesvd
from queue import Queue
def fsgp(X,Y, c_func='centroid'):
    Xs=to_csr_matrix(X).tocsc()
    partitions=np.array([0 for e in X])
    q=Queue()
    q.put(0)
    M,mi=0,0
    while not q.empty() and mi<5:
        g=q.get()
        M=np.amax(partitions)+1
        indices=[i for i,gx in enumerate(partitions) if gx==g]
        Xl=Xs[indices,:]
        ut,s,vt=sparsesvd(Xl,1)
        xm=center_function[c_func]([X[i] for i in indices])
        vt=sp.fromlist(vt[0]).add(xm)
        R=[(i,x.add(xm.mul2(-1)).dot(vt)) for x,i in [(X[i],i) for i in indices]]
        li,lr=[i for i,r in R if r<0],[i for i,r in R if r>0]
        if li==lr or (not li) or (not lr):
            if li: partitions[li]=-1
            if lr: partitions[lr]=-1
        else:
            partitions[li]=g
            partitions[lr]=M
            if not all(y == Y[li[0]] for y in [Y[i] for i in li]):
                q.put(g)
            if not all(y == Y[lr[0]] for y in [Y[i] for i in lr]):
                q.put(M)
    Xg,Yg=group_elements(X,Y,partitions)
    return _prepare(Xg,Yg)


def _assign_prototype(X,C):
    #print(partitions)
    #Xi=[X[i] for i,l in enumerate(partitions) if l!=-1]
    return np.array([max([(cosine_similarity(x,v[0]), i) for i,v in C.items()])[1] for x in X])


def _centroids(X,Y):
    C={}
    for ci,l in enumerate(sorted(set(Y))):
        indices=[j for j,y in enumerate(Y) if l==y]
        Xi =[X[i] for i in indices]
        C[ci]=(centroid(Xi),l)
    return C

def _divide(X,Xs,xm,indices):
    Xl=Xs[indices,:]
    xm=centroid([X[i] for i in indices])
    print(">"*10,indices)
    ut,s,vt=sparsesvd(Xl,1)
    vt=sp.fromlist(vt[0]).add(xm)
    R=[(i,x.add(xm.mul2(-1)).dot(vt)) for x,i in [(X[i],i) for i in indices]]
    return [i for i,r in R if r<0],[i for i,r in R if r>0]

def _update_centroids(X,C,partitions):
    for ci in set(partitions):
            indices=[j for j,l in enumerate(partitions) if l==ci]
            if indices:
                Xi =[X[i] for i in indices]
                C[ci]=(centroid(Xi),C[ci][1])
            else:
                C.pop(ci)

def _nearest_prototype(x,G):
    dk,ck,lk=max([(cosine_similarity(x,c[0][0]), ci, c[0][1]) for ci,c in G.items()])
    return dk,ck,lk

def _update_group(G,X,k,indices,lk):
    Xk=[X[i] for i in indices]
    G[k]=((centroid(Xk),lk),indices)

def _update_groups(X,Y,G,indices,l):
    M=max(G.keys())+1
    ind={M:[[],l]}
    for i in indices:
        d,cn,ln=_nearest_prototype(X[i],G)
        if ln==Y[i]:
            if ln in ind.keys():
                ind[cn][0].append(i)
            else:
                ind[cn]=[[i],ln]
        else:
            ind[M][0].append(i)
    #print(ind)
    for k,indl in ind.items():
        indk,lk=indl
        _update_group(G,X,k,indk,lk)

def _update_all(X,G,q):
    C=dict([(k,c[0]) for k,c in G.items()])
    partitions=_assign_prototype(X,C)
    G=dict([(ic,(c, [i for i,x in enumerate(partitions) if x==ic])) for ic,c in C.items()])
    q.queue.clear()
    for g in sorted(G.keys()):
        q.put(g)

def sgp(X,Y,c_func='centroid'):
    C=_centroids(X,Y)
    partitions=_assign_prototype(X,C)
    G=dict([(ic,(c, [i for i,x in enumerate(partitions) if x==ic])) for ic,c in C.items()])
    Xs,q=to_csr_matrix(X).tocsc(),Queue()
    [q.put(g) for g in sorted(G.keys())]
    im=0
    nc=1
    M=max(G.keys())
    while (nc  or (not q.empty())) and im<1000000:
        im+=1
        if nc!=0:
            nc=0
            _update_all(X,G,q)
        k=q.get()
        #print("%s: xxxxxxxxxxxx" %im)
        gk,indices_k=G[k]
        ck,lk=gk
        all_same=all(y==lk for y in [Y[i] for i in indices_k]) and len(indices_k)>1
        all_diff=all(y!=lk for y in [Y[i] for i in indices_k]) and len(indices_k)>1
        some_same=(lk in [Y[i] for i in indices_k])
        #print(all_same,some_same,all_diff,indices_k,k)
        if len(indices_k)==1:
             print("grupo Unitario")
             ik=indices_k[0]
             d,cn,ln=_nearest_prototype(X[ik],G)
             if ln==Y[ik] and cn!=ik:
                 print(ik,"Added to prototype -->",cn)
                 gcn,indices_cn=G[cn]
                 _update_group(G,X,cn,indices_cn+[ik],ln)
                 nc=1
             else:
                 print(ik,"created his own prototype -->",ik)
                 G[ik]=((X[ik],Y[ik]),[ik])
                 print(G[ik],Y[ik])
        elif all_diff:
            print("Todos diferentes")
            indk,indj=_divide(X,Xs,ck,indices_k)
            print(indk, indj)
            _update_group(G,X,k,indk,lk)
            _update_group(G,X,max(G.keys())+1,indj,lk)
            nc=1
        elif some_same and not all_same:
            print("Algunos diferentes")
            indk,indj=[i for i in indices_k if lk==Y[i]],[i for i in indices_k if lk!=Y[i]]
            _update_group(G,X,k,indk,lk)
            _update_groups(X,Y,G,indj,lk)
            nc=1
        print(k,im,nc, not(q.empty()), im<len(X))
    return G

def test_svm(C,Y,source,dest):
    le = preprocessing.LabelEncoder()
    le.fit(list(set(Y)))
    Yr=le.transform(Y)
    R,T=get_data(source)
    d=R[0]['vecsize']
    data,row,col=[],[],[]
    for i in range(len(C)):
        c=list(C[i].index)
        row+=[i for e in c]
        col+=c
        data+=list(C[i].data)
    X=csr_matrix((data,(row,col)),shape=(len(C),d))
    data,row,col=[],[],[]
    for i in range(len(T)):
        c=list(T[i].index)
        row+=[i for e in c]
        col+=c
        data+=list(T[i].data)
    Ts=csr_matrix((data,(row,col)),shape=(len(T),d))
    clf = SVC(kernel='linear')
    clf.fit(X, Yr)
    P=list(le.inverse_transform(clf.predict(Ts)))
    S=[]
    for i in range(len(P)):
        S.append({})
        S[-1]['text']=R[i]['text']
        S[-1]['klass']=P[i]
    S[-1]['datasize']=len(C)
    open(dest,'w').write("%s" % "\n ".join(json.dumps(r) for r in S))
    return P


#def set_size(source):
#    f=open(source,'r')
#    txt=f.readlines()[0]
#    f.close()
#    R=json.loads(txt)
#    R['traninig_set_size']=0
#    open(source,'w').write("%s" % "\n ".join(json.dumps(r) for r in [R]))

#set_size('r8.res')

Xgr,Ygr=[],[]

def pdist(data):
    import sys
    #imin,imax,n=ind
    #print(ind)
    #indices=[(i,j) for i in range(0,n) for j in range(0,n) if i<j]
    res=np.array([(cosine_distance(xi,xj),int(i),int(j)) for xi,xj,i,j in data])
    return res
    #print(imin,imax)

def ppdist(X,n_proc=32):
    n=len(X)
    print(n_proc, n)
    p = Pool(n_proc)
    list_of_ranges=distribute(X,n_proc)
    #print(list_of_ranges)
    res =list(itertools.chain(* p.map(pdist, list_of_ranges)))
    MM=np.zeros((n,n), dtype='float64')
    for D,i,j in res:
        MM[int(i),int(j)]=D
        MM[int(j),int(i)]=D
    return MM
    #f=open('semeval.dist.np','wb')
    #pickle.dump(res,f)
    #np.save(f,res)


def distribute(X,n_proc):
    n=len(X)
    total=int(n*(n+1)/2-n)
    r=total%n_proc
    nz=int(total/n_proc)
    l=[[i,i+nz,n] for i in range(0,total-r-1,nz)]
    if r:
        l[-1][1]=l[-1][1]+r
    aux=[(X[i],X[j],i,j) for i in range(0,n) for j in range(0,n) if i<j]
    data=[]
    for imin,imax,d in l:
         data.append([aux[i] for i in range(imin,imax)])
    return data


import _pickle as pickle





#     f=open('matrix.dist.np','wb')
#     np.save(f,np.array(MM))
#     #f=open('knn.dist.np','wb')
#     #np.save(f,np.array(knn))
#     end1=timer()
#     print("Termine de guardar los indices", end1-start)
#     print("Tiempo total", end1-start1)
#     #pickle.dump(res,f)

def epsilon_nets(X,Y,M,eps=0.1):
    indices=[[i for i in range(len(X)) if Y[i]==l] for l in set(Y)]
    partitions=[-1 for x in X]
    Xr,Yr=[],[]
    #print(indices)
    for ind in indices:
        zz=len(ind)
        centers={}
        i=random.choice(ind)
        partitions[i]=i
        m=M[i,ind].argsort()[-1]
        centers[i]=(M[i,ind[m]],ind[m],M[i,ind[m]].mean())
        #print(centers)
        while any([Dm>eps for D,j,Dm in centers.values()]) and zz>0:
            zz=zz-1
            dm,im,dd=max([(Dm,j,D) for D,j,Dm in centers.values()])
            #if zz%100==0:
                #print([(Dm,Dm>eps) for D,j,Dm in centers.values()])
                #print(centers.values())
            #   print(any([Dm>eps for D,j,Dm in centers.values()]))
            #   print(len(centers),len(ind),Y[ind[0]],dm,dd,dm>eps)
            partitions[im]=im
            centers[im]=()
            for i in set(ind)-set(centers.keys()):
                    cl=[c for c in centers.keys()]
                    m=M[i,cl].argsort()[0]
                    partitions[i]=cl[m]
            #print(partitions,centers)
            for c in centers.keys():
                ci=[j for j,l in enumerate(partitions) if c==l]
                im=M[c,ci].argsort()[-1]
                centers[c]=(M[c][ci[im]],ci[im],M[c,ci].mean())
            #print(centers,partitions)
        Xr+=[X[c] for c in centers.keys()]
        Yr+=[Y[c] for c in centers.keys()]
    return Xr,Yr


def chunks(X, n):
    L=[]
    for i in range(0, len(X), n):
        L.append(X[i:i + n])
    return L


if  __name__=='__main__':

    from timeit import default_timer as timer
    ftrain='3-32.train-70-30.json.train.vspace.json'
    ftest='3-32.train-70-30.json.test.vspace.json'
    fen='external_data.vspace.json'
    ccl={'positive':'P','neutral':'NEU','negative':'N','NONE':'NONE'}
    Xtr,Ytr,dtr=load(ftrain)
    #Ytr=[ccl[y] for y in Ytr]
    Xt,Yt,dt=load(ftrain)
    Xe,Ye,de=load(fen)
    Ye=[ccl[y] for y in Ye]
    clases=set(Ye)
    lista={}
    for clase in clases:
        l=[i for i,y in enumerate(Ye) if y==clase]
        random.shuffle(l)
        aux=chunks(l,10000)
        if len(aux)>1 and len(aux[-1])<5000:
            aux[-2]=aux[-2]+aux[-1]
            del aux[-1]
        lista[clase]=aux
    for eps in [0.1]:
        for clave,vals in  lista.items():
            k=0
            for indices in vals:
                print("%s chunk de: %s" %(clave,len(indices)))
                X=[Xe[i] for i in indices]
                Y=[Ye[i] for i in indices]
                MM=ppdist(X)
                Xep,Yep=epsilon_nets(Xgr,Ygr,MM,eps)
                Xt=Xt+Xep
                Yt=Yt+Yep
            print("Termine %s chunk de: %s, agregue %s instancias (eps:%s)" %(clave,len(indices), len(Yep),eps))
        test_svm(X,Y,ftest,Xt,Yt,'enets/tass.predicte.enets-%(eps)s.json' %locals())




# if  __name__=='__main__':
#     from timeit import default_timer as timer
#     ff='external_data.vspace.json'
#     log=open('/home/job/smtxtf/logs.txt','w')
#     start1 = timer()
#     X,Y,d=load(ff)
#     Xgr,Ygr=X[:10000],Y[:10000]
#     end=timer()
#     print("Termine de cargar los archivos", end-start1)
#     start = timer()
#     res=ppdist()
#     end = timer()
#     print("Termine de calcular las distancias", end-start)
# #     #start = timer()
# #     #res.sort()
# #     #end=timer()
# #     #print("Termine de ordenar las distancias", end-start)
# #     #knn=[[] for j in range(len(Xgr))]
#     MM=np.zeros((len(Xgr),len(Xgr)), dtype='float64')
#     star=timer()
#     for D,i,j in res:
#         if (i==0 and j==5000) or (j==0 and i==5000):
#             print(">>>>>>>>>>>>>",i,j,D)
#         MM[int(i),int(j)]=D
#         MM[int(j),int(i)]=D
#     end=timer()
#     print("Termine de crear los indices", end-start)
#     start=timer()
#     f=open('tass.matrix.dist.np','wb')
#     np.save(f,np.array(MM))
#     #f=open('knn.dist.np','wb')
#     #np.save(f,np.array(knn))
#     end1=timer()
#     print("Termine de guardar los indices", end1-start)
#     print("Tiempo total", end1-start1)
#     #pickle.dump(res,f)
#     Xr,Yr=cnn(Xgr,Ygr,MM)



# if __name__ == "__main__":
#       import argparse
#       parser = argparse.ArgumentParser()
#       parser.add_argument("training_file", help=" File wiht trainning samples in microtc format")
#       parser.add_argument("test_file", help="File  wiht  test samples in microtc format")
#       parser.add_argument("-e", "--epsilon")

#       args = parser.parse_args()
#       X,Y,d=load(args.training_file)
#       eps=load(args.epsilon)
#       test_epsnets(X,Y,args.test_file,float(eps))





# if __name__ == "__main__":
#      import argparse
#      parser = argparse.ArgumentParser()
#      parser.add_argument("training_file", help=" File wiht trainning samples in microtc format")
#      parser.add_argument("test_file", help="File  wiht  test samples in microtc format")
#      parser.add_argument("enforcement_file", help=" File with additional samples in microtc format")

#      args = parser.parse_args()
#      X,Y,d=load(args.training_file)
#      Xe,Ye,de=load(args.enforcement_file)
#      #test_dnets(X,Y,args.test_file)
#      if args.method=='rocchio':
#          C,Yc=rocchio(Xr,Yr)
#          del Xr
#          del Yr
#          X,Y,d=load(args.training_file)
#          test_svm(X+C,Y+Yc,args.test_file,'semeval/rocchio/semeval_english2007.predicte.json')

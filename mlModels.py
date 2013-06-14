import numpy as np

from sklearn import preprocessing as skpp
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import svm 

def modelAccuracy(model,x,y,x2=[],y2=[]):
# fit and predict model accuracy for general model
    model.fit(x,y)
    if len(x2) == 0:
        # if no cross validation set
        x2 = x
        y2 = y
    predict = model.predict(x2)
    accuracy = float(sum(y2 == predict))/len(y2)
    return((model,accuracy))


def nearest(x,y,x2,y2,neighbors=5,weights="uniform"):
# nearest neighbor classification
# http://scikit-learn.org/0.13/modules/neighbors.html#neighbors
    
    knn = KNeighborsClassifier(n_neighbors=neighbors,weights=weights)
    (knn,accuracy) = modelAccuracy(knn,x,y,x2,y2)
    print neighbors,weights,accuracy
    return(knn)

def knnLoop(x,y,x2,y2):
    
    for w in ['uniform','distance']:
        for i in range(4,10):
            model = nearest(x,y,x2,y2,i,w)


def logisticReg(x,y,x2,y2,C=1):
# logistic regression classifier
# http://scikit-learn.org/0.13/modules/linear_model.html#logistic-regression

    model = linear_model.LogisticRegression(C=C)
    model,accuracy = modelAccuracy(model,x,y,x2,y2)
    print C,accuracy
    return(model)
    
def lrLoop(x,y,x2,y2):
    for i in range(7):
        C = float(10**i)
        model = logisticReg(x,y,x2,y2,C)

    
def pcaLoop(xtrain,y):
# check PCA variance explained
    for i in range(len(xtrain[0])):
        pca = PCA(n_components=i)
        pca.fit(xtrain)
        print(sum(pca.explained_variance_ratio_))

        
def supportVecMachine(x,y,x2,y2,C=1,kernel="linear",degree=2):
# support vector machines
# http://scikit-learn.org/0.13/modules/svm.html
    if kernel == "poly":
        model = svm.SVC(C=C,kernel=kernel,degree=degree)
    else:
        model = svm.SVC(C=C,kernel=kernel)
        
    (model,accuracy) = modelAccuracy(model,x,y,x2,y2)
    print C,kernel,degree,accuracy
    return(model)


def svmLoop(x,y,x2,y2):
    kernel = "poly"
    polyMax = 5

    if kernel == "poly":
        for j in range(1,polyMax):
            for i in range(1,11):
                C = float(i)
                supportVecMachine(x,y,x2,y2,C,"poly",j)
    else:
        for i in range(1,11):
            C = float(i)
            model = svm.SVC(C=C)
            # print C,accuracy



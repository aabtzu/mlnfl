import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing as skpp



dataDir = "data/"
def readData(datafile):
    
    data = np.loadtxt(dataDir+datafile,delimiter=',')
    return(data)

def splitTrainingData(x,y,trainPct=0.75):
    # make sure data is legit
    if len(y) != len(x):
        print "len mismatch btw x and y", len(x), len(y)
        return None

    n = int(len(x) * trainPct)
    xtrain = x[:n]
    ytrain = y[:n]
    xtest = x[n:]
    ytest = y[n:]
    return (xtrain,ytrain,xtest,ytest)

def scaleData(data):

    ss = skpp.StandardScaler().fit(data)
    data2 = ss.transform(data)
    return(data2)
    
def hist(x,nbins):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(x,nbins)
    
    plt.show(block=False)

def scatter(x,y):

    fig = plt.figure()
    plt.scatter(x,y)
    plt.show(block=False)

def line(x,y=None):
    fig = plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x,y)
    plt.show(block=False)


def nearest(x,y,x2,y2,neighbors=5,weights="uniform"):
    
    knn = KNeighborsClassifier(n_neighbors=neighbors,weights=weights)
    (knn,accuracy) = modelAccuracy(knn,x,y,x2,y2)
    print neighbors,weights,accuracy
    return(knn)

def knnLoop(x,y,x2,y2):
    
    for w in ['uniform','distance']:
        for i in range(4,10):
            model = nearest(x,y,x2,y2,i,w)


def logisticReg(x,y,x2,y2,C=1):

    model = linear_model.LogisticRegression(C=C)
    model,accuracy = modelAccuracy(model,x,y,x2,y2)
    print C,accuracy
    return(model)
    
def lrLoop(x,y,x2,y2):
    for i in range(7):
        C = float(10**i)
        model = logisticReg(x,y,x2,y2,C)

    
def pca(xtrain,y):

    for i in range(len(xtrain[0])):
        pca = PCA(n_components=i)
        pca.fit(xtrain)
        print(sum(pca.explained_variance_ratio_))

        
def supportVecMachine(x,y,x2,y2,C=1,kernel="linear",degree=2):

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

def modelAccuracy(model,x,y,x2=[],y2=[]):
    model.fit(x,y)
    if len(x2) == 0:
        x2 = x
        y2 = y
    predict = model.predict(x2)
    accuracy = float(sum(y2 == predict))/len(y2)
    return((model,accuracy))

def writePredict(y,outfile):
    np.savetxt(outfile,y,"%d")

    

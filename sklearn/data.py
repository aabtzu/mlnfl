import numpy as np
from sklearn import preprocessing as skpp

dataDir = "data/"
def readData(datafile): 
    data = np.loadtxt(dataDir+datafile,delimiter=',')
    return(data)

def writePredict(y,outfile,fmt=None):
    if fmt is None:
        fmt = "%d"
    np.savetxt(outfile,y,fmt)

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

def scaleData(data,ss=None):
# standardize data w/ sklearn
    if ss is None:
        ss = skpp.StandardScaler().fit(data)
    data2 = ss.transform(data)
    return(ss,data2)





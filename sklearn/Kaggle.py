import data
import mlModels as mlm
import plot
from sklearn import cross_validation as cv


def runTrain():

    print "getting data ..."
    (x,y,x2,y2) = loadData()

    print "run nearest neighbors ..."
    m = mlm.knnLoop(x,y,x2,y2)

    print "run logistic regression ..."
    m = mlm.lrLoop(x,y,x2,y2)

    print "run support vector machines - linear..."
    m = mlm.svmLoop(x,y,x2,y2)    

    print "run support vector machines - poly..."
    m = mlm.svmLoop(x,y,x2,y2,"poly")

    print "run support vector machines - rbf..."
    m = mlm.svmLoop(x,y,x2,y2,"rbf")




def loadData():

    trainPct = .80
    d = data.readData("train.csv")
    y = data.readData("trainLabels.csv")

    (scaler,d2) = data.scaleData(d)
    (x1,x2,y1,y2) = cv.train_test_split(d2,y,test_size=1-trainPct)
    # (x,y,x2,y2) = data.splitTrainingData(d2,y,trainPct)
    return (x1,y1,x2,y2)

    
def getSubmit(m):
    x = data.readData("test.csv")
    xx = data.scaleData(x)
    yy =m.predict(xx)
    data.writePredict(yy,"out.csv")
    return(yy)



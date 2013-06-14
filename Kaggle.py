import data
import mlModels as mlm
import plot


def runTrain():

    trainPct = 1
    d = data.readData("train.csv")
    y = data.readData("trainLabels.csv")

    d2 = data.scaleData(d)
    (x,y,x2,y2) = data.splitTrainingData(d2,y,trainPct)

    print "run nearest neighbors ..."
    m = mlm.knnLoop(x,y,x2,y2)

    print "run logistic regression ..."
    m = mlm.lrLoop(x,y,x2,y2)

    print "run support vector machines - linear..."
    m = mlm.svmLoop(x,y,x2,y2)    

    print "run support vector machines - poly..."
    m = mlm.svmLoop(x,y,x2,y2,"poly")




    

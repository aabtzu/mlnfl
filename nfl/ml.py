__author__ = 'amit.bhattacharyya'

import numpy as np

def sigmoid(z):

    g = 1 / ( 1 + np.exp(-z))
    return(g)


# cost function
def costFunction(theta,X,y):

    m,n  = X.shape
    h = sigmoid(np.dot(X,theta))
    h = h.reshape((m,1))
    
    #print h.shape
    #print y.shape

    J = -1 *  sum( y * np.log(h) + (1-y) * np.log(1-h) ) / m
    grad = (sum (np.dot(h-y, np.ones((1,n))) * X) / m)
    grad = grad.reshape((n,1))
    return J #,grad

def costDerivative(theta,X,y):

	m,n  = X.shape
	h = sigmoid(np.dot(X,theta))
	h = h.reshape((m,1))

	grad = (sum (np.dot(h-y, np.ones((1,n))) * X) / m)
	#grad = grad.reshape((n,1))
	return grad


def predict(theta,X):

	prob = sigmoid(np.dot(X,theta))
	p = prob > 0.5;
	return prob



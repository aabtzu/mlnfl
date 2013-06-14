import matplotlib.pyplot as plt

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


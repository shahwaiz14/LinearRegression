import numpy as np
import matplotlib.pyplot as plt

#data
x = [-2,-1,0,1,2]
y = [0,0,1,1,3]

#model : y = w*x + b

#plugs in x and returns the list of ytilda based on w,b learnt from GD
def evaluate(w,b):
    return [w*xi + b for xi in x]

#finds the loss 
def loss(w,b):
    evaluate_all = evaluate(w,b)
    listloss = [(y-yt)**2 for y, yt in zip(y, evaluate_all)]
    return sum(listloss)

#computes the gradient of cost with respect to weight
def gradw(w,b):
    l = [(y-yt)*xi for y, yt, xi in zip(y, evaluate(w,b),x)]
    return sum(np.dot(-2,l))

#computes the gradient of cost with respect to bias
def gradb(w,b):
    l = [(y-yt) for y, yt in zip(y, evaluate(w,b))]
    return sum(np.dot(-2,l))

#plots the graph of loss vs no. of iterations
def pl(iterationsArr, lostArr):
    plt.plot(iterationsArr, lostArr)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()

##################################################
w = 1 #initial weight
b = 2 #initial bias
list_iteration = []
list_loss = []
numberOfIterations = 50

for i in range(numberOfIterations):
    list_iteration.append(i)
    list_loss.append(loss(w,b))
    #0.01 is the learning rate
    w = w - 0.01*gradw(w,b)
    b = b - 0.01*gradb(w,b)
pl(list_iteration,list_loss)
print ("w = " + str(w))
print ("b = " + str(b))
    

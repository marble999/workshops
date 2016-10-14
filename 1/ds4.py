""" Plot stuff first to see what it looks like before we try out different feature functions"""
%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Data
dataset = pd.read_csv('dataset_4_full.txt')
dataset = dataset.values

#Plot
x = []
y = []

for i in range(len(dataset)):
    x.append(float(dataset[i][0]))
    y.append(float(dataset[i][1]))
    
plt.figure(1)
plt.plot(x,y, "ro")
plt.title('First attempt to see what it looks like')

""" This data looks kinda like a sinusoidal function. """

def ols(xes, yes):
    """Nick OLS code because we're lazy"""
    n = len(xes)
    xbar = np.mean(xes)
    ybar = np.mean(yes)
    xvar = np.mean([x*x for x in xes])
    xybar = np.mean([x*y for (x,y) in zip(xes, yes)])
    m = (xybar - (xbar * ybar))/(xvar - xbar**2)
    b = ybar - m*xbar
    return m, b

#Change our features to a sinusoidal
y_sin = []
amplitude = (max(y)-min(y))/2
avg = np.mean(y)

for y_val in y:
    y_sin.append(math.asin((y_val-avg)/amplitude))
    
print y
print y_sin
plt.figure(2)
plt.plot(x, y_sin, "ro")

#Apply OLS to sinusoidal
m,b = ols(x,y_sin)
y_predict = []
for y_val in y_sin:
    y_predict.append(m*y_val+b)
    
# plt.figure(2)
# plt.plot(x,y_predict)


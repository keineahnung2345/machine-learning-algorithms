# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:56:07 2018

@author: user
"""

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

m = 5
n0 = 3
n1 = 4
n2 = 2

np.random.seed(1)

w1 = np.asmatrix(np.zeros((n1,n0), dtype=np.float32))
w2 = np.asmatrix(np.zeros((n2,n1), dtype=np.float32))
#b1 = np.asmatrix(np.random.randn(n1)).T
#b2 = np.asmatrix(np.random.randn(n2)).T
b1 = np.asmatrix(np.zeros((n1,1), dtype=np.float32))
b2 = np.asmatrix(np.zeros((n2,1), dtype=np.float32))

#
x = np.asmatrix(np.random.randn(n0, m), dtype=np.float32)
y = np.asmatrix(np.random.randint(n2, size=m))
y_onehot = np.zeros((n2, m), dtype=np.float32)
y_onehot[y, np.arange(m)] = 1.
y_onehot = np.asmatrix(y_onehot)
y = y_onehot

steps = 3

history = {}

for step in range(steps):
    # Forward Propagation
    z1 = w1*x+b1
    a1 = sigmoid(z1)
    z2 = w2*a1+b2
    a2 = sigmoid(z2)
    
    #Back Propagation
    dz2 = a2 - y
    dw2 = (1/m)*dz2*a1.T
    db2 = (1/m)*np.sum(dz2, axis=1)
    da1 = w2.T*dz2
    dz1 = np.multiply(w2.T*dz2, np.multiply(a1, (1-a1)))
    t1 = w2.T*dz2
    t2 = np.multiply(a1, (1-a1))
    assert np.all(dz1==np.multiply(w2.T*dz2, np.multiply(a1, (1-a1))))
    dw1 = (1/m)*dz1*x.T
    db1 = (1/m)*np.sum(dz1, axis=1)
    
    print(step+1, 'th step(before weight update)')
    history[step+1] = {}
    history[step+1]['w2(before)'] = w2
    history[step+1]['b2(before)'] = b2
    history[step+1]['z2'] = z2
    history[step+1]['a2'] = a2
    history[step+1]['w1(before)'] = w1
    history[step+1]['b1(before)'] = b1
    history[step+1]['z1'] = z1
    history[step+1]['a1'] = a1
    history[step+1]['dz2'] = dz2
    history[step+1]['dw2'] = dw2
    history[step+1]['db2'] = db2
    history[step+1]['da1'] = da1
    history[step+1]['dz1'] = dz1
    history[step+1]['dw1'] = dw1
    history[step+1]['db1'] = db1
    
    #Update weight and bias
    lr = 0.01
    w2 = w2 - dw2*lr
    b2 = b2 - db2*lr
    w1 = w1 - dw1*lr
    b1 = b1 - db1*lr
    
    history[step+1]['w2(after)'] = w2
    history[step+1]['b2(after)'] = b2
    history[step+1]['w1(after)'] = w1
    history[step+1]['b1(after)'] = b1
    
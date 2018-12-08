# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:18:46 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython

#ipython = get_ipython()
#ipython.magic("matplotlib qt5")

def sigmoid(z):
    return 1/(1+np.exp(-z))

def msl(x, y, w, b):
    y_hat = sigmoid(w*x+b)
    return (y-y_hat)**2

m = 100
b= 0
ws = np.linspace(-10,10,1001)

ann_str = ""

for count in range(100): #plot 100 images
    flag = False
    while not flag:
        m = np.random.randint(3,100)
        X = np.random.rand(m)
        Y = np.random.choice([0,1],m)
        
        # construct mean square loss function
        # (x_i, y_i) fixed, given w, output loss
        losses = [[msl(x,y,w,b) for w in ws] for x,y in zip(X,Y)]
        losses = np.array(losses)
        
        # construct mean square cost function
        # (X, Y) fixed, given w, output cost
        # average of loss functions
        costs = np.mean(losses, axis=0)
        
        # find local minimum in msc, if < global minimum, bingo!
        lm_ixs = []
        lms = []
        for i in range(1,len(costs)-1):
            if costs[i-1] > costs[i] and costs[i+1] > costs[i]:
                lm_ixs.append(i)
                lms.append(costs[i])
        #lms = [costs[i] for i in range(1,len(costs)-1) 
        #    if costs[i-1] > costs[i] and costs[i+1] > costs[i]]
        gm_ix, gm = np.argmin(costs), np.min(costs)
        print(gm, lms)
        
        if len(lms)==0:
            print('no local minimum')
        elif len(lms)==1:
            if min(lms)>gm:
                ann_str = 'local minimum greater than global minimum'
                print(ann_str)
                flag = True
            else:
                # print('nothing')
                pass
        else:
            ann_str = 'multiple local minima'
            print(ann_str)
            flag = True

    plt.cla()
    plt.plot(ws, costs)
    plt.plot(ws[gm_ix], gm,'ro')
    #plt.annotate("global minimum: (%.5f, %.5f)" % (ws[gm_ix], gm), 
    #             (ws[gm_ix], gm*0.97),
    #            )
    for lm_ix, lm in zip(lm_ixs, lms):
        if lm_ix!=gm_ix:
            plt.plot(ws[lm_ix], lm, 'gx')
            #plt.annotate("local minimum: (%.5f, %.5f)" % (ws[lm_ix], lm), 
            #         (ws[lm_ix], lm*0.97),
            #        )
    plt.title(ann_str)
    plt.savefig("{}_{}.svg".format(ann_str, count), format="svg")
    # plt.show()
    
    np.savetxt('X_{}.txt'.format(count), X, fmt ='%f')
    np.savetxt('Y_{}.txt'.format(count), Y, fmt ='%f')
    
# find the id of image plotted from least samples
#m_min = 100
#count_min = -1
#for count in range(100):
#    m_cur = np.loadtxt('X_{}.txt'.format(count)).shape[0]
#    if m_cur < m_min:
#        count_min = count
#        m_min = m_cur
    
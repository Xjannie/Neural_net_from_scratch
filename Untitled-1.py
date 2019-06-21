#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../Neural_net_from_scratch'))
	print(os.getcwd())
except:
	pass

#%%



#%%
import pandas as pd
import numpy as np


#%%
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

#%% [markdown]
# https://cdn-images-1.medium.com/max/800/1*sX6T0Y4aa3ARh7IBS_sdqw.png
# https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7

#%%
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[-1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.out     = np.zeros(self.y.shape)
        
    def forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.out = sigmoid(np.dot(self.layer1,self.weights2))
        
    def cal_loss(self):
        self.loss = np.sum((self.y-self.out)**2)
        
    def backpropagate(self):
        self.weight2_delta = np.dot(self.layer1.T,(2*(self.y-self.out)*sigmoid_derivative(self.out)))
        self.weight1_delta = np.dot(self.input.T,np.dot(self.weights2.T,(2*(self.y-self.out)*sigmoid_derivative(self.out)))*sigmoid_derivative(self.layer1))

    
    def update_weights(self):
        self.weights1 += self.weight1_delta
        self.weights2 += self.weight2_delta

    
    
   


def sigmoid_derivative(x):
    return x * (1.0 - x)



#%%
X = np.array([[0,0,1,1],[0,1,0,1],[1,1,1,1],[0,1,1,0]])


#%%
y = np.array([[0],[1],[1],[0]])


#%%
X.shape[-1]


#%%
net = NeuralNetwork(X,y)


#%%
for i in range(500):
    net.forward()
    net.backpropagate()
    net.update_weights()
    net.cal_loss()
    print(net.loss)

#%%
net.cal_loss()


#%%
net.loss

#%%
#test
net.backpropagate()

#%%
net.update_weights()


net.out


#%%

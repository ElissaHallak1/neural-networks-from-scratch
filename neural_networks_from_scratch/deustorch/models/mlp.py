import deustorch.nn as nn
import numpy as np 

class MLP4:
    def __init__(self):
        self.layers = [
            nn.Linear(784, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 10)
        ]
        self.f = [nn.ReLU() for _ in range (4)] + [None]
     
    def forward(self, A):
        L = len(self.layers)
        for i in range(L):
            Z = self.layers[i].forward(A) # forward pass - sum
            if self.f[i]!= None:
                A = self.f[i].forward(Z) # this is what the activation function gives us
            else:
                A = Z
        return A
    
    def backward(self, dLdA):
        L = len(self.layers)
        for i in reversed(range(L)):
            if self.f[i]!=None:
                dAdZ = self.f[i].backward()
                dLdZ = dLdA * dAdZ
            else:
                dLdZ = dLdA
            dLdA = dLdZ @ self.layers[i].W

            self.layers[i].dLdW = dLdZ.T @ self.layers[i].A
            self.layers[i].dLdb = np.sum(dLdZ, axis=0, keepdims=True).T
            
        return None
    

    
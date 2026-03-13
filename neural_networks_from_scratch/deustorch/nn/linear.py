import numpy as np
class Linear:

    def __init__(self, in_features, out_features):
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        self.dLdW = np.zeros((out_features, in_features))
        self.dLdb = np.zeros((out_features, 1))
    
    def forward(self, A):

        self.A = A
        N, C = np.shape(A)
        self.N = N
        self.Ones = np.ones((N, 1))
        Z = A @ self.W.T + self.Ones @ self.b.T

        return Z

    def backward(self, dLdZ):
        
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.Ones

        return dLdA
    

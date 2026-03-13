import numpy as np
class MSELoss():
    def forward ( self , A , Y ) :
        N, C = np.shape(A)
        SE = (A - Y) * (A - Y)
        SSE = np.sum(SE)
        L = 1/(N * C) * SSE
        self.A = A
        self.Y = Y
        return L

    def backward ( self , A, Y ) :
        dLdA = A - Y
        return dLdA


class CrossEntropyLoss():
    
    def forward(self, A, Y):
        
        self.A = A
        self.Y = Y

        N, C = np.shape(A)
        self.N = N

        A_shifted = A - np.max(A, axis = 1, keepdims=True)
        exp_A = np.exp(A_shifted)
        softmax_A = exp_A/(np.sum(exp_A, axis=1, keepdims=True))
        self.softmax_A = softmax_A

        eps = 1e-10

        H = -Y * np.log(softmax_A + eps)
        L = 1/N * np.sum(H)

        return L
    
    def backward(self):
        
        dLdA = (self.softmax_A - self.Y) / self.N
        return dLdA

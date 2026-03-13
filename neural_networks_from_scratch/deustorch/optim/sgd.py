import numpy as np
class SGD:
    def __init__(self, model, lr = 0.1, momentum = 0):
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum

        self.v_W = [np.zeros(layer.W.shape) for layer in self.l]
        self.v_b = [np.zeros(layer.b.shape) for layer in self.l]
    
    def step(self):
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW 
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb 

        
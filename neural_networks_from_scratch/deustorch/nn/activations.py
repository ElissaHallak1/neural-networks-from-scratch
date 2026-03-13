import numpy as np
class ReLU():
    def __init__(self):
       self.A = None

    def forward ( self , Z ) :
      self . A = np.maximum(0, Z) # compute and store activation
      return self . A
    
    def backward ( self ) :
      dAdZ = (self.A > 0).astype(float) # TODO : compute element - wise derivative
      return dAdZ

class Sigmoid():
    def __init__(self):
        self.A = None
    def forward(self, Z):
       self.A = 1/(1 + np.exp(-Z))
       return self.A
    def backward(self):
       dAdZ = self.A * (1 - self.A)
       return dAdZ

class Tanh():
    def __init__(self):
      self.A = None
    def forward(self, Z):
       self.A = np.tanh(Z)
       return self.A
    def backward(self):
       dAdZ = 1 - self.A * self.A
       return dAdZ
      

    







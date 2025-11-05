import numpy as np
from scipy.optimize import nnls
import time

class NNLSsolver: 
  
    def nonrandom(self):
        
        # observed vector (e.g., pollutant load at C after background removal)
        self.b = np.array([9, 6.0000, 15.0, 24.0, 21.000000])
    
        # signature (or mixing) matrix with two candidate sources (columns = source profiles)
        A = np.array([[1, 2],
                  [2, 1],
                  [3,3],
                  [4,5],
                  [5,4]])
        
        b = np.array([7,5,12,19,17])
    
        # solve nonnegative least squares:  minimize ||A x - b||_2  subject to x >= 0
        x, residual = nnls(A, b)
    
        print(x, residual)
    
    def random(self) :
        
        # set random seed for reproducibility (change or remove for different results each run)
        #np.random.seed(42)
    
        # dimensions
        n_obs = 5   # number of observations (rows)
        n_src = 2   # number of possible sources (columns)
    
        # generate random A (signatures matrix)
        # values between 1 and 10
        A = np.random.randint(1, 10, size=(n_obs, n_src))
    
        # generate random b (observed vector)
        # values between 5 and 30
        b = np.random.randint(5, 30, size=n_obs)
    
        # solve nonnegative least squares: minimize ||A x - b||_2  subject to x >= 0
        x, residual = nnls(A, b)
    
        print("A (signature matrix):\n", A)
        print("b (observed vector):\n", b)
        print("x (source contributions):", x)
        print("residual norm:", residual)

NNLSsolver = NNLSsolver()
NNLSsolver.nonrandom()
NNLSsolver.random()


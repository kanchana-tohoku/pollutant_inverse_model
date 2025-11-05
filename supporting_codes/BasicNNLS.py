import numpy as np

def nnls_basic(A, b, max_iter=1000, alpha=1e-3, tol=1e-6):
    """
    Very basic NNLS solver using projected gradient descent.
    A: m x n matrix
    b: m-vector
    """
    m, n = A.shape
    x = np.zeros(n)   # start with zero
    for it in range(max_iter):
        r = A @ x - b                 # residual
        grad = 10 * A.T @ r           # gradient
        x_new = x - alpha * grad      # gradient step
        x_new = np.maximum(0, x_new)  # project onto x >= 0
        # check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.linalg.norm(A @ x - b)

# Example usage
A = np.array([[1, 2],
              [2, 1],
              [3, 3],
              [4, 5],
              [5, 4]], dtype=float)
b = np.array([9, 6, 15, 24, 21], dtype=float)

x, residual = nnls_basic(A, b)
print("Solution x:", x)
print("Residual norm:", residual)

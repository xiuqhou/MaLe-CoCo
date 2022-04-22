import numpy as np
from numba import jit
## in this file, we define some commonly used utils function including soft-threshold, single-value threshold, and solution process for Sylvester function and Hadamard linear system.

def soft_threshold(x,labda):
    """
    soft threshold operation
    :param x: a matrix
    :param labda: a constant threshold for the given matrix x
    :return y: $y = \mathcal S_\lambda(x)$
    """
    y = np.zeros_like(x)
    y[x > labda] = x[x>labda]-labda
    y[x<-1*labda] = x[x<-1*labda]+labda
    return y

def single_value_threshold(x,tau):
    """
    single value threshold
    :param x: a matrix
    :param tau: a constant threshold for the single value matrix in the svd of the given matrix x
    :return y: $y = \mathcal D_\tau(x) = \mathbf U\mathcal D_\tau(\mathbf\Sigma)\mathbf V^\top$
    """
    n,m = x.shape
    min_ = min(n,m)
    u,sgm,v = np.linalg.svd(x)
    s = np.diag(sgm)
    S = np.zeros_like(x,dtype=np.float)
    S[:min_,:min_] = s
    tmp_S = abs(S)>tau
    tmp_SS = S[tmp_S]
    S[abs(S)<tau] = 0
    S[tmp_S] = np.sign(tmp_SS)*(abs(tmp_SS)-tau)
    z = u.dot(S).dot(v)
    return z

@jit
def Hadamard_Solver(A,B,C):
    """
    A solver for Hadamard linear system, whose equation could be written as  A \circ X + B X = C,
    the solver costs O(nm^3) time complexity,
    ref: https://scicomp.stackexchange.com/questions/31001/solving-a-hadamard-linear-system
    :param A: Hadamard coefficient with size (m, n)
    :param B: matrix product coefficient with size (m, m)
    :param C: right side of the equation, with size (m, n)
    :return X: the solution of the equation
    """

    X = np.zeros_like(A)
    for i in range(A.shape[1]):
        X[:,i] = np.linalg.inv(np.diag(A[:,i]) + B).dot(C[:,i])

    return X

@jit
def solve_sylvester(A,S,W,C):
    X_ = np.zeros_like(C)
    C_ = C.dot(np.ascontiguousarray(W))
    for i in range(C.shape[1]):
        X_[:,i] = np.linalg.inv(A+S[i]*np.identity(A.shape[0])).dot(np.ascontiguousarray(C_[:,i]))
    X = X_.dot(np.linalg.pinv(W))
    return X


def Solve_Sylvester(A,B,C):
    '''
    A solver for sylvester equation: A X + X B = C
    Large A(s,s), Small B(d,d), C(s,d)
    Requirements: C.shape[0]>C.shape[1], that means s > d
    '''
    
    S, W = np.linalg.eig(B)
    S, W = np.real(S), np.real(W)
    X = solve_sylvester(A,S,W,C)
    
    return X


def Hankel(M,n):
    """
    Given a multivariant time series, the function returns the result of n order Hankel embedding
    :param M: the data matrix of given multivariant time series with size (m, s), where m is the spacial dimension and s is temporal dimension.
    :param n: Hankel embedding order
    :return: Hankel embedding result whose size is (nm, s-n+1)
    """
    A = np.array([M[:,i:M.shape[1]-n+i+1]  for i in range(n)])
    B = np.reshape(A,(-1,M.shape[1]-n+1),order='C')
    return B

def deHankel(X_h,n):
    """
    Reverse Hankel embedding
    :param X_h: the given data matrix to reverse Hankel embedding
    :param n: the spacial dimension of the reversed matrix, not the Hankel embedding order!!!
    :return: the reversed data matrix, whose spacial dimension is n
    """
    result = np.zeros((n,int(X_h.shape[0]/n+X_h.shape[1]-1)))
    result = np.array([np.mean([X_h[n*j:n*(j+1),i-j] for j in range(i+1) if(n*(j+1) <= X_h.shape[0] and i-j < X_h.shape[1])],axis=0) for i in range(result.shape[1])])
    return result.T

def rbf(dist,t=1.0):
    return np.exp(-(dist/t))

def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x),1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist

def cal_rbf_dist(data, t = 1):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    rbf_dist = rbf(dist, t)

    W = rbf_dist - np.eye(rbf_dist.shape[0])
    return W

def calculate_relation(x,y):
    return np.sum((x-y)**2)

def calculate_adjacency(M):
    adjacency = np.zeros((M.shape[1],M.shape[1]))
    for i,line1 in enumerate(M.T):
        for j,line2 in enumerate(M.T):
            adjacency[i][j] = calculate_relation(line1, line2)
    adjacency_normalized = np.exp(-adjacency)
    adjacency_normalized -= np.identity(adjacency_normalized.shape[0])
    row_index = np.arange(adjacency_normalized.shape[0]).repeat(adjacency_normalized.shape[0]).reshape(-1,adjacency_normalized.shape[0])
    column_index = row_index.T
    normal_factor = np.exp(-np.abs(row_index-column_index)**2/np.sqrt(adjacency_normalized.shape[0]))
    adjacency_normalized = adjacency_normalized*normal_factor
    return adjacency_normalized

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
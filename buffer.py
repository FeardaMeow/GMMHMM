import scipy.stats as st
import numpy as np
#Set seed
np.random.seed(seed=123)

#Generating data from a 2 state, 2 observation HMM
#A = [0.6, 0.2, 0.2; 0.2, 0.6, 0.2; 0.2, 0.2, 0.6]
#B = [0.6, 0.2, 0.2; 0.2, 0.6, 0.2; 0.2, 0.2, 0.6]
#Pi = [0.4, 0.3, 0.3]

X = np.zeros(10000)
Y = np.zeros(10000)
p = st.uniform.rvs(size=1)[0]

X[0] = 0 if p <= 0.33 else 1 if p <= 0.66 else 2
    
for i in range(X.shape[0]):
    p = st.uniform.rvs(size=1)[0]
    if X[i] == 0:
        Y[i] = np.random.normal(loc=0, scale=1)
        if i < 1999:
            p = st.uniform.rvs(size=1)[0]
            X[i+1] = 0 if p <= 0.005 else 1 if p <= 0.995 else 2
    elif X[i] == 1:
        Y[i] = np.random.normal(loc=0.5, scale=1)
        if i < 1999:
            p = st.uniform.rvs(size=1)[0]
            X[i+1] = 0 if p <= 0.01 else 1 if p <= 0.04 else 2
    else:
        Y[i] = np.random.normal(loc=-0.5, scale=1)
        if i < 1999:
            p = st.uniform.rvs(size=1)[0]
            X[i+1] = 0 if p <= 0.95 else 1 if p <= 0.955 else 2

X = None
#Train HMM on window size or burn in obs length using standard BW algo
pi = np.array([0.4,0.3,0.3])
A = np.array([[0.005,0.99,0.005],[0.01,0.03,0.96],[0.95,0.005,0.045]])
mu = np.array([0,0.5,-0.5])
var = np.array([1,1,1])

alpha = np.zeros([A.shape[0],Y.shape[0]])
B = np.zeros([A.shape[0],Y.shape[0]])

#Forward
e=1
error = 1e-15
a = 0
temp_A = A[0:A.shape[0]-1,0:A.shape[0]-1]
for s in range(A.shape[0]):
    B[s,:] = st.multivariate_normal.pdf(Y, mean=mu[s], cov=var[s])

for t in range(Y.shape[0]):
    if (t==0):
        alpha[:,t] = pi*B[:,t]
        alpha[:,t] /= np.sum(alpha[:,t])
    else:
        alpha[:,t] = B[:,t]*np.dot(alpha[:,t-1],A.T)
        alpha[:,t] /= np.sum(alpha[:,t])
    J = np.zeros([A.shape[0]-1,A.shape[0]-1])
    r = np.log(alpha[:,t]/alpha[:,t][alpha[:,t].shape[0]-1])
    for j in range(J.shape[0]):
        J[j,:] = np.exp(r[0:r.shape[0]-1])*temp_A[:,j]/np.dot(np.exp(r),A[:,j]) - np.exp(r[0:r.shape[0]-1])*A[0:A.shape[0]-1,A.shape[0]-1]/np.dot(np.exp(r),A[:,A.shape[0]-1])
    e *= J
    a += np.log(np.linalg.norm(e))
    e /= np.abs(np.linalg.norm(e))

B = np.log(error)/(a/Y.shape[0])
B
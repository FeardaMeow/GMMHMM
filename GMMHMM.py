import scipy.stats as st
import numpy as np

class gmmhmm:
    
    #Hidden states (n), 
    def __init__(self, n_states):
        self.n_states = n_states

        self.pi = self._norm(np.random.RandomState().rand(self.n_states,1)) #nx1
        self.A = self._rownorm(np.random.RandomState().rand(self.n_states, self.n_states)) #nxn
        
        self.mu = None
        self.covs = None
        self.n_dims = None
        
    def _norm(self, x):
        return (x + (x==0))/np.sum(x + (x==0))
    
    def _rownorm(self, x):
        return ((x+(x==0))/np.sum((x+(x==0)),axis=1, keepdims=True))
    
    def _norm_likelihood(self, obs):
        B = np.zeros((self.n_states, obs.shape[1])) # nxT
        for s in range(self.n_states):
            B[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s])
            #This function can (and will!) return values >> 1
        return B
    
    #method = em, slide
    def train(self, obs, A=None, mu=None, covs=None, n_iter=30, method='em'):
        # obs = (number of featres, time)
        obs = np.atleast_2d(obs)
        if (A is not None):
            self.A = A
            
        if (mu is not None):
            self.mu = mu
            
        if (covs is not None):
            self.covs = covs
        
        self._train_init(obs)
        if method == 'em':
            #Call EM function
            for n in range(n_iter):
                self._em_step(obs)
    
    def _train_init(self, obs):
        #Initializes the starting point for the parameters
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = np.random.choice(np.arange(obs.shape[1]), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
            self.covs += np.diag(np.diag(np.atleast_2d(np.cov(obs))))[:,:,None]
        return(self)
      
    def _forward(self, B):
        alpha = np.zeros(B.shape) # nxT
        for t in range(B.shape[1]):
            if t==0:
                alpha[:,t] = self.pi.ravel()*B[:,t]
                alpha[:,t] /= np.sum(alpha[:,t])
            else:
                alpha[:,t] = B[:,t]*np.dot(alpha[:,t-1],self.A.T)
                alpha[:,t] /= np.sum(alpha[:,t])
        return(alpha)
        
    def _backward(self, B):
        beta = np.zeros(B.shape) #nxT
        beta[:, -1] = np.ones(self.n_dims)
        for t in range(B.shape[1] - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return(beta)
    
    def _em_step(self, obs):
        B = self._norm_likelihood(obs)
        alpha = self._forward(B)
        beta = self._backward(B)
        
        gamma =  alpha*beta #nxT
        gamma /= np.sum(gamma,axis=0)
        
        xi = np.zeros([self.n_states,self.n_states])
        for t in range(obs.shape[1]-1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi += self._norm(partial_sum)
        
        #Update Params
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :] #1xT
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            expected_covs[:,:,s] = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3091008/
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        
        self.A = self._rownorm(xi.T/np.sum(gamma,axis=1))
        self.mu = expected_mu
        self.covs = expected_covs

    def _slide_forward(self, B, alpha_old):
        alpha = np.zeros(B.shape) # nxT
        for t in range(B.shape[1]):
            if t==0:
                alpha[:,t] = B[:,t]*np.dot(alpha_old,self.A.T)
                alpha[:,t] /= np.sum(alpha[:,t])
            else:
                alpha[:,t] = B[:,t]*np.dot(alpha[:,t-1],self.A.T)
                alpha[:,t] /= np.sum(alpha[:,t])
        return(alpha)
        
    def _slide_backward(self, B, beta_old):
        beta = np.zeros(B.shape) #nxT
            
        for t in range(B.shape[1]):
            if t==0:
                beta[:,t] = beta_old/np.sum(np.dot(self.A,B[:,t]))
                beta[:,t] /= np.sum(beta[:, t])
            else:
                beta[:,t] = beta[:,t-1]/np.sum(np.dot(self.A,B[:,t]))
                beta[:,t] /= np.sum(beta[:, t])
        return(beta)

    def _slide_step(self, obs, alpha_old, beta_old, gamma_old, obs_old):
        B = self._norm_likelihood(obs)
        alpha = self._slide_forward(B)
        beta = self._slide_backward(B)
        
        gamma =  alpha*beta #nxT
        gamma /= np.sum(gamma,axis=0)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        gamma_old_sum = np.sum(gamma_old[obs.shape[1]:], axis=1)
        
        xi = np.zeros([self.n_states,self.n_states])
        for t in range(obs.shape[1]):
            if (t==0):
                partial_sum =  self.A * np.dot(alpha_old, (beta[:, t] * B[:, t]).T)
                xi += self._norm(partial_sum)
            else:
                partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t]).T)
                xi += self._norm(partial_sum)
                
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :] #1xT
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            expected_covs[:,:,s] = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        self.A = self.A*(gamma_old_sum/(gamma_state_sum+gamma_old_sum)) + self._rownorm((xi.T/(gamma_old_sum+gamma_state_sum)).T)
        self.mu = self.mu*(gamma_old_sum/(gamma_state_sum+gamma_old_sum)) + expected_mu/(gamma_old_sum+gamma_state_sum)
        self.covs = self.covs*(gamma_old_sum/(gamma_state_sum+gamma_old_sum)) + expected_covs/(gamma_old_sum+gamma_state_sum)[:,None]
        
        return(np.concatenate((gamma_old[:,gamma.shape[1]:],gamma),axis=1))
        
        
#Set seed
np.random.seed(seed=123)

#Generating data from a 2 state, 2 observation HMM
#A = [0.7, 0.3; 0.3, 0.7]
#mu = [-1, 1]
#covs = [0.5,0.5]
#Pi = [0.5, 0.5]

X = np.zeros(2000)
Y = np.zeros(2000)
if st.uniform.rvs(size=1)[0] >= 0.5:
    X[0] = 1
    
for i in range(X.shape[0]):
    p1 = st.uniform.rvs(size=1)[0]

    if X[i] == 0:
        Y[i] = np.random.normal(-1,0.5)
        if i < 1999:
            if p1 > 0.7:
                X[i+1] = 1              
    else:
        Y[i] = np.random.normal(1,0.5)
        if i < 1999:
            if p1 > 0.7:
                X[i+1] = 0

test_hmm = gmmhmm(2)
test_hmm.train(Y, A =np.array([[0.8,0.2],[0.2,0.8]]), mu = np.array([[-1,1]]), covs=np.array([[[0.5,0.5]]]), n_iter = 100)

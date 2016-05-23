import numpy as np

import scipy as sp
from math import pi
import spams
import numpy as np

from kernel_exp_family.kernels.kernels import gaussian_kernel
from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.estimators.lite.gaussian_low_rank import KernelExpLiteGaussianLowRank
from kernel_exp_family.examples.tools import visualise_fit_2d

from scipy.stats import multivariate_normal

from kernel_exp_family.kernels.kernels import rff_feature_map, rff_feature_map_single, \
    rff_sample_basis, rff_feature_map_grad_single, theano_available
from scipy.constants.constants import sigma
    
#methods

#displays A
def visualise_array(Xs, Ys, A, samples=None):
    im = plt.imshow(A, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], 'bx')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])

#compute submatrix, receives i_row, j_row, i_col and j_col and computes $C_{(i_row,j_row),(i_col,j_col)}$
def compute_submatrix(X,allomega,allu,i_row,j_row,i_col,j_col):
    N = X.shape[0]
    m = allu.shape[2]
        
        
        
    submat = np.zeros((m,m))

    X1=(X[:, np.array([i_row,j_row])])
    omega1=((allomega[i_row,j_row,:,:]))
    U1=((allu[i_row,j_row,:]))

    H1 = np.dot(X1,omega1)+U1
    np.sin(H1, H1)
    H1 *= -np.sqrt(2. / m)

    X2=(X[:, np.array([i_col,j_col])])
    omega2=((allomega[i_col,j_col,:,:]))
    U2=((allu[i_col,j_col,:]))


    H2 = np.dot(X2,omega2)+U2
    np.sin(H2, H2)
    H2 *= -np.sqrt(2. / m)    

    if i_row == i_col:
        phi1 = -H1 * omega1[0, :]
        phi2 = -H2 * omega2[0, :]
        submat += np.tensordot(phi1, phi2, [0, 0])

    if i_row == j_col:
        phi1 = -H1 * omega1[0, :]
        phi2 = -H2 * omega2[1, :]
        submat += np.tensordot(phi1, phi2, [0, 0])

    if j_row == i_col:
        phi1 = -H1 * omega1[1, :]
        phi2 = -H2 * omega2[0, :]
        submat += np.tensordot(phi1, phi2, [0, 0])

    if j_row == j_col:
        phi1 = -H1 * omega1[1, :]
        phi2 = -H2 * omega2[1, :]
        submat += np.tensordot(phi1, phi2, [0, 0])


    submat = submat/N
    return submat

#computes submatrices and puts them together to whole C-matrix
def compute_C(X, allomega, allu):
    d = X.shape[1]
    m = allu.shape[2] 
    N = X.shape[0]
    size = (d*(d-1)/2) * m
    C = np.zeros((size, size)) + np.nan
    for i_row in range(d):
        for j_row in range(i_row+1,d):
            row = d*i_row-(i_row+1)*(i_row)/2 + j_row-i_row-1 # row in d^2 meta matrix
            for i_col in range(d):              
                for j_col in range(i_col+1,d):
                    col = d*i_col-(i_col+1)*(i_col)/2 + j_col-i_col-1 # col in d^2 meta matrix
                    if col>row:
                        continue
                    submat = compute_submatrix(X, allomega, allu, i_row,j_row,i_col,j_col)
                    C[row*m:row*m+m, col*m:col*m+m] = submat
                    if row != col:
                        C[col*m:col*m+m, row*m:row*m+m] = submat.T
    C =C +0.1*np.eye(size)
    return C

#computes $b_{(i,j)}$`s and puts them together to b-vector
def compute_b(X, allomega, allu):
    N = X.shape[0]
    d = X.shape[1]
    m = allu.shape[2] 
    size = (d*(d-1)/2) * m
    b = np.zeros(size)+ np.nan
    for i in range(d):
        for j in range(i+1,d):
            XH = (X[:, np.array([i,j])])
            omegaH = (allomega[i,j,:,:])
            uH =(allu[i,j,:])
            
            Phi1 = np.dot(XH,omegaH)+uH
            np.sin(Phi1, Phi1)
            Phi1 *= -np.sqrt(2. / m) 
            
            
            Phi2 = rff_feature_map(XH, omegaH, uH)

            projections_sum = np.zeros(m)   
            for k in range(2):
                bandwidth = (4.0/3.0/N)**(1.0/5.0)
                estimate_var = np.var(XH[:,k])    
                projections_sum += np.mean(-Phi2 * (omegaH[k, :] ** 2), axis=0) 
                
                #print for comparison
                LGD1 = XH[:,k].reshape(-1,1)/estimate_var
                LGD2 = learn_1d_log_grad(XH[:,k],XH[:,k],bandwidth).reshape(-1,1)
                print "true log gradient density", LGD1[1:10]
                print "learned log gradient density", LGD2[1:10]

                #use this if you know that all marginal densities are gaussian
                projections_sum += (np.mean(-Phi1*omegaH[k, :] * LGD1, axis=0))
                
                #use this for KDE 
                #projections_sum += (np.mean(-Phi1*omegaH[k, :] * LGD2, axis=0))
                

                
            phi = -projections_sum
            b[m*(d*i-(i+1)*(i)/2 + j-i-1):m*(d*i-(i+1)*(i)/2 + j-i-1)+m] = phi
    return b

def learn_1d_density(Y,X,bandwidth):#Y:evaluate on Y, X: data
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    K = gaussian_kernel(Y,X,bandwidth)
    est_dens = np.mean((K), axis=1)

    return est_dens


def learn_1d_gradient(Y, X, bandwidth): #Y:evaluate on Y, X: data
    Y = Y.reshape(-1,1)

    X = X.reshape(-1,1)
    

    k = gaussian_kernel(Y, X, bandwidth)

    
    Yh = np.repeat(Y, len(X),axis = 1)
    Xh = np.repeat(X, len(Y),axis = 1)


    differences = -Yh.T+Xh

    G = (2.0 / bandwidth) * (np.dot(k ,differences))
    G = np.diag(G)/len(X)
    return -G

def learn_1d_log_grad(Y,X,bandwidth):#Y:evaluate on Y, X: data
    Y = Y.reshape(-1,1)
    X = X.reshape(-1,1)

    D = learn_1d_density(Y,X,bandwidth)
    G = learn_1d_gradient(Y, X, bandwidth)
    
    
    LG = G/D
    return LG

#turn b from long vector into array
def compute_allb(b,d,m):
    allb = np.zeros((d,d,m))
    counter = 0
    for i in range(d):  
        for j in range(i+1,d):
            allb[i,j,:] = b[counter*m: counter*m+m]
            counter += 1
    return allb

#turn C from big matrix into array with matrices
def compute_allC(C,d,m):
    allC = np.zeros((d,d,m,m))
    counter = 0
    for i in range(d):  
        for j in range(i+1,d):
            allC[i,j,:,:] = C[counter*m: counter*m+m, counter*m: counter*m+m]
            counter += 1
    return allC

#turn theta from long vector into array
def compute_alltheta(theta,d,m):
    alltheta = np.zeros((d,d,m))
    counter = 0
    for i in range(d):  
        for j in range(i+1,d):
            alltheta[i,j,:] = theta[counter*m: counter*m+m]
            counter += 1
    return alltheta  

#compute logpdf of a point, given vector theta, omega and u
def log_pdf_2d(x, theta, omega,u):
    phi = rff_feature_map_single(x, omega, u)
    return np.dot(theta,phi)

#compute logpdf gradient of a point, given vector theta, omega and u
def log_pdf_gradient_2d(x, theta, omega,u):
    phi = rff_feature_map_grad_single(x, omega, u)
    return np.dot(phi,theta)

# extracts C_((1,2),(1,2)) for a simple test
#just for tests
def extract_simple_C(C,m,d): #just for tests
    simple_C = C[0:m, 0:m]
    return simple_C

# extracts b_(1,2) for a simple test
# just for tests
def extract_simple_b(b,m,d): #just for tests
    simple_b = b[0:m]
    return simple_b

# solves Y = w^t X , given groupstructure as vector G and regularisation parameter lam    
def group_lasso(X,Y, G, lam):
    myfloat = np.float64

    param = {'numThreads' :-1, 'verbose' : True,
    'lambda1' : lam, 'it0' : 10, 'max_it' : 200,
    'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
    'pos' : False}

    #size = C.shape[1]
    #normalise problem
    X = np.asfortranarray(X - np.tile(np.mean(X, 0), (X.shape[0], 1)), dtype=myfloat)
    X = spams.normalize(X)
    Y = np.asfortranarray(Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1)), dtype=myfloat)
    Y = spams.normalize(Y)

    W0 = np.zeros((X.shape[1], Y.shape[1]), dtype=myfloat, order="FORTRAN")


    param['compute_gram'] = True
    param['loss'] = 'square'

    print '\nFISTA + Group Lasso L2 with variable size of groups'
    param['regul'] = 'group-lasso-l2'

    param['groups'] = np.array(G, dtype=np.int32)#group structure
    param['lambda1'] *= 10
    (W, optim_info) = spams.fistaFlat(Y, X, W0, True, **param)#optimisation
    
    
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' % (np.mean(optim_info[0, :], 0), np.mean(optim_info[2, :], 0), np.mean(optim_info[3, :], 0))

    return W

#computes grid of log_pdfs for set of points Xs, Ys
def pdf_grid(Xs,Ys, theta, omega, u): 

    from scipy.stats import multivariate_normal 

    L = np.zeros((len(Xs), len(Ys)))

    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])

            L[j, i] = log_pdf_2d(point, theta, omega, u)


    return L

#computes grid of log_pdf_gradients for set of points Xs, Ys
def pdf_grid_gradient(Xs,Ys,theta,omega,u): 


    L = np.zeros((len(Xs), len(Ys)))


    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
                
            L[j, i] = np.linalg.norm(log_pdf_gradient_2d(point, theta, omega, u))


    return L

#plots solution for two dimensions i,j
def visualise_fit_structured_densities_two_dimensions(X, alltheta,allomega,allu, i,j,Xs=None, Ys=None):

    plt.figure()
    if Xs is None:
        Xs = np.linspace(-7.5,7.5)

    if Ys is None:
        Ys = np.linspace(-7.5, 7.5)

    L = pdf_grid(Xs, Ys,alltheta[i,j,:],allomega[i,j,:,:],allu[i,j,:]) #only L useful
     
      
    #plt.subplot(221)
    #visualise_array(Xs, Ys, L,X[:, np.array([i,j])])
    #plt.title("log pdf learned with data")
        
    #plt.subplot(222)
    #visualise_array(Xs, Ys, L)
    #plt.title("log pdf learned")
    
    #plt.tight_layout()
    #    plt.savefig("plotlogpdf.eps")

    L2 = pdf_grid_gradient(Xs, Ys,alltheta[i,j,:],allomega[i,j,:,:],allu[i,j,:])
    plt.subplot(221)
    visualise_array(Xs, Ys, L2, X[:, np.array([i,j])])
    plt.title("gradient log pdf learned \\ with data")
        
    plt.subplot(222)
    visualise_array(Xs, Ys, L2)
    plt.title("gradient log pdf learned")


    plt.tight_layout()
    #    plt.savefig("plotgradientlogpdf.eps")

# creates groupstructure so that each theta_(i,j) is one group
def create_groupstructure(d,m):
    G1 = np.arange(1, 1+ d*(d-1)/2)
    G = np.repeat(G1,m)
    return G


# gets set of data points X and a regularisation parameter lamb, computes array containong all theta_(i,j)    
def fit_theta(X,allomega,allu,lam):
    N = X.shape[0]
    d = X.shape[1]
    m = allu.shape[2]
    C = compute_C(X, allomega, allu)
    b = compute_b(X, allomega, allu)
    
    #execute group lasso    
    size = C.shape[1]
    Xh = C+0.1*np.eye(size)
    Yh = b.reshape(-1,1)
    G = create_groupstructure(d,m)
    w = group_lasso(Xh,Yh, G, lam)
    w = np.squeeze(w.T)
    
    return w
    

def fit_alltheta(X,allomega,allu,lam):
    N = X.shape[0]
    d = X.shape[1]
    m = allu.shape[2]
    w = fit_theta(X,allomega,allu,lam)
    alltheta = compute_alltheta(w,d,m)
    
    return alltheta  
    
# gets set of datapoints X and learns and plots thetas


# gets set of data points X and a regularisation parameter lambda, computes array containing all theta_(i,j)

def log_pdf_single(x, alltheta,allomega,allu):
    d = allu.shape[0]
    ret = 0
    for i in range(d):
        for j in range(i, d):
            theta = alltheta[i,j,:]
            omega = allomega[i,j,:,:]
            u = allu[i,j,:]
            ret += log_pdf_2d(x, theta, omega,u)
    return ret
            
        
def log_pdf_gradient_single(x, alltheta,allomega,allu):
    d = allu.shape[0]
    ret = 0
    for i in range(d):
        for j in range(i,d):
            theta = alltheta[i,j,:]
            omega = allomega[i,j,:,:]
            u = allu[i,j,:]
            ret += log_pdf_gradient_2d(x, theta, omega,u)    
    return ret

def objective(X, theta, allomega, allu, b=None, C=None):
    if b is None:
        b = compute_b(X, allomega, allu)
    
    if C is None:
        C = compute_C(X, allomega, allu)

    return 0.5 * np.dot(theta, np.dot(C, theta)) - np.dot(theta, b)

#requires datamatrix, computes crossvalidated objective function, 
def crossvalidate_objective(Xmat,sigma, lam,m): #Xmat: CN*N*d-matrix
    CN = Xmat.shape[0]
    N = Xmat.shape[1]
    d = Xmat.shape[2]
    ret = 0
    for k in range(CN):#validate on Xk
        counter =0
        for l in range(CN):
            if k != l:
                if counter == 0:
                    Xlearn = Xmat[l,:,:]
                    counter = 1
                else:
                    Xlearn = np.concatenate((Xlearn,Xmat[l,:,:]),axis =0)
        Xtest = Xmat[k,:,:]
        
        #learn theta
        allomegahelp = np.zeros((d,d,2,m))
        alluhelp = np.zeros((d,d,m))
        gammahelp = 1./sigma
        for i in range(d):
            for j in range(d):       
                allomegahelp[i,j,:,:] = gammahelp * np.random.randn(2, m)        
                alluhelp[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
        theta = fit_theta(Xlearn, allomegahelp, alluhelp, lam)
        
        #compute objective 
        allomegahelp = np.zeros((d,d,2,m))
        alluhelp = np.zeros((d,d,m))
        gammahelp = 1./sigma
        for i in range(d):
            for j in range(d):       
                allomegahelp[i,j,:,:] = gammahelp * np.random.randn(2, m)        
                alluhelp[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
        obj = objective(Xtest, theta, allomegahelp, alluhelp, b=None, C=None)
        ret += obj
    ret + ret/CN
    return ret
    
    
    
    
    

class KernelExpStructuredGaussianMarginals(EstimatorBase):
    def __init__(self,m,sigma,lam):
        self.m = m
        self.sigma =sigma  
        self.lam =lam
        
    def fit(self, X):
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.allomega = np.zeros((self.d,self.d,2,self.m))
        self.allu = np.zeros((self.d,self.d,self.m))
        self.gamma = 1./self.sigma
        for i in range(self.d):
            for j in range(self.d):       
                self.allomega[i,j,:,:] = self.gamma * np.random.randn(2, self.m)        
                self.allu[i,j,:] = np.random.uniform(0, 2 * np.pi, self.m)
        self.alltheta = fit_alltheta(X,self.allomega,self.allu,self.lam)
        self.theta = fit_theta(X,self.allomega,self.allu,self.lam)
        
        
    def log_pdf(self, x):
        log_pdf = log_pdf_single(x, self.alltheta,self.allomega,self.allu)
        return log_pdf
    
    def grad(self, x):
        log_pdf_gradient = log_pdf_gradient_single(x, self.alltheta,self.allomega,self.allu)
        return log_pdf_gradient
    
    def log_pdf_multiple(self, X):
        
        ret = np.zeros((self.N, self.m))
        
        for i in range(self.d):
            for j in range(i, self.d):
                omega = self.allomega[i,j,:,:]
                u = self.allu[i,j,:]
                Phi = rff_feature_map(X, omega, u)
                theta = self.alltheta[i,j,:]
                theta = theta.reshape(-1,1)
                
                temp = np.dot(Phi, theta)
                temp2 = ret
                
                ret += np.dot(Phi, theta)
        return  ret
        
    def objective(self, X):
        
        return objective(X, self.theta, self.allomega, self.allu, b=None, C=None)
    
    def crossvalidate_objective(self,Xmat):
        return crossvalidate_objective(Xmat,self.sigma, self.lam,self.m)
        
        
        
        
        
        
        

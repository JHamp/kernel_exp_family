
import numpy as np
import matplotlib.pyplot as plt

from kernel_exp_family.estimators.structured_densities.gaussian_with_marginals import KernelExpStructuredGaussianMarginals
from kernel_exp_family.examples.tools import visualise_fit_2d

N = 200
D = 2

# fit model to samples from a standard Gaussian

CN =5
Xmat = np.zeros((CN,N,D))
for i in range(CN):
    Xmat[i,:,:] = np.random.randn(N, D)

X = Xmat[1,:,:]
for i in range(1,CN):
    X = np.concatenate((X,Xmat[i,:,:]),axis =0)
#X: needed for computing of theta and plotting
#Xmat: needed for crossvalidation

sigma = 2
lmbda = 0.01
m = N
est = KernelExpStructuredGaussianMarginals(m,sigma, lmbda)

est.fit(X)

# main interface for log pdf and gradient
#print est.log_pdf_multiple(np.random.randn(2, 2))
#print est.log_pdf(np.zeros(D))
#print est.grad(np.zeros(D))

# score matching objective function (can be used for parameter tuning)
visualise_fit_2d(est, X)
plt.show()
print est.crossvalidate_objective(Xmat)
print "is the crossvalidated objective."
print est.objective(X)
print "is the objective."
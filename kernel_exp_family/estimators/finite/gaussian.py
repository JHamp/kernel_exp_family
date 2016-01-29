from choldate._choldate import cholupdate

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.kernels.kernels import rff_feature_map, rff_feature_map_single, \
    rff_sample_basis, rff_feature_map_grad_single, theano_available
from kernel_exp_family.tools.assertions import assert_array_shape,\
    assert_array_non_negative
from kernel_exp_family.tools.numerics import log_sum_exp
import numpy as np
import scipy as sp


if theano_available:
    from kernel_exp_family.kernels.kernels import rff_feature_map_comp_hessian_theano, \
    rff_feature_map_comp_third_order_tensor_theano
    
def compute_b(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections_sum += np.mean(-Phi2 * (omega[d, :] ** 2), axis=0)
        
    return -projections_sum

def compute_C(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    C = np.zeros((m, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C += np.tensordot(temp, temp, [0, 0])

    return C / N

def update_b(X, b, n, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections_sum += np.sum(-Phi2 * (omega[d, :] ** 2), 0)
        
    b_new_times_N = -projections_sum
    N = len(X)
    return (b * n + b_new_times_N) / (n + N)

def update_L_C(X, L_C, n, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    # unscale
    L_C *= np.sqrt(n)
    
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        for u in temp:
            cholupdate(L_C, u)

    # since cholupdate works on transposed version
    L_C = L_C.T
    
    # since the new C has a 1/(n+len(X)) term in it
    L_C /= np.sqrt(n + len(X))

    return L_C

def compute_C_weighted(X, omega, u, weights):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    X_weighted = (X.T * weights).T
    
    C = np.zeros((m, m))
    projection = np.dot(X_weighted, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C += np.tensordot(temp, temp, [0, 0])

    return C / np.sum(weights)

def update_L_C_weighted(X, L_C, sum_weights, omega, u, weights):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    X = (X.T * weights).T
    
    # unscale
    L_C *= np.sqrt(sum_weights)
    
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        for u in temp:
            cholupdate(L_C, u)

    # since cholupdate works on transposed version
    L_C = L_C.T
    
    # since the new C has a 1/(n+sum(weights)) term in it
    L_C /= np.sqrt(sum_weights + np.sum(weights))

    return L_C

def compute_b_weighted(X, omega, u, weights):
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]

    X_weighted = (X.T * weights).T

    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X_weighted, omega, u)
    for d in range(D):
        projections_sum += np.sum(-Phi2 * (omega[d, :] ** 2), axis=0)
        
    return -projections_sum / np.sum(weights)

def update_b_weighted(X, b, sum_weights_old, omega, u, weights):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    X_weighted = (X.T * weights).T
    
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X_weighted, omega, u)
    for d in range(D):
        projections_sum += np.sum(-Phi2 * (omega[d, :] ** 2), 0)
        
    b_new_times_sum_weights = -projections_sum
    sum_weights = np.sum(weights)
    return (b * sum_weights_old + b_new_times_sum_weights) / (sum_weights_old + sum_weights)

def fit(X, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
    
    if C is None:
        C = compute_C(X, omega, u)
    
    theta = np.linalg.solve(C, b)
    return theta

def fit_L_C_precomputed(b, L_C):
    theta = sp.linalg.cho_solve((L_C, True), b)
    return theta

def objective(X, theta, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    return 0.5 * np.dot(theta, np.dot(C, theta)) - np.dot(theta, b)

def update_C(x, C, n, omega, u):
    D = omega.shape[0]
    assert x.ndim == 1
    assert len(x) == D
    m = 1 if np.isscalar(u) else len(u)
    N = 1
    
    C_new = np.zeros((m, m))
    projection = np.dot(x[np.newaxis, :], omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C_new += np.tensordot(temp, temp, [0, 0])
    
    # Knuth's running average
    n = n + 1
    delta = C_new - C
    C += delta / n
    
    return C

class KernelExpFiniteGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, m, D):
        self.sigma = sigma
        self.lmbda = lmbda
        self.m = m
        self.D = D
        self.omega, self.u = rff_sample_basis(D, m, sigma)
        
        # zero actual data, different from n_with_fake data below
        self.n = 0
        
        # self.sum_weights is number of data and fake data if weights are all 1
        self.b, self.L_C, self.sum_weights = self._gen_initial_solution()
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
    def supports_update_fit(self):
        return True
    
    def supports_weights(self):
        return True
    
    def _gen_initial_solution(self):
        # components of linear system, stored for online updating
        b_fake = np.zeros(self.m)
        L_C_fake = np.eye(self.m) * np.sqrt(self.lmbda)
        
        # assume have observed m terms, which is needed for making the system well-posed
        # the above L_C says that the m terms had covariance self.lmbda
        # the above b says that the m terms had mean 0
        n_fake = self.m
        
        return b_fake, L_C_fake, n_fake
    
    def fit(self, X, weights=None):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        N = len(X)
        
        if weights is None:
            weights = np.ones(N)
        assert_array_shape(weights, ndim=1, dims={0: N})
        assert_array_non_negative(weights)
        
        # initialise solution
        b_fake, L_C_fake, n_fake = self._gen_initial_solution()
        
        # "update" initial "fake" solution in the way the it is the same as repeated updating
        sum_weights = np.sum(weights)
        new_sum_weights = n_fake + sum_weights
        self.b = (b_fake * n_fake + compute_b_weighted(X, self.omega, self.u, weights) * sum_weights) / new_sum_weights
        C = (np.dot(L_C_fake, L_C_fake.T) * n_fake + compute_C_weighted(X, self.omega, self.u, weights) * sum_weights) / new_sum_weights
        self.L_C = np.linalg.cholesky(C)
        self.sum_weights = new_sum_weights
        self.n = N
        
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
    def update_fit(self, X, weights=None):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        N = len(X)
        
        if weights is None:
            weights = np.ones(N)
        assert_array_shape(weights, ndim=1, dims={0: N})
        
        self.b = update_b_weighted(X, self.b, self.sum_weights, self.omega, self.u, weights)
        self.L_C = update_L_C_weighted(X, self.L_C, self.sum_weights, self.omega, self.u, weights)
        self.n += len(X)
        self.sum_weights += np.sum(weights)
        
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
    def log_pdf(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        phi = rff_feature_map_single(x, self.omega, self.u)
        return np.dot(phi, self.theta)
    
    def grad(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        
        grad = rff_feature_map_grad_single(x, self.omega, self.u)
        return np.dot(grad, self.theta)
    
    if theano_available:
        def hessian(self, x):
            """
            Computes the Hessian of the learned log-density function.
            
            WARNING: This implementation slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            H = np.zeros((self.D, self.D))
            for i, theta_i in enumerate(self.theta):
                H += theta_i * rff_feature_map_comp_hessian_theano(x, self.omega[:, i], self.u[i])
        
            # RFF is a monte carlo average, so have to normalise by np.sqrt(m) here
            return H / np.sqrt(self.m)
        
        def third_order_derivative_tensor(self, x):
            """
            Computes the third order derivative tensor of the learned log-density function.
            
            WARNING: This implementation is slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            G3 = np.zeros((self.D, self.D, self.D))
            for i, theta_i in enumerate(self.theta):
                G3 += theta_i * rff_feature_map_comp_third_order_tensor_theano(x, self.omega[:, i], self.u[i])
        
            # RFF is a monte carlo average, so have to normalise by np.sqrt(m) here
            return G3 / np.sqrt(self.m)
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        Phi = rff_feature_map(X, self.omega, self.u)
        return np.dot(Phi, self.theta)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # note we need to recompute b and C here
        return objective(X, self.theta, self.omega, self.u)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
    
    def set_parameters_from_dict(self, param_dict):
        EstimatorBase.set_parameters_from_dict(self, param_dict)
        
        # update basis
        self.omega, self.u = rff_sample_basis(self.D, self.m, self.sigma)

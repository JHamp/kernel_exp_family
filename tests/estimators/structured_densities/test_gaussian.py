#tests
import os
from nose import SkipTest
from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose
import numpy as np
from kernel_exp_family.estimators.structured_densities.gaussian import compute_submatrix, compute_C, \
    compute_b, compute_allb, compute_allC, compute_alltheta,\
    create_groupstructure, extract_simple_C, extract_simple_b, KernelExpStructuredGaussian
from kernel_exp_family.kernels.kernels import rff_feature_map


def test_alltheta_correctly_extracted_from_theta_vector():
    N=20
    d=2
    m=2
    size_theta = m*d*(d+1)/2
    
    theta = np.random.randn(size_theta)
    alltheta = compute_alltheta(theta,d,m)
    theta2 = np.concatenate([alltheta[0,0,:],alltheta[0,1,:],alltheta[1,1,:]])
    assert np.all(theta == theta2)


def test_allb_correctly_extracted_from_b_vector():
    N=20
    d=2
    m=2
    size_b = m*d*(d+1)/2
    
    b = np.random.randn(size_b)
    allb = compute_allb(b,d,m)
    b2 = np.concatenate([allb[0,0,:],allb[0,1,:],allb[1,1,:]])
    assert np.all(b == b2)


def test_allC_correctly_extracted_from_C_matrix():
    d=2
    m=2
    size_C = m*d*(d+1)/2
    C =np.random.randn(size_C,size_C)
    allC = compute_allC(C,d,m)
    
    counter = 0
    for i in range(d):
        for j in range(i,d):
            C_help = C[m*counter: m*(counter+1), m*counter: m*(counter+1)]
            assert np.all(allC[i,j,:,:]==C_help)
            counter +=1 
            

    
    

def test_all_group_numbers_positive():
    d = 10
    m =20
    G = create_groupstructure(d,m)
    assert np.all(G>0)


def test_C_12_correct():
    N=20
    d=2
    m=2
    sigma =1
    X = np.random.randn(N, d)

    size = (d*(d+1)/2) * m



    #create feature map parameters
    allomega = np.zeros((d,d,2,m))
    allu = np.zeros((d,d,m))

    for i in range(d):
    #for i in [1]:    
        for j in range(d):
        #for j in [0]:    

            gamma = 1./sigma

            allomega[i,j,:,:] = gamma * np.random.randn(2, m)

            allu[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
            
    C = compute_C(X,allomega,allu)
    C_simple = extract_simple_C(C,m,d)
    
    omega = allomega[0,1,:,:]
    u = allu[0,1,:]
    
    #Code from Heiko
    C_old = np.zeros((m, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for p in range(d):
        temp = -projection * omega[p, :]
        C_old += np.tensordot(temp, temp, [0, 0])
    C_old = C_old/N
    
    assert_allclose(C_simple, C_old)


def test_b_12_correct():
    N=200
    d=2
    m=2
    sigma =1
    X = np.random.randn(N, d)

    size = (d*(d+1)/2) * m



    #create feature map parameters
    allomega = np.zeros((d,d,2,m))
    allu = np.zeros((d,d,m))

    for i in range(d):
    #for i in [1]:    
        for j in range(d):
        #for j in [0]:    

            gamma = 1./sigma

            allomega[i,j,:,:] = gamma * np.random.randn(2, m)

            allu[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
            
    b = compute_b(X,allomega,allu)
    b_simple = extract_simple_b(b,m,d)
    
    omega = allomega[0,1,:,:]
    u = allu[0,1,:]
    #code from heiko
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X, omega, u)
    for p in range(d):
        projections_sum += np.mean(-Phi2 * (omega[p, :] ** 2), axis=0)          
    
    b_old=-projections_sum
    
    assert_allclose(b_simple, b_old)


def test_b_has_no_NaN():
    N=20
    d=2
    m=2
    sigma =1
    X = np.random.randn(N, d)

    size = (d*(d+1)/2) * m



    #create feature map parameters
    allomega = np.zeros((d,d,2,m))
    allu = np.zeros((d,d,m))

    for i in range(d):
    #for i in [1]:    
        for j in range(d):
        #for j in [0]:    

            gamma = 1./sigma

            allomega[i,j,:,:] = gamma * np.random.randn(2, m)

            allu[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
    b = compute_b(X, allomega, allu)
    assert not np.any(np.isnan(b))

def test_C_has_no_NaN():
    N=20
    d=2
    m=2
    sigma =1
    X = np.random.randn(N, d)
    # create C

    size = (d*(d+1)/2) * m


    #create feature map parameters
    allomega = np.zeros((d,d,2,m))
    allu = np.zeros((d,d,m))

    for i in range(d):
    #for i in [1]:    
        for j in range(d):
        #for j in [0]:    

            gamma = 1./sigma

            allomega[i,j,:,:] = gamma * np.random.randn(2, m)

            allu[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
    C = compute_C(X, allomega, allu)
    assert not np.any(np.isnan(C))


def test_C_is_symmetric():
    N=20
    d=2
    m=2
    sigma =1
    X = np.random.randn(N, d)
    # create C

    size = (d*(d+1)/2) * m


    #create feature map parameters
    allomega = np.zeros((d,d,2,m))
    allu = np.zeros((d,d,m))

    for i in range(d):
    #for i in [1]:    
        for j in range(d):
        #for j in [0]:    

            gamma = 1./sigma

            allomega[i,j,:,:] = gamma * np.random.randn(2, m)

            allu[i,j,:] = np.random.uniform(0, 2 * np.pi, m)
    C = compute_C(X, allomega, allu)
    assert_allclose(C, np.transpose(C))
    
def test_oop_interface():
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    
    sigma = 2
    lmbda = 1
    m = 100
    est = KernelExpStructuredGaussian(sigma, lmbda, m)
    
    est.fit(X)
    
    # main interface for log pdf and gradient
    
    #print est.log_pdf_multiple(np.random.randn(2, 2))
    print est.log_pdf(np.zeros(D))
    print est.grad(np.zeros(D))
    
    # score matching objective function (can be used for parameter tuning)
    print est.objective(X)
    
if __name__ == "__main__":
    test_oop_interface()

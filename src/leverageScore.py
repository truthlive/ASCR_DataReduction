import numpy as np
import scipy 
from numpy.random import default_rng
import scipy.linalg as sla
from frequentDirections import FrequentDirections
import time
import h5py
# from errorEstimator import *
# from rSVDsp import truncateSVD

import argparse
# from plot_gif import make_gif_matrix
import multiprocessing

def det_ridge_leverage(A, k, epsilon, plot=False, without_replacement=False, rs=32342342):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d, d>> n
    k is the rank of the projection with theoretical guarantees.
    choose epsilon and delta to be less than one.
    """
    AAt = A.dot(A.T)
    U, eig, Ut = scipy.linalg.svd(AAt)
    print(AAt.shape, eig[k:].shape)
    print('fraction of frob norm captured by k', np.sum(eig[0:k]/np.sum(eig) ))
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel = U.dot( np.diag(1/(eig + AnotkF2/k))).dot(Ut) #! ridge leverage 
    # ridge_kernel = U[:, 0:k].dot( np.diag(1/(eig[0:k]))).dot(U[:, 0:k].T) #! ksub leverage 

    #ridge_kernel = scipy.linalg.inv(AAt + AnotkF2/k *np.diag(np.ones(AAt.shape[0])))
    tau = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        tau[i] = A[:, i].T.dot(ridge_kernel).dot(A[:, i])

    tau_tot = np.sum(tau)
    print(2 * k, tau_tot)
    tau_sorted = np.sort(tau)[::-1]
    idx_sorted = np.argsort(tau)[::-1]
    

    add = 0.
    theta = 0
    for i in range(A.shape[1]):
        add = add + tau_sorted[i]
        if add > epsilon:
            theta = i
            break

    # theta = (len(tau_sorted) - np.sum(tau_sorted_sum> tau_tot - epsilon)+1)[0]
    # if theta < k:
    #     theta = k
    # index_keep = tau_sorted.index[0:theta]
    index_keep = idx_sorted[0:theta]
    # index_drop = tau_sorted.index[theta:]
    index_drop = idx_sorted[theta:]

    return(theta, index_keep, tau, tau_sorted, index_drop, tau_tot, AnotkF2)

def random_ridge_leverage(A, k, theta, plot=False, without_replacement=False, rs=32342342):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d, d>> n
    k is the rank of the projection with theoretical guarantees.
    choose epsilon and delta to be less than one.
    """
    AAt = A.dot(A.T)
    U, eig, Ut = sla.svd(AAt)
    print(AAt.shape, eig[k:].shape)
    AnotkF2 = np.sum(eig[k:])
    ridge_kernel = U.dot( np.diag(1/(eig + AnotkF2/k))).dot(Ut)
    tau =np.zeros((A.shape[1]))
    for i in range(A.shape[1]):
        tau[i] = A[:, i][:, None].T.dot(ridge_kernel).dot(A[:, i])
    #U, sing, Vt = scipy.sparse.linalg.svds(A)
    #tau = pd.DataFrame(tau)
    #print(tau)
    tau_tot = np.sum(tau)
    print('number of columns to sample', theta)
    p = tau/tau_tot
    print('pshape', p.shape)
    if without_replacement:
        q=np.minimum(p * theta *12 , np.ones(p.shape))
        R= scipy.stats.bernoulli.rvs(q, random_state=rs)
        index_keep = np.array(range(A.shape[1]))[(R != 0)]
        C=(A/np.sqrt(q))[:, index_keep]
        print(C.shape)
    else:
        R = scipy.stats.multinomial.rvs(theta, p, size=1, random_state=rs)
        R=R[0, :]
        print('number of distinct columns sampled ', np.sum(R != 0))
        index_keep = np.array(range(A.shape[1]))[(R != 0)]
        C = np.zeros((A.shape[0], theta))
        #D= np.zeros((A.shape[0], np.sum(R != 0)))
        counter=0
        #counterD=0
        for i in range(A.shape[1]):
            if R[i]!=0: 
                # D[:, counterD] = np.sqrt(R[i]) * A[:, i] /np.sqrt(t* p[i])
                # counterD=counterD+1
                for j in range(R[i]):
                    C[:, counter] =  A[:, i] /np.sqrt(theta* p[i])
                    counter = counter +1  
    return(C, R, index_keep)

def rID_streaming_ridge_leverage_old(A, k, t, epsilon, delta, c, os, rng = default_rng()):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d
    k is the rank of the projection with theoretical guarantees.
    t is the stored column size
    epsilon: accuracy parameter
    delta: (1-delta) is the success probability
    c: oversampling parameter
    os: oversampling number for random projection
    choose epsilon and delta to be less than one.
    """
    n, d = np.shape(A)
    ll = k + os
    Omg = rng.standard_normal(size=(ll, n))

    # t = np.ceil(32. * c * (k*np.log(k)+k*np.log(1./delta)/epsilon)).astype(int)
    # print(t)


    count = 0
    nBlock = 0
    C = np.zeros((n, t)) # Stores actual column subset
    D = np.zeros((n, t)) # Stores a queue of new columns
    frobA = 0

    Q = np.zeros((t, d))
    C_old = C
    OmgA = np.zeros((ll, d))

    tau_old = np.ones((t)) # Initialize sampling probabilities
    tau = tau_old
    l = 2  # parameter for FrequentDirections

    sketcher = FrequentDirections(n, (l+1)*k)
    for i in range(d):
        a = A[:,i].T
        sketcher.append(a)
        B = sketcher.get().T
        OmgA[:,i] = Omg @ A[:,i]
        # print(np.shape(B))
        if count < t:
            D[:, count] = A[:,i]
            frobA = frobA + sla.norm(a)**2
            count = count + 1
        else:
            # print(B)
            tau = np.minimum(tau_old, ApproximateRidgeScores(B, C, frobA, k)) 
            tau_D = ApproximateRidgeScores(B, D, frobA, k)
            # print(tau)
            # print(tau_D)
            for j in range(t):
                if not np.all(C[:,j] == 0):
                    prob_rej = 1. - tau[j]/tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:,j] = 0
                        tau_old[j] = 1.
                    else:
                        tau_old[j]=tau[j]
                if np.all(C[:,j] == 0):
                    for l in range(t):
                        prob = tau_D[l]*c*(k*np.log(k)+k*np.log(1./delta)/epsilon)/t
                        roll = rng.random()
                        if roll < prob:
                            C[:,j] = D[:, l]
                            tau_old[j] = tau_D[l]
            count = 0

    idx = np.argwhere(np.all(C[..., :] == 0, axis=0))
    C = np.delete(C, idx, axis=1)

    print(np.shape(C))

    # Q = sla.lstsq(Omg @ C, OmgA)[0]
    Q = sla.lstsq(C, A)[0]
    return C, Q

def rID_streaming_ridge_leverage_update(A, k, t, epsilon, delta, c, os, rng = default_rng()):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d, d>> n
    k is the rank of the projection with theoretical guarantees.
    t is the stored column size
    epsilon: accuracy parameter
    delta: (1-delta) is the success probability
    c: oversampling parameter
    os: oversampling number for random projection
    choose epsilon and delta to be less than one.
    """
    n, d = np.shape(A)
    ll = k + os
    Omg = rng.standard_normal(size=(ll, n))


    count = 0
    nBlock = 0
    C = np.zeros((n, t)) # Stores actual column subset
    D = np.zeros((n, t)) # Stores a queue of new columns
    frobA = 0

    Q = np.zeros((t, d))
    C_old = C
    OmgA = np.zeros((ll, d))

    tau_old = np.ones((t)) # Initialize sampling probabilities
    tau = tau_old
    l = 2  # parameter for FrequentDirections

    sketcher = FrequentDirections(n, (l+1)*k)
    for i in range(d):
        a = A[:,i].T
        sketcher.append(a)
        B = sketcher.get().T
        OmgA[:,i] = Omg @ A[:,i]
        # print(np.shape(B))
        if count < t:
            D[:, count] = A[:,i]
            frobA = frobA + sla.norm(a)**2
            count = count + 1
        else:
            
            tau = np.minimum(tau_old, ApproximateRidgeScores(B, C, frobA, k)) 
            tau_D = ApproximateRidgeScores(B, D, frobA, k)
            # print(tau)
            # print(tau_D)
            for j in range(t):
                if not np.all(C[:,j] == 0):
                    prob_rej = 1. - tau[j]/tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:,j] = 0
                        tau_old[j] = 1.
                    else:
                        tau_old[j]=tau[j]
                if np.all(C[:,j] == 0):
                    for l in range(t):
                        prob = tau_D[l]*c*(k*np.log(k)+k*np.log(1./delta)/epsilon)/t
                        roll = rng.random()
                        if roll < prob:
                            C[:,j] = D[:, l]
                            tau_old[j] = tau_D[l]

            Q[:,(nBlock)*t:(nBlock+1)*t] = sla.lstsq(Omg @ C, OmgA[:, (nBlock)*t:(nBlock+1)*t])[0]
            if nBlock == 0:
                C_old = C
            else:
                P = sla.lstsq(C, C_old)[0]
                C_old = C
                Q[:,:(nBlock)*t] = P @ Q[:,:(nBlock)*t]
            count = 0
            nBlock = nBlock + 1

    if nBlock * t < d:
        Q[:, (nBlock)*t:] = sla.lstsq(Omg @ C, OmgA[:, (nBlock)*t:])[0]
    # Q = sla.lstsq(Omg @ C, OmgA)[0]
    return C, Q

def rID_streaming_ridge_leverage_fullsketch(A, k, t, epsilon, delta, c, os, rng = default_rng()):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d, d>> n
    k is the rank of the projection with theoretical guarantees.
    t is the stored column size
    epsilon: accuracy parameter
    delta: (1-delta) is the success probability
    c: oversampling parameter
    os: oversampling number for random projection
    choose epsilon and delta to be less than one.
    """
    n, d = np.shape(A)
    ll = k + os
    Omg = rng.standard_normal(size=(ll, n))


    count = 0
    nBlock = 0
    C = np.zeros((n, t)) # Stores actual column subset
    D = np.zeros((n, t)) # Stores a queue of new columns
    frobA = 0

    Q = np.zeros((t, d))
    C_old = C
    OmgA = np.zeros((ll, d))

    tau_old = np.ones((t)) # Initialize sampling probabilities
    tau = tau_old
    l = 2  # parameter for FrequentDirections

    sketcher = FrequentDirections(n, (l+1)*k)
    for i in range(d):
        a = A[:,i].T
        sketcher.append(a)
        B = sketcher.get().T
        OmgA[:,i] = Omg @ A[:,i]
        # print(np.shape(B))
        if count < t:
            D[:, count] = A[:,i]
            frobA = frobA + sla.norm(a)**2
            count = count + 1
        else:
            
            tau = np.minimum(tau_old, ApproximateRidgeScores(B, C, frobA, k)) 
            tau_D = ApproximateRidgeScores(B, D, frobA, k)
            # print(tau)
            # print(tau_D)
            for j in range(t):
                if not np.all(C[:,j] == 0):
                    prob_rej = 1. - tau[j]/tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:,j] = 0
                        tau_old[j] = 1.
                    else:
                        tau_old[j]=tau[j]
                if np.all(C[:,j] == 0):
                    for l in range(t):
                        prob = tau_D[l]*c*(k*np.log(k)+k*np.log(1./delta)/epsilon)/t
                        roll = rng.random()
                        if roll < prob:
                            C[:,j] = D[:, l]
                            tau_old[j] = tau_D[l]

            # Q[:,(nBlock)*t:(nBlock+1)*t] = sla.lstsq(Omg @ C, OmgA[:, (nBlock)*t:(nBlock+1)*t])[0]
            # if nBlock == 0:
            #     C_old = C
            # else:
            #     P = sla.lstsq(C, C_old)[0]
            #     C_old = C
            #     Q[:,:(nBlock)*t] = P @ Q[:,:(nBlock)*t]
            count = 0
            # nBlock = nBlock + 1

    # if nBlock * t < d:
    #     Q[:, (nBlock)*t:] = sla.lstsq(Omg @ C, OmgA[:, (nBlock)*t:])[0]
    Q = sla.lstsq(Omg @ C, OmgA)[0]
    return C, Q

def truncateSVD_efficient(A, k):
    Q, R = sla.qr( A.T , mode='economic', pivoting=False, check_finite=False ) # A.T = QR so R.T Q.T = A
    U, s, Vh = sla.svd(R.T) # a little m x m SVD
    Vh = Vh @ Q.T

    U_k = U[:, range(k)]
    Vh_k = Vh[range(k), :]
    s_k = s[range(k)]

    return U_k, s_k, Vh_k

def ApproximateRidgeScores(B, M, frobA, k):
    n, t = np.shape(M)
    tau = np.zeros((t))

    # Uk_B, Sk_B ,Vk_B = truncateSVD_efficient(B, k)
    # Bk = Uk_B @ (Sk_B.reshape((-1, 1)) * Vk_B)

    # kernel_inv = sla.pinv(B@B.T+ (frobA - sla.norm(Bk, 'fro')**2)/k * np.eye(n) )

    U, s, _ = sla.svd(B, full_matrices=False)
    UtM = U.T@M
    eig = s**2

    BkF2 = np.sum(eig[1:k]) 
    # ridge_kernel = U.dot( np.diag(1/(eig + (frobA - BkF2)/k))).dot(U.T) #! ridge leverage 
    kernel = np.diag(1/(eig + (frobA - BkF2)/k))
    

    for i in range(t):
        m = UtM[:,i]
        tau[i] = 4. * m.T.dot(kernel).dot(m)

    return tau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int,
                        help='Select method', default = 1)
    parser.add_argument('-d', type=int,
                        help='Select dataset', default = 1)
    parser.add_argument('-q', type=int,
                        help='Select method for comparison')
    parser.add_argument('--random',  action=argparse.BooleanOptionalAction, default = True)
    args = parser.parse_args()
    return args
    
def load_JHTDB_data(
    data_directory="../test_JHTDB/", ntstep=1000, nsample=64, which_component="x",data_name = "channel"
):
    """Loads channel flow data, either x or y or z component (of velocity?)
    The output matrix is reshaped to be nsample^2 x ntstep
        where it seems nsample is the number of x and z grid samples (guessing that y is fixed at 256)
        and ntstep is the number of time steps.   e.g., output is 4096 x 1000, meaning 1000 time steps
    """
    import os

    # ! Load testing channel flow

    # fname = os.path.join(
    #     data_directory, f"channel_x1_{nsample}_y256_z1_{nsample}_t{ntstep}.h5"
    # )
    fname = os.path.join(
        data_directory, f"{data_name}_x1_{nsample}_y256_z1_{nsample}_t{ntstep}.h5"
    )

    # f = h5py.File("../test_JHTDB/channel_x1_64_y256_z1_64_t1000.h5")
    f = h5py.File(fname)
    list(f.keys())
    U = np.zeros((ntstep, nsample, 1, nsample, 3))
    X = np.zeros((nsample, 3))
    list_key = list(f.keys())
    num_keys = len(list_key)
    for i in range(num_keys):
        if i < ntstep:
            U[i, :] = f[list_key[i]]
            # print(U[i,:].shape)
        else:
            X[:, i - ntstep] = f[list_key[i]]

    if which_component.lower() == "x":
        U = U[:, :, :, :, 0].squeeze()
    elif which_component.lower() == "y":
        U = U[:, :, :, :, 1].squeeze()
    elif which_component.lower() == "z":
        U = U[:, :, :, :, 2].squeeze()
    else:
        raise ValueError

    U_reshape = U.reshape([ntstep, -1]).T
    U_reshape = np.array(U_reshape, dtype=np.float64, order="F")
    return U_reshape

def main():
    args = parse_args()
    # method = args.m
    idx_data = args.d

    """Runs a simple test to see if things are working; not exhaustive test!"""
    if idx_data == 1:
        A = load_JHTDB_data(which_component="x",nsample=64,data_name="channel")
    elif idx_data == 2:
        A = load_JHTDB_data(which_component="x",nsample=128,data_name="channel")
    elif idx_data == 3:
        A = load_JHTDB_data(which_component="x",nsample=256,data_name="channel")
    elif idx_data == 4:
        A_ignition = np.load('../ignition_grid/center_cut/ignition_center_cut.npy')
        ntstep = np.shape(A_ignition)[0]
        A = A_ignition[:,:,:,1].squeeze()
        A = A.reshape([ntstep,-1]).T
    elif idx_data == 5:
        A = np.load('/home/angranl/Documents/dataset/nstx_data_ornl_demo.npy')
        A = A.reshape((1280,-1))
        A = A[:,:300000].astype('float64')
    
    # A = A[:,:100]

    dimReduced = 20  # the "rank" of our approximation
    flg_random = args.random
    # print(flg_random)
    rng = default_rng(1) # !For debugging
    xi = 0.005

    flg_debug = False
    

    m, n = np.shape(A)
    if flg_debug:
        flg_random = False
        l = 400
        Omg = rng.standard_normal(size=(l, m))
        
        A = Omg @ A
        print(f"Matrix is {l} x {n}")
    else:
        print(f"Matrix is {m} x {n}")




    # rng = default_rng(1)  # make it reproducible (useful for checking for bugs)
    # # rng = default_rng()   # not reproducible

    nReps = 5
    nAlgo = 1
    err = 0.
    errors= np.zeros( (nReps,nAlgo) )

        # np.random.seed(41)
    for rep in range(nReps):
        t_start = time.time()
        # C, P = rID_streaming_ridge_leverage_old(A, 20, dimReduced, 0.05,0.1, 1.0, 100, rng=default_rng())
        C, P = rID_streaming_ridge_leverage_fullsketch(A, 20, dimReduced, 0.05,0.1, 1.0, 100, rng=default_rng())

        t_end = time.time() - t_start
        A_recon = C @ P
        A_err = A-A_recon
        err = sla.norm(A_err,'fro')/sla.norm(A,'fro')
        # print(f"err: {err}")
        # print(Idx_final)

        print(
            "Online randomized ID (Solve least-square), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                err, t_end
            )
        )
        errors[rep,:] = [ err ]
    err_avg = np.mean(errors,axis=0)
    print(f'Rank: {dimReduced}, Error: {err_avg}')



if __name__ == "__main__":
    main()
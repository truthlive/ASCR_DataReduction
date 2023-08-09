import numpy as np
import scipy
from numpy.random import default_rng
import scipy.linalg as sla
import scipy.linalg.interpolative as sli

from frequentDirections import FrequentDirections
import time
import h5py
# from errorEstimator import *
# from postProcess import *
from rSVDsp import *

import argparse
# from plot_gif import make_gif_matrix
import matplotlib.pyplot as plt

import multiprocessing

def streaming_ridge_leverage(A, k, t, epsilon, delta, c, rng=default_rng()):
    """
    from http://arxiv.org/abs/1511.07263
    A should be n by d
    k is the rank of the projection with theoretical guarantees.
    t is the stored column size
    epsilon: accuracy parameter
    delta: (1 - delta) is the success probability
    c: oversampling parameter
    choose epsilon and delta to be less than one.
    """
    n, d = np.shape(A)
    count = 0
    C = np.zeros((n, t))  # Stores actual column subset
    D = np.zeros((n, t))  # Stores a queue of new columns
    frobA = 0

    tau_old = np.ones((t))  # Initialize sampling probabilities
    tau = tau_old
    l = 2  # parameter for FrequentDirections

    sketcher = FrequentDirections(n, (l + 1) * k)
    for i in range(d):
        a = A[:, i].T
        sketcher.append(a)
        B = sketcher.get().T
        # print(np.shape(B))
        if count < t:
            D[:, count] = A[:, i]
            frobA = frobA + sla.norm(a) ** 2
            count = count + 1
        else:
            tau = np.minimum(tau_old, ApproximateRidgeScores(B, C, frobA, k))
            tau_D = ApproximateRidgeScores(B, D, frobA, k)
            # print(tau)
            # print(tau_D)
            for j in range(t):
                if not np.all(C[:, j] == 0):
                    prob_rej = 1.0 - tau[j] / tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:, j] = 0
                        tau_old[j] = 1.0
                    else:
                        tau_old[j] = tau[j]
                if np.all(C[:, j] == 0):
                    for l in range(t):
                        prob = (
                            tau_D[l]
                            * c
                            * (k * np.log(k) + k * np.log(1.0 / delta) / epsilon)
                            / t
                        )
                        roll = rng.random()
                        if roll < prob:
                            C[:, j] = D[:, l]
                            tau_old[j] = tau_D[l]
            count = 0

    return C


def rID_streaming_ridge_leverage_old(A, k, t, epsilon, delta, c, os, rng=default_rng()):
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
    C = np.zeros((n, t))  # Stores actual column subset
    D = np.zeros((n, t))  # Stores a queue of new columns
    frobA = 0

    Q = np.zeros((t, d))
    C_old = C
    OmgA = np.zeros((ll, d))

    tau_old = np.ones((t))  # Initialize sampling probabilities
    tau = tau_old
    l = 2  # parameter for FrequentDirections

    sketcher = FrequentDirections(n, (l + 1) * k)
    for i in range(d):
        a = A[:, i].T
        sketcher.append(a)
        B = sketcher.get().T
        OmgA[:, i] = Omg @ A[:, i]
        # print(np.shape(B))
        if count < t:
            D[:, count] = A[:, i]
            frobA = frobA + sla.norm(a) ** 2
            count = count + 1
        else:
            # print(B)
            tau = np.minimum(tau_old, ApproximateRidgeScores(B, C, frobA, k))
            tau_D = ApproximateRidgeScores(B, D, frobA, k)
            # print(tau)
            # print(tau_D)
            for j in range(t):
                if not np.all(C[:, j] == 0):
                    prob_rej = 1.0 - tau[j] / tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:, j] = 0
                        tau_old[j] = 1.0
                    else:
                        tau_old[j] = tau[j]
                if np.all(C[:, j] == 0):
                    for l in range(t):
                        prob = (
                            tau_D[l]
                            * c
                            * (k * np.log(k) + k * np.log(1.0 / delta) / epsilon)
                            / t
                        )
                        roll = rng.random()
                        if roll < prob:
                            C[:, j] = D[:, l]
                            tau_old[j] = tau_D[l]
            count = 0

    idx = np.argwhere(np.all(C[..., :] == 0, axis=0))
    C = np.delete(C, idx, axis=1)

    print(np.shape(C))

    # Q = sla.lstsq(Omg @ C, OmgA)[0]
    Q = sla.lstsq(C, A)[0]
    return C, Q


def rID_streaming_ridge_leverage3(
    A, k, t, epsilon, delta, c, os, list_t=None, rng=default_rng()
):
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

    count = 0
    nBlock = 0
    C = np.zeros((n, t))  # Stores actual column subset
    D = np.zeros((n, t))  # Stores a queue of new columns
    IC = -1 * np.ones((t)).astype(int)
    ID = np.zeros((t)).astype(int)
    frobA = 0

    Q = np.zeros((t, d))
    C_old = C
    OmgA = np.zeros((ll, d))
    OmgAOmgAt = np.zeros((ll, ll))

    tau_old = np.ones((t))  # Initialize sampling probabilities
    tau = np.zeros((t))
    probabilities = np.zeros((t))

    for i in range(d):
        a = A[:, i].T
        OmgA[:, i] = Omg @ A[:, i]
        OmgAOmgAt = OmgAOmgAt + OmgA[:, i] @ OmgA[:, i].T
        if count < t:
            D[:, count] = A[:, i]
            ID[count] = i
            frobA = frobA + sla.norm(a) ** 2
            count = count + 1
        else:
            # tau = np.minimum(tau_old, ApproximateRidgeScores2(OmgA[:,:i], Omg, C, frobA, k))
            # tau_D = ApproximateRidgeScores2(OmgA[:,:i], Omg, D, frobA, k)
            tau = np.minimum(
                tau_old, ApproximateRidgeScores3(OmgAOmgAt, Omg, C, frobA, k)
            )
            tau_D = ApproximateRidgeScores3(OmgAOmgAt, Omg, D, frobA, k)
            for j in range(t):
                if IC[j] != -1:
                    prob_rej = 1.0 - tau[j] / tau_old[j]
                    roll = rng.random()
                    if roll < prob_rej:
                        C[:, j] = 0
                        tau_old[j] = 1.0
                        IC[j] = -1
                    else:
                        tau_old[j] = tau[j]

            num_sample = np.sum(IC < 0)
            # print(f'Sum tauD: {np.sum(tau)}')
            # print(f'Sum tauD: {np.sum(tau_D)}')
            # print(f'Sum: {np.sum(probabilities)}')
            idx_sample = np.random.choice(
                ID, num_sample, p=tau_D / np.sum(tau_D), replace=False
            )
            # print(num_sample)
            # print(IC)
            # print(ID)
            # print(idx_sample)
            count_sample = 0
            for j in range(t):
                if IC[j] < 0:
                    IC[j] = idx_sample[count_sample]
                    idx_D = np.where(ID == IC[j])[0][0]
                    # print(idx_D)
                    tau_old[j] = tau_D[idx_D]
                    C[:, j] = D[:, idx_D]
                    count_sample = count_sample + 1

                # if IC[j] == -1:
                #     for l in range(t):
                #         prob = tau_D[l]*c*(k*np.log(k)+k*np.log(1./delta)/epsilon)/t
                #         # prob = tau_D[l]/32.
                #         roll = rng.random()
                #         if roll < prob:
                #             C[:,j] = D[:, l]
                #             tau_old[j] = tau_D[l]
                #             IC[j] = ID[l]

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
    C = np.unique(C, axis=1)
    # print(C)

    Q = sla.lstsq(Omg @ C, OmgA)[0]


    IC = np.unique(IC)

    return C, Q, IC


def truncateSVD_efficient(A, k):
    Q, R = sla.qr(
        A.T, mode="economic", pivoting=False, check_finite=False
    )  # A.T = QR so R.T Q.T = A
    U, s, Vh = sla.svd(R.T)  # a little m x m SVD
    Vh = Vh @ Q.T

    U_k = U[:, range(k)]
    Vh_k = Vh[range(k), :]
    s_k = s[range(k)]

    return U_k, s_k, Vh_k


def ApproximateRidgeScores_old(B, M, frobA, k):
    n, t = np.shape(M)
    tau = np.zeros((t))

    # Uk_B, Sk_B ,Vk_B = truncateSVD_efficient(B, k)
    # Bk = Uk_B @ (Sk_B.reshape((-1, 1)) * Vk_B)

    # kernel_inv = sla.pinv(B@B.T+ (frobA - sla.norm(Bk, 'fro')**2)/k * np.eye(n) )

    U, s, _ = sla.svd(B, full_matrices=False)
    eig = s**2

    BkF2 = np.sum(eig)
    ridge_kernel = U.dot(np.diag(1 / (eig + (frobA - BkF2) / k))).dot(
        U.T
    )  #! ridge leverage

    for i in range(t):
        m = M[:, i]
        tau[i] = 4.0 * m.T.dot(ridge_kernel).dot(m)

    return tau


def ApproximateRidgeScores(B, M, frobA, k):
    """
    Use frequent direction sketch to compute ridge leverage scores
    """

    n, t = np.shape(M)
    tau = np.zeros((t))

    # Uk_B, Sk_B ,Vk_B = truncateSVD_efficient(B, k)
    # Bk = Uk_B @ (Sk_B.reshape((-1, 1)) * Vk_B)

    # kernel_inv = sla.pinv(B@B.T+ (frobA - sla.norm(Bk, 'fro')**2)/k * np.eye(n) )

    U, s, _ = sla.svd(B, full_matrices=False)
    UtM = U.T @ M
    eig = s**2

    BkF2 = np.sum(eig[1:k])
    # ridge_kernel = U.dot( np.diag(1/(eig + (frobA - BkF2)/k))).dot(U.T) #! ridge leverage
    kernel = np.diag(1 / (eig + (frobA - BkF2) / k))

    for i in range(t):
        m = UtM[:, i]
        tau[i] = 4.0 * m.T.dot(kernel).dot(m)

    return tau


def ApproximateRidgeScores2(OmgA, Omg, M, frobA, k):
    """
    Use random projected sketch to compute ridge leverage scores
    """
    n, t = np.shape(M)
    ll, d = np.shape(OmgA)
    tau = np.zeros((t))

    if ll >= d:
        U, s, _ = sla.svd(OmgA, full_matrices=False)
        UtOmgM = U.T @ Omg @ M
        eig = s**2
    else:
        # _, R = sla.qr( OmgA.T, mode='economic', pivoting=False, check_finite=False ) # A.T = QR so R.T Q.T = A
        U, eig, _ = sla.svd(OmgA @ OmgA.T, full_matrices=False)  # a little m x m SVD
        UtOmgM = U.T @ Omg @ M

    BkF2 = np.sum(eig[1:k])
    kernel = np.diag(1 / (eig + (frobA - BkF2) / k))

    for i in range(t):
        m = UtOmgM[:, i]
        tau[i] = 4.0 * m.T.dot(kernel).dot(m)

    return tau


def ApproximateRidgeScores3(OmgAOmgAT, Omg, M, frobA, k):
    """
    Use random projected sketch to compute ridge leverage scores, better implementation to handle larger time steps
    """
    _, t = np.shape(M)
    tau = np.zeros((t))

    U, eig, _ = sla.svd(OmgAOmgAT, full_matrices=False)  # a little m x m SVD
    UtOmgM = U.T @ Omg @ M

    BkF2 = np.sum(eig[1:k])
    kernel = np.diag(1 / (eig + (frobA - BkF2) / k))

    for i in range(t):
        m = UtOmgM[:, i]
        tau[i] = 4.0 * m.T.dot(kernel).dot(m)

    return tau


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, help="Select method", default=1)
    parser.add_argument("-d", type=int, help="Select dataset", default=1)
    parser.add_argument("-q", type=int, help="Select method for comparison", default=1)
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return args


def load_JHTDB_data(
    data_directory="../test_JHTDB/",
    ntstep=1000,
    nsample=64,
    which_component="x",
    data_name="channel",
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
    method = args.m
    idx_data = args.d

    
    A = np.load(
        '/home/angranl/Documents/dataset/nstx_data_ornl_demo_50000.npy'
    ).astype("float64")
    list_rank = [100, 200, 400, 800]
    list_t = None

    flg_random = args.random
    print(flg_random)
    rng = default_rng(1)  # !For debugging
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

    # list_rank = [10]
    nReps = 1
    nAlgo = 1
    err = 0.0
    errors = np.zeros((nReps, nAlgo))

    for dimReduced in list_rank:
        # np.random.seed(41)
        for rep in range(nReps):
            t_start = time.time()

            C, P, IC = rID_streaming_ridge_leverage3(
                A,
                dimReduced,
                dimReduced,
                0.05,
                0.1,
                1.0,
                400,
                list_t=list_t,
                rng=default_rng(),
            )
            # print(IC)

            _, nc = np.shape(C)
            print(f"Number of selected columns is {nc}")

            t_end = time.time() - t_start
            A_recon = C @ P
            A_err = A - A_recon
            err = sla.norm(A_err, "fro") / sla.norm(A, "fro")
            print(
                f"Online randomized ID (k={dimReduced}, overall), relative error:\t{err:.4e}, Time: {t_end:.4f} sec"
            )

            # # #! Randomized ID scipy
            # t_start = time.time()
            # idx, proj = sli.interp_decomp(A.T, dimReduced, rand=False)
            # B = sli.reconstruct_skel_matrix(A.T, dimReduced, idx)
            # Q = sli.reconstruct_interp_matrix(idx, proj)
            # t_end2 = time.time() - t_start

            # A_rID = (B @ Q).T
            # err_scipy = sla.norm(A - A_rID, "fro") / sla.norm(A, "fro")

            #! Randomized SVD Yu
            t_start = time.time()
            U, S, V = rSVDsp_streaming(A.T, b=dimReduced, k=dimReduced, rng=rng)
            V = S.reshape((-1, 1)) * V
            t_end3 = time.time() - t_start
            nrmA = sla.norm(A,'fro')
            norm_f = fastFrobeniusNorm(U, V, A.T, nrmA) / nrmA
            print(
                "rSVDsp_unblock_streaming, relative error:\t\t\t{0:.2e}, Time: {1:.4f} sec".format(
                    norm_f, t_end3
                )
            )


if __name__ == "__main__":
    main()

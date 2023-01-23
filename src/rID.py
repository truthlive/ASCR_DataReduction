#!/usr/bin/env python3
import numpy as np
import matplotlib as plt
import time
import h5py
from numpy.random import default_rng
import scipy.linalg as sli

def rID_res(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID based on  Bhaskara et al. (2019)

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 30
    if m <= k:
        k = m
    l = k + os
    # l=40

    rng = default_rng()
    Omg = rng.standard_normal(size=(l, m))
    S_pre = []
    S_cur = []

    C_pre = []
    C_cur = []

    Idx_pre = []
    Idx_cur = []

    count = 0 # Num of columns in the subset
    sigma = 0
    eye_mat = np.eye(k)

    for col in range(n):
        a = A[:,col]
        # b = Omg @ a
        if flg_random:
            b = Omg @ a
        else:
            b = a
        
            
        if len(S_pre) == 0:
            S_pre.append(b)
            Idx_pre.append(col)
            tmp_coeff = eye_mat[count,:]
            C_pre.append(tmp_coeff)
            count = count + 1
            continue
        else:
            S_pre_mat = np.asarray(S_pre).T
            b_perp_pre = b - (S_pre_mat @ np.linalg.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = np.linalg.norm(b_perp_pre)
            pb = k * norm_b_perp * norm_b_perp/160./xi
        
        prob_sample = np.minimum(pb, 1)
        # print(prob_sample)
        
        # if pb < 1:
        #     prob_sample = pb
        if np.random.random_sample()<= prob_sample and count < k:
            S_cur.append(b)
            Idx_cur.append(col)
            tmp_coeff = eye_mat[count,:]
            C_cur.append(tmp_coeff)
            count = count + 1
        else:
            S_pre_mat = np.asarray(S_pre).T
            tmp_coeff, _ ,_,_= sli.lstsq(S_pre_mat,b)
            tmp_coeff = np.pad(tmp_coeff,(0,k-tmp_coeff.shape[0]))
            C_cur.append(tmp_coeff)
            
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                S_pre = S_pre + S_cur
                S_cur = []
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
                C_pre = C_pre + C_cur
                C_cur = []
        else:
            sigma = 0
            S_pre = S_pre + S_cur
            S_cur = []
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []
            C_pre = C_pre + C_cur
            C_cur = []

    S_final = S_pre + S_cur
    Idx_final = Idx_pre + Idx_cur
    C_final = C_pre + C_cur

    S_final_mat = np.asarray(S_final).T
    C_final_mat = np.asarray(C_final).T

    return S_final_mat, C_final_mat, Idx_final

def onepass_update(A, Omega, blockSize=1):
    """Computes G = A@Omega and H = A.T@G in one-pass,
    loading in at most 'blockSize' rows of A at a time"""
    m, n = np.shape(A)
    k = np.shape(Omega)[1]

    G = np.zeros((0, k))
    H = np.zeros((n, k))

    if blockSize == 1:
        for row in range(m):
            a = A[row, :].reshape((1, n))
            g = a @ Omega
            G = np.vstack((G, g))
            H = H + a.T @ g
    else:
        # New code
        nBlocks = int(m / blockSize)
        for j in range(nBlocks):
            a = A[j * blockSize : (j + 1) * blockSize, :]  # .reshape((1,n))
            g = a @ Omega
            G = np.vstack((G, g))
            H = H + a.T @ g
        if nBlocks * blockSize < m:
            # add the stragglers
            a = A[nBlocks * blockSize :, :]  # .reshape((1,n))
            g = a @ Omega
            G = np.vstack((G, g))
            H = H + a.T @ g

    return G, H

def load_JHTDB_data(
    data_directory="../test_JHTDB/", ntstep=1000, nsample=64, which_component="x"
):
    """Loads channel flow data, either x or y or z component (of velocity?)
    The output matrix is reshaped to be nsample^2 x ntstep
        where it seems nsample is the number of x and z grid samples (guessing that y is fixed at 256)
        and ntstep is the number of time steps.   e.g., output is 4096 x 1000, meaning 1000 time steps
    """
    import os

    # ! Load testing channel flow

    fname = os.path.join(
        data_directory, f"channel_x1_{nsample}_y256_z1_{nsample}_t{ntstep}.h5"
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


def fastFrobeniusNorm(U, Vt, A, nrmA=None):
    """computes || A - U*V ||_F using the expansion
    || A - U*V^T ||_F^2 =
        ||A||^2 + ||U*V^T||^2 - 2 tr(A^T*U*V^T)
    and then tricks like
        || UV^T ||^2 = || tr(VU'UV') = tr(U'U V'V )
        U and V need not have orthonormal columns.
        This code is only efficient if U, V.T have a lot more
        columns than rows (tall matrices).
        Note: following numpy's svd convention, V is NOT transposed
    """
    if nrmA is None:
        nrmA = np.linalg.norm(A, "fro")
    nrm2 = (
        nrmA**2 + np.trace((U.T @ U) @ (Vt @ Vt.T)) - 2 * np.trace((A @ Vt.T).T @ U)
    )
    return np.sqrt(nrm2)


def main():
    """Runs a simple test to see if things are working; not exhaustive test!"""
    A = load_JHTDB_data(which_component="x",nsample=256)
    # A = A[:,:100]

    dimReduced = 10  # the "rank" of our approximation
    xi = 0.05

    m, n = np.shape(A)
    print(f"Matrix is {m} x {n}, using rank {dimReduced}")

    # rng = default_rng(1)  # make it reproducible (useful for checking for bugs)
    # # rng = default_rng()   # not reproducible

    t_start = time.time()
    _, C_final, Idx_final = rID_res(A,dimReduced,xi)   
    t_end = time.time() - t_start

    A_recon = A[:,Idx_final] @ C_final
    A_err = A-A_recon
    err = sli.norm(A-A_recon,'fro')/sli.norm(A,'fro')

    print(
        "Online randomized ID (update basis col by col), relative error:\t{0:.2e}, Time: {1:.4f} sec".format(
            err, t_end
        )
    )

if __name__ == "__main__":
    main()
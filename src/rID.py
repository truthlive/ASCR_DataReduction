#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
from numpy.random import default_rng
import scipy.linalg as sli
import scipy.io as sio

import argparse


def rID_res(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS 

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 30
    if m <= k:
        k = m
    l = k + os
    # l=40

    if flg_random:
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
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
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

def rID_res_new(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS, solve least square problem to get coefficient of the new columns

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 10
    if m <= k:
        k = m
    l = k + os
    # l=40

    if flg_random:
        rng = default_rng()
        Omg = rng.standard_normal(size=(l, m))
        S = np.zeros((l,k))
        C = np.zeros((k,n))
    else:
        S = np.zeros((m,k))
        C = np.zeros((k,n))

    Idx_pre = []
    Idx_cur = []

    count = 0 # Num of columns in the subset
    count_pre = 0
    count_cur = 0
    sigma = 0
    eye_mat = np.eye(k)

    for col in range(n):
        a = A[:,col]
        if flg_random:
            b = Omg @ a
        else:
            b = a

        if count_pre == 0:
            S[:,count_pre] = b
            Idx_pre.append(col)
            C[count_pre, count_pre] = 1.0
            count_pre = count_pre + 1
            continue
        else:
            S_pre_mat = S[:, :count_pre]
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
            # if col <=20:
            #     print(np.shape(S_pre_mat))
            #     print(norm_b_perp)
            pb = k * norm_b_perp * norm_b_perp/160./xi
            
        prob_sample = np.minimum(pb, 1)

        roll = np.random.random_sample()
        # print(roll)
        if roll <= prob_sample and (count_pre + count_cur) < k:
            S[:,count_pre + count_cur] = b
            Idx_cur.append(col)
            C[count_pre + count_cur, col] = 1.0
            count_cur = count_cur + 1
        else:
            S_pre_mat = S[:, :count_pre+ count_cur]
            tmp_coeff, _ ,_,_= sli.lstsq(S_pre_mat,b)
            tmp_coeff = np.pad(tmp_coeff,(0,k-tmp_coeff.shape[0]))
            C[:, col] = tmp_coeff

            
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                count_pre = count_pre + count_cur
                count_cur = 0
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
                # C_pre = C_pre + C_cur
                # C_cur = []
        else:
            sigma = 0
            count_pre = count_pre + count_cur
            count_cur = 0
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []


    Idx_final = Idx_pre + Idx_cur


    return S, C, Idx_final

def rID_res_Stephen(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS, orthognoalize sketched vector

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 10
    if m <= k:
        k = m
    l = k + os

    if flg_random:
        Omg = rng.standard_normal(size=(l, m))
        S = np.zeros((l,k))
        S_ortho = np.zeros((l,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((l,n))
    else:
        S = np.zeros((m,k))
        S_ortho = np.zeros((m,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((m,n))

    Y = np.zeros((k,n))
    

    Idx_pre = []
    Idx_cur = []

    count = 0 # Num of columns in the subset

    sigma = 0
    count_pre = 0 # Num of columns in previous subset
    count_cur = 0 # Num of columns in new selected subset

    for col in range(n):
        a = A[:,col]
        if flg_random:
            b = Omg @ a
        else:
            b = a

        Omg_A[:,col] = b # Store sketched matrix

        if count_pre == 0:
            S[:,count_pre] = b
            S_ortho[:,count_pre] = b /sli.norm(b)
            Idx_pre.append(col)
            C[count_pre, count_pre] = 1.0
            count_pre = count_pre + 1
            Y[0,0] = b.T @ b
            continue
        else:
            S_pre_mat = S[:, :count_pre]
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
            pb = k * norm_b_perp * norm_b_perp/160./xi
            
        prob_sample = np.minimum(pb, 1)

        roll = np.random.random_sample()
        if roll <= prob_sample and (count_pre + count_cur) < k:
            S_new = b - S_ortho[:,:count_pre + count_cur] @ (S_ortho[:,:count_pre + count_cur].T @ b)
            S_new = S_new /sli.norm(S_new)
            S[:,count_pre + count_cur] = b
            S_ortho[:,count_pre + count_cur] = S_new
            Idx_cur.append(col)
            S_cur = S_ortho[:, :count_pre + count_cur+1]

            # print(np.linalg.cond(S_cur.T@S_cur))

            
            # Y[:count_pre + count_cur+1, :col] = 
            # ! Only update needed 
            Y[:count_pre + count_cur, col] = S_ortho[:, :count_pre + count_cur].T @ S_new
            Y[count_pre + count_cur, :col] = S_new.T @ Omg_A[:, :col]

            # C[:count_pre + count_cur+1, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur+1,:col]
            C[:count_pre + count_cur+1, :col] = sli.solve((S_cur.T@S_cur), Y[:count_pre + count_cur+1,:col])
            count_cur = count_cur + 1
        else:
            # S_pre_mat = S[:, :count_pre + count_cur]
            S_cur = S_ortho[:, :count_pre + count_cur]
            Y[:count_pre + count_cur, col] = S_cur.T @ Omg_A[:, col]
            # C[:count_pre + count_cur, col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur, col]


            
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                count_pre = count_pre + count_cur
                count_cur = 0
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
                # S_cur = S[:, :count_pre + count_cur]
                # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
                # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

        else:
            sigma = 0
            count_pre = count_pre + count_cur
            count_cur = 0
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []
            # S_cur = S[:, :count_pre + count_cur]
            # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
            # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

    # C[:count_pre + count_cur, :] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:]

    C= sli.solve((S_ortho.T@S_ortho), Y)
    Idx_final = Idx_pre + Idx_cur


    return S_ortho, C, Idx_final

def rID_res_Stephen_new(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS, orthognoalize original vector

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    A_ortho = np.zeros((m,k))
    os = 10
    if m <= k:
        k = m
    l = k + os

    if flg_random:
        Omg = rng.standard_normal(size=(l, m))
        S = np.zeros((l,k))
        S_ortho = np.zeros((l,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((l,n))
    else:
        Omg = np.eye(m)
        S = np.zeros((m,k))
        S_ortho = np.zeros((m,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((m,n))

    AT_A = np.zeros((k,k))
    Y = np.zeros((k,n))

    

    Idx_pre = []
    Idx_cur = []

    count = 0 # Num of columns in the subset

    sigma = 0
    count_pre = 0 # Num of columns in previous subset
    count_cur = 0 # Num of columns in new selected subset

    for col in range(n):
        a = A[:,col]
        if flg_random:
            b = Omg @ a
        else:
            b = a

        Omg_A[:,col] = b # Store sketched matrix

        if count_pre == 0:
            S[:,count_pre] = b
            # S_ortho[:,count_pre] = b /sli.norm(b)
            A_ortho[:, count_pre] = a/sli.norm(a)
            AT_A[0,0] = A_ortho[:, count_pre].T @ A_ortho[:, count_pre]

            Idx_pre.append(col)
            # C[count_pre, count_pre] = 1.0
            # C[count_pre, count_pre] = sli.norm(a) # if output A_o rtho
            count_pre = count_pre + 1
            Y[0,0] = (a/sli.norm(a)).T @ a
            continue
        else:
            S_pre_mat = S[:, :count_pre+count_cur]
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
            pb = k * norm_b_perp * norm_b_perp/160./xi
            
        prob_sample = np.minimum(pb, 1)

        roll = np.random.random_sample()
        if roll <= prob_sample and (count_pre + count_cur) < k:
            # S_new = b - S_ortho[:,:count_pre + count_cur] @ (S_ortho[:,:count_pre + count_cur].T @ b)
            # S_new = S_new /sli.norm(S_new)
            # S_ortho[:,count_pre + count_cur] = S_new

            A_new = a - A_ortho[:,:count_pre + count_cur] @ (A_ortho[:,:count_pre + count_cur].T @ a)
            A_new = A_new/sli.norm(A_new)
            # print(A_new)
            # print(sli.norm(A_new))
            A_ortho[:,count_pre + count_cur] = A_new
            
            S[:,count_pre + count_cur] = b
            
            Idx_cur.append(col)
            # A_cur = A_ortho[:, :count_pre + count_cur+1]
            AT_A[:count_pre + count_cur+1, :count_pre + count_cur+1] = A_ortho[:, :count_pre + count_cur+1].T @ A_ortho[:, :count_pre + count_cur+1]

            # print(np.linalg.cond(A_cur.T@A_cur))
            # print(count_cur)

            
            # Y[:count_pre + count_cur+1, :col] = 
            # ! Only update needed 
            Y[:count_pre + count_cur, col] = A_ortho[:, :count_pre + count_cur].T @ a

            Y_exact = A_new.T @ A[:,:col]
            Y_approx = (Omg @ A_new).T @ Omg_A[:, :col]

            # Y[count_pre + count_cur, :col] = 0.8 * (Omg @ A_new).T @ Omg_A[:, :col]
            Y[count_pre + count_cur, :col] = np.sign(Y_approx) * np.minimum(np.abs(Y_exact), np.abs(Y_approx)) # Under-estimate

            # C[:count_pre + count_cur+1, :col] = sli.inv(A_cur.T@A_cur) @ Y[:count_pre + count_cur+1,:col]
            # C[:count_pre + count_cur+1, :col] = sli.solve((A_cur.T@A_cur), Y[:count_pre + count_cur+1,:col])
            count_cur = count_cur + 1
        else:
            # S_pre_mat = S[:, :count_pre + count_cur]
            # A_cur = A_ortho[:, :count_pre + count_cur]
            Y[:count_pre + count_cur, col] = A_ortho[:, :count_pre + count_cur].T @ a
            # C[:count_pre + count_cur, col] = sli.inv(A_cur.T@A_cur) @ Y[:count_pre + count_cur, col]


            
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                count_pre = count_pre + count_cur
                count_cur = 0
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
                # S_cur = S[:, :count_pre + count_cur]
                # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
                # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

        else:
            sigma = 0
            count_pre = count_pre + count_cur
            count_cur = 0
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []
            # S_cur = S[:, :count_pre + count_cur]
            # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
            # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

    # C[:count_pre + count_cur, :] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:]

    # C= sli.solve((A_ortho.T@A_ortho), Y)
    C= sli.inv(A_ortho.T@A_ortho) @ Y
    Idx_final = Idx_pre + Idx_cur


    return A_ortho, C, Idx_final

def rID_res_Stephen2(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS, substract previous residual when updating

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 10
    if m <= k:
        k = m
    l = k + os

    if flg_random:
        Omg = rng.standard_normal(size=(l, m))
        S = np.zeros((l,k))
        S_ortho = np.zeros((l,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((l,n))
    else:
        S = np.zeros((m,k))
        S_ortho = np.zeros((m,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((m,n))

    Y = np.zeros((k,n))
    

    Idx_pre = []
    Idx_cur = []

    count = 0 # Num of columns in the subset

    sigma = 0
    count_pre = 0 # Num of columns in previous subset
    count_cur = 0 # Num of columns in new selected subset

    for col in range(n):
        a = A[:,col]
        if flg_random:
            b = Omg @ a
        else:
            b = a

        Omg_A[:,col] = b # Store sketched matrix

        if count_pre == 0:
            S[:,count_pre] = b
            S_ortho[:,count_pre] = b /sli.norm(b)
            Idx_pre.append(col)
            C[count_pre, count_pre] = 1.0
            count_pre = count_pre + 1
            Y[0,0] = b.T @ b
            continue
        else:
            S_pre_mat = S[:, :count_pre]
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
            pb = k * norm_b_perp * norm_b_perp/160./xi
            
        prob_sample = np.minimum(pb, 1)

        roll = np.random.random_sample()
        if roll <= prob_sample and (count_pre + count_cur) < k:
        #     S_new = b - S_ortho[:,:count_pre + count_cur] @ (S_ortho[:,:count_pre + count_cur].T @ b)
        #     S_new = S_new /sli.norm(S_new)
            S[:,count_pre + count_cur] = b
            # S_ortho[:,count_pre + count_cur] = S_new
            Idx_cur.append(col)
            S_cur = S[:, :count_pre + count_cur+1]

            # print(np.linalg.cond(S_cur.T@S_cur))

            
            # Y[:count_pre + count_cur+1, :col] = 
            # ! Only update needed 
            C[:count_pre + count_cur, col],_,_,_ = sli.lstsq(S[:, :count_pre + count_cur], Omg_A[:, col])
            
            Res_a = Omg_A[:, :col] -  S[:, :count_pre + count_cur] @ C[:count_pre + count_cur, :col]
            C[count_pre + count_cur, :col],_,_,_ = sli.lstsq(S[:, count_pre + count_cur].reshape(-1,1), Res_a)

            # C[:count_pre + count_cur+1, col] = sli.solve(S_cur, Res_a)
            count_cur = count_cur + 1

        else:
            # S_pre_mat = S[:, :count_pre + count_cur]
            S_cur = S[:, :count_pre + count_cur]
            # Y[:count_pre + count_cur, col] = S_cur.T @ Omg_A[:, col]
            # C[:count_pre + count_cur, col] = sli.solve(S_cur, Omg_A[:,col])
            C[:count_pre + count_cur, col],_,_,_ = sli.lstsq(S[:, :count_pre + count_cur], Omg_A[:, col])


            
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                count_pre = count_pre + count_cur
                count_cur = 0
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
                # S_cur = S[:, :count_pre + count_cur]
                # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
                # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

        else:
            sigma = 0
            count_pre = count_pre + count_cur
            count_cur = 0
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []
            # S_cur = S[:, :count_pre + count_cur]
            # Y[:count_pre + count_cur, :col] = S_cur.T @ Omg_A[:, :col]
            # C[:count_pre + count_cur, :col] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:col]

    # C[:count_pre + count_cur, :] = sli.inv(S_cur.T@S_cur) @ Y[:count_pre + count_cur,:]

    # C= sli.solve((S_ortho.T@S_ortho), Y)
    Idx_final = Idx_pre + Idx_cur


    return S_ortho, C, Idx_final

def rID_res_qr_update(A, k, xi, rng=default_rng(), flg_random = True):
    """
    Description: Randomized ID using residual based CSS, update coefficient with QR update method.

    Notes: 1. Each column in the input matrix A represents the data from one time step
           2. The entire matrix A is processed col by col
    """
    m, n = np.shape(A)
    os = 10
    if m <= k:
        k = m
    l = k + os

    if flg_random:
        rng = default_rng()
        Omg = rng.standard_normal(size=(l, m))
        S = np.zeros((l,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((l,n))
        # ! Store QR of selected column
        Q_scol = np.zeros((l,k))
        R_scol = np.zeros((k,k)) 
    else:
        S = np.zeros((m,k))
        C = np.zeros((k,n))
        Omg_A = np.zeros((m,n))
        # ! Store QR of selected column
        Q_scol = np.zeros((m,k))
        R_scol = np.zeros((k,k)) 
    
    
    Idx_pre = []
    Idx_cur = []

    sigma = 0
    count_pre = 0
    count_cur = 0

    for col in range(n):
        a = A[:,col]
        if flg_random:
            b = Omg @ a
        else:
            b = a

        Omg_A[:,col] = b # Store sketched matrix

        if count_pre == 0:
            S[:,count_pre] = b
            Idx_pre.append(col)
            C[count_pre, count_pre] = 1.0
            Q_scol[:,count_pre] = b/sli.norm(b)
            R_scol[0,0] = sli.norm(b)
            count_pre = count_pre + 1
            continue
        else:
            S_pre_mat = S[:, :count_pre]
            b_perp_pre = b - (S_pre_mat @ sli.inv(S_pre_mat.T@S_pre_mat) @ S_pre_mat.T) @ b
            norm_b_perp = sli.norm(b_perp_pre)
            pb = k * norm_b_perp * norm_b_perp/160./xi
            
        prob_sample = np.minimum(pb, 1)

        roll = np.random.random_sample()
        if roll <= prob_sample and (count_pre + count_cur) < k:
            S[:,count_pre + count_cur] = b
            Idx_cur.append(col)
            Q_tmp, R_tmp = sli.qr(np.hstack((Q_scol[:,:count_pre + count_cur],b.reshape(-1,1))), mode = "economic")
            Q_scol[:,:count_pre + count_cur + 1] = Q_tmp

            RR = R_tmp @ np.block([[R_scol[:count_pre+count_cur,:count_pre+count_cur],np.zeros((count_pre+count_cur, 1))],[np.zeros((1,count_pre + count_cur)),1.0]])
            R_scol[:count_pre + count_cur + 1, :count_pre + count_cur + 1] = RR

            # ! inverse solving
            # C[:count_pre + count_cur + 1, :col] = sli.inv(R_scol[:count_pre + count_cur + 1, :count_pre + count_cur + 1]) @ Q_tmp.T @ (Omg_A[:,:col])
            # ! triangular solving
            C[:count_pre + count_cur + 1, :col] = sli.solve_triangular((R_scol[:count_pre + count_cur + 1, :count_pre + count_cur + 1]), Q_tmp.T @ (Omg_A[:,:col]))

            count_cur = count_cur + 1
            
        else:
            # C[:count_pre + count_cur, col] = sli.inv(R_scol[:count_pre + count_cur, :count_pre + count_cur]) @ Q_scol[:,:count_pre + count_cur].T @ (Omg_A[:,col])

            C[:count_pre+ count_cur, col] = sli.solve_triangular((R_scol[:count_pre+ count_cur, :count_pre+ count_cur]),  Q_scol[:,:count_pre + count_cur].T @ (Omg_A[:,col]))
            
        if pb < 1:
            sigma = sigma + pb
            if sigma >= 1:
                sigma = 0
                count_pre = count_pre + count_cur
                count_cur = 0
                Idx_pre = Idx_pre + Idx_cur
                Idx_cur = []
        else:
            sigma = 0
            count_pre = count_pre + count_cur
            count_cur = 0
            Idx_pre = Idx_pre + Idx_cur
            Idx_cur = []


    C[:count_pre, :] = sli.solve_triangular(R_scol[:count_pre, :count_pre], Q_scol[:,:count_pre].T @ (Omg_A))
    Idx_final = Idx_pre + Idx_cur


    return C, Idx_final

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
        nrmA = sli.norm(A, "fro")
    nrm2 = (
        nrmA**2 + np.trace((U.T @ U) @ (Vt @ Vt.T)) - 2 * np.trace((A @ Vt.T).T @ U)
    )
    return np.sqrt(nrm2)

def LOO_CV_sketching(SA, Sb):
    m, n = np.shape(SA)
    nrm2 = 0

    for i in range(m):
        idx = range(i-1) + range(i, m)
        beta = sli.solve(SA[idx,:],Sb[idx,:])
        error = sli.norm(SA[i,:] @beta -Sb[i,:],'fro')**2
        nrm2 = nrm2 + error

def LOO_CV_sketching_Ttest():
    args = parse_args()
    method = args.m

    method_compare = args.q

    """Runs a simple test to see if things are working; not exhaustive test!"""
    A = load_JHTDB_data(which_component="x",nsample=64)
    m, n = np.shape(A)

    # A = sio.loadmat("../test_mat2/A1.mat")['A1']
    
    
    # A = A[:,:100]

    dimReduced = 40  # the "rank" of our approximation
    rng = default_rng(1) # !For debugging
    xi = 0.05

    flg_random = False

    os = 10
    l = dimReduced + os
    Omg = rng.standard_normal(size=(l, m))
    A = Omg @ A
    ms, n = np.shape(A)
    
    print(f"Sketched Matrix is {ms} x {n}")
    
    
    nrm2 = 0
    nrm2_compare = 0
    list_res = []
    list_res_compare = []

    list_err = []
    list_err_compare = []

    np.random.seed(41)
    for i in range(ms):
        idx = [*range(ms)]
        # print(idx,i)
        idx.remove(i)
        # idx = range(i-1) + range(i, ms)
        SA = A[idx,:]
        if method == 1:
            # ! Solve Least-square to keep adding coefficient
            t_start = time.time()
            _, C_final, Idx_final = rID_res_new(SA,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err.append(tmp_err2)
            nrm2 = nrm2 + tmp_err2**2
           

            print(
                "Online randomized ID (Solve least-square), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end, i
                )
            )
        elif method == 2:
            # ! Use Stephen's idea to update coefficient
            t_start = time.time()
            S_final, C_final, Idx_final = rID_res_Stephen_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            # A_recon = A[:,Idx_final] @ C_final
            A_recon = S_final @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (Stephen's idea), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        elif method == 3:
            # # ! Use Stephen's second idea to update coefficient based on residual
            t_start = time.time()
            S_final, C_final, Idx_final = rID_res_Stephen2(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err.append(tmp_err2)
            nrm2 = nrm2 + tmp_err2**2

            print(
                "Online randomized ID (update based on residual), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end
                )
            )
        elif method == 4:
            # ! Use QR update to update coefficient
            t_start = time.time()
            C_final, Idx_final = rID_res_qr_update(SA,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err.append(tmp_err2)
            nrm2 = nrm2 + tmp_err2**2
           

            print(
                "Online randomized ID (Solve least-square), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end, i
                )
            )

        if method_compare == 1:
            # ! Solve Least-square to keep adding coefficient
            t_start = time.time()
            _, C_final, Idx_final = rID_res_new(SA,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res_compare.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err_compare.append(tmp_err2)
            nrm2_compare = nrm2_compare + tmp_err2**2
           

            print(
                "Online randomized ID (Solve least-square), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end, i
                )
            )
        elif method_compare == 2:
            # ! Use Stephen's idea to update coefficient
            t_start = time.time()
            S_final, C_final, Idx_final = rID_res_Stephen_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            # A_recon = A[:,Idx_final] @ C_final
            A_recon = S_final @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (Stephen's idea), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        elif method_compare == 3:
            # # ! Use Stephen's second idea to update coefficient based on residual
            t_start = time.time()
            S_final, C_final, Idx_final = rID_res_Stephen2(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res_compare.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err_compare.append(tmp_err2)
            nrm2_compare = nrm2_compare + tmp_err2**2

            print(
                "Online randomized ID (update based on residual), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end,i
                )
            )
        elif method_compare == 4:
            # ! Use QR update to update coefficient
            t_start = time.time()
            C_final, Idx_final = rID_res_qr_update(SA,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final

            tmp_res = A[i,Idx_final] @ C_final - A[i,:]
            list_res_compare.append(tmp_res)
            tmp_err2 = sli.norm(tmp_res)
            list_err_compare.append(tmp_err2)
            nrm2_compare = nrm2_compare + tmp_err2**2
           

            print(
                "Online randomized ID (QR update), leave-out index:{2} relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    tmp_err2, t_end, i
                )
            )

    print("Estimated relative error:\t{0:.4e}".format(np.sqrt(nrm2)/sli.norm(A,'fro')))
    print("Estimated relative error compare:\t{0:.4e}".format(np.sqrt(nrm2_compare)/sli.norm(A,'fro')))


    diff = np.zeros((ms,1))
    for i in range(ms):
        diff[i]=(list_err[i]-list_err_compare[i])
    mean = np.mean(diff)
    std = np.std(diff, ddof = 1)
    t_stat = mean * np.sqrt(ms)/std
    print(f'T statitics: {t_stat}')


    if method == 1:
        # ! Solve Least-square to keep adding coefficient
        t_start = time.time()
        _, C_final, Idx_final = rID_res_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
        t_end = time.time() - t_start

        A_recon = A[:,Idx_final] @ C_final
        A_err = A-A_recon
        err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
        # print(Idx_final)

        print(
            "Online randomized ID (Solve least-square), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                err, t_end
            )
        )
    elif method == 2:
        # ! Use Stephen's idea to update coefficient
        t_start = time.time()
        S_final, C_final, Idx_final= rID_res_Stephen_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
        t_end = time.time() - t_start

        # A_recon = A[:,Idx_final] @ C_final
        A_recon = S_final @ C_final
        A_err = A-A_recon
        err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
        # print(Idx_final)

        print(
            "Online randomized ID (Stephen's idea), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                err, t_end
            )
        )
    elif method == 3:
        # # ! Use Stephen's second idea to update coefficient based on residual
        t_start = time.time()
        S_final, C_final, Idx_final = rID_res_Stephen2(A,dimReduced,xi, rng = rng, flg_random = flg_random)
        t_end = time.time() - t_start

        A_recon = A[:,Idx_final] @ C_final
        # A_recon = S_final @ C_final
        A_err = A-A_recon
        err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
        # print(Idx_final)

        print(
            "Online randomized ID (update based on residual), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                err, t_end
            )
        )
    elif method == 4:
        # ! Use QR update to update coefficient
        t_start = time.time()
        C_final, Idx_final= rID_res_qr_update(A,dimReduced,xi, rng = rng, flg_random = flg_random)
        t_end = time.time() - t_start

        A_recon = A[:,Idx_final] @ C_final
        A_err = A-A_recon
        err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
        # print(Idx_final)

        print(
            "Online randomized ID (QR_update), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                err, t_end
            )
        )
    # t_start = time.time()
    # _, C_final, Idx_final = rID_res_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
    # t_end = time.time() - t_start

    # A_recon = A[:,Idx_final] @ C_final
    # A_err = A-A_recon
    # err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
    

    # print(
    #     "Online randomized ID (Solve least-square), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
    #         err, t_end
    #     )
    # )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int,
                        help='Select method')
    parser.add_argument('-q', type=int,
                        help='Select method for comparison')
    parser.add_argument('--random',  action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    method = args.m

    """Runs a simple test to see if things are working; not exhaustive test!"""
    A = load_JHTDB_data(which_component="x",nsample=64,data_name="channel")

    # A = sio.loadmat("../test_mat2/A1.mat")['A1']
    
    
    # A = A[:,:100]

    dimReduced = 5  # the "rank" of our approximation
    flg_random = args.random
    print(flg_random)
    rng = default_rng(1) # !For debugging
    xi = 0.05

    flg_debug = True
    

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

    # t_start = time.time()
    # _, C_final, Idx_final = rID_res(A,dimReduced,xi)
    # t_end = time.time() - t_start
    # list_rank = [5,10,20,40,50,100]
    list_rank = [20]

    for dimReduced in list_rank:
        np.random.seed(41)
        if method == 1:
            # ! Solve Least-square to keep adding coefficient
            t_start = time.time()
            _, C_final, Idx_final = rID_res_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (Solve least-square), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        elif method == 2:
            # ! Use Stephen's idea to update coefficient
            t_start = time.time()
            S_final, C_final, Idx_final = rID_res_Stephen_new(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            # A_recon = A[:,Idx_final] @ C_final
            A_recon = S_final @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (Stephen's idea), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        elif method == 3:
            # # ! Use Stephen's second idea to update coefficient based on residual
            t_start = time.time()
            S_final, C_final, Idx_final= rID_res_Stephen2(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final
            # A_recon = S_final @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (update based on residual), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        elif method == 4:
            # ! Use QR update to update coefficient
            t_start = time.time()
            C_final, Idx_final = rID_res_qr_update(A,dimReduced,xi, rng = rng, flg_random = flg_random)
            t_end = time.time() - t_start

            A_recon = A[:,Idx_final] @ C_final
            A_err = A-A_recon
            err = sli.norm(A_err,'fro')/sli.norm(A,'fro')
            # print(Idx_final)

            print(
                "Online randomized ID (QR_update), relative error:\t{0:.4e}, Time: {1:.4f} sec".format(
                    err, t_end
                )
            )
        
        # tt = np.linspace(0,n,n)
        # fig = plt.figure()
        # for i in range(dimReduced):
        #     plt.plot(tt, C_final[i,:])
        #     print(np.average(np.abs(C_final[i,:])))

        # list_legend = [str(x) for x in range(dimReduced)]
        # plt.legend(list_legend)

        # plt.show()
        # plt.save('../test_JHTDB/Coefficient_k{0}.png'.format(dimReduced))



if __name__ == "__main__":
    main()
    # LOO_CV_sketching_Ttest()

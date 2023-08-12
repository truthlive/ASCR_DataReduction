#!/usr/bin/env python3
import numpy as np
import matplotlib as plt
import time
import h5py
from numpy.random import default_rng
import scipy.linalg

# def rSVDsp(A, k, b=10):

#     m, n = np.shape(A)
#     os = 10
#     if n <= k:
#         k=n
#     k = k + os

#     # np.random.seed(42)
#     Omg = np.random.randn(n,k)
#     G = np.dot(A,Omg)
#     H = np.dot(A.T, G)
#     Q = np.zeros((m, 0))
#     B = np.zeros((0, n))

#     s =np.floor(k/b).astype(int)
#     for i in range(s):
#         temp = np.dot(B, Omg[:, range(i*b,(i+1)*b)])
#         Yi = G[:, range(i*b,(i+1)*b)] - np.dot(Q, temp)

#         Qi, Ri = np.linalg.qr(Yi)
#         Qi, Rit= np.linalg.qr(Qi-np.dot(Q, np.dot(Q.T, Qi)))
#         Ri = np.dot(Rit, Ri)
#         Bi = Ri.T 
#         Bi,_,_,_ = np.linalg.lstsq(Ri.T, H[:, range(i*b, (i+1)*b)].T-np.dot(np.dot(Yi.T,Q),B)-np.dot(temp.T,B))
#         # Bi = np.linalg.solve(Ri.T, H[:, range(i*b, (i+1)*b)].T-np.dot(np.dot(Yi.T,Q),B)-np.dot(temp.T,B))
#         # Bi = np.linalg.inv(Ri.T)@(H[:, range(i*b, (i+1)*b)].T-np.dot(np.dot(Yi.T,Q),B)-np.dot(temp.T,B))


#         Q = np.hstack((Q, Qi))
#         B = np.vstack((B, Bi))

#     U1, S, V = np.linalg.svd(B)
#     U = np.dot(Q,U1)

#     # S= np.diag(S);
#     U= U[:, range(k-os)]
#     V= V[range(k-os),:]
#     S= S[range(k-os)]


#     return U, S, V

def rSVDsp(A, k, b=10, rng = default_rng() ):
    """
    Description: Single-pass Randomized Blocked SVD From Yu et al (2017)
    
    Notes: 1. Each row in the input matrix A represents the data from one time step
           2. The entire matrix A is read without streaming
    """
    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = rng.standard_normal( size=(n,k))
    G = A @ Omg
    H = A.T @ G
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))

    s =np.floor(k/b).astype(int)
    for i in range(s):
        temp = B @ Omg[:, range(i*b,(i+1)*b)]
        Yi = G[:, range(i*b,(i+1)*b)] - Q @ temp

        Qi, Ri = np.linalg.qr(Yi)
        Qi, Rit= np.linalg.qr(Qi-Q@(Q.T@Qi))
        Ri = Rit@ Ri
        Bi = Ri.T 
        
        # Bi,_,_,_ = np.linalg.lstsq(Ri.T, H[:, range(i*b, (i+1)*b)].T-Yi.T@Q@B-temp.T@B)
        # Bi = np.linalg.solve(Ri.T, H[:, range(i*b, (i+1)*b)].T-np.dot(np.dot(Yi.T,Q),B)-np.dot(temp.T,B))
        #Bi = np.linalg.inv(Ri.T)@(H[:, range(i*b, (i+1)*b)].T-Yi.T@Q@B-temp.T@B)
        Bi = scipy.linalg.solve_triangular( Ri,  H[:, range(i*b, (i+1)*b)].T-Yi.T@Q@B-temp.T@B  , trans='T')


        Q = np.hstack((Q, Qi))
        B = np.vstack((B, Bi))
    U1, S, V = np.linalg.svd(B, full_matrices=False)

    U = Q@U1

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]


    return U, S, V


def onepass_update(A,Omega,blockSize=1):
    """ Computes G = A@Omega and H = A.T@G in one-pass,
    loading in at most 'blockSize' rows of A at a time """
    m, n = np.shape(A)
    k = np.shape(Omega)[1]

    G = np.zeros((0,k))
    H = np.zeros((n,k))
    #print(f"Row-major order (C order) is: {A.flags['C_CONTIGUOUS']}")
    
    if blockSize == 1:
        G = np.zeros((m,k))   # preallocate for speed
        aTg = np.zeros((n,k)) # we'll reuse the same memory for speed
        for row in range(m):
            a = A[row, :].reshape((1,n))
            g = a @ Omega
            G[row,:] = g # faster than G = np.vstack((G, g))

            # H = H + a.T @ g # Slow (23.1 sec)

            # H += np.outer(a,g)  # Medium (16.3 sec)

            # np.outer(a,g, out = aTg) 
            # H += aTg  # Medium fast (11.6 sec)
            
            # H = scipy.linalg.blas.dger(1.0,a.flatten(), g.flatten(), a=H, overwrite_a = 0) # Fast (7.3 sec)
            H = scipy.linalg.blas.dger(1.0,a.flatten(), g.flatten(), a=H, overwrite_a = 1) # Fastest (3.2 sec)
    else:
        # New code
        nBlocks = int( m / blockSize )
        for j in range(nBlocks):
            a = A[j*blockSize:(j+1)*blockSize, :] #.reshape((1,n))
            g = a @ Omega
            G = np.vstack((G, g))
            H = H + a.T@g
        if nBlocks*blockSize < m:
            # add the stragglers
            a = A[nBlocks*blockSize:, :] #.reshape((1,n))
            g = a @ Omega
            G = np.vstack((G, g))
            H = H + a.T@g
    
    return G, H

def rSVDsp_streaming(A, k, b=10, rng = default_rng(), blocksize_A = 1 ):
    """
    Description: Single-pass Randomized Blocked SVD From Yu et al (2017) with streaming input
    
    Notes: 1. Each row in the input matrix A represents the data from one time step
           2. The matrix A is read row by row 
           
    b is the blocksize used for the "blocking" in Yu et al. (for parallel computation)
    while blocksize_A is the blocksize used for reading in rows of A for one-pass computation
    """
    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = rng.standard_normal( size=(n,k))

#     G = np.zeros((0,k))
#     H = np.zeros((n,k))

#     for row in range(m):
#         a = A[row, :].reshape((1,n))
#         g = a @ Omg
#         G = np.vstack((G, g))
#         H = H + a.T@g
    G, H = onepass_update(A,Omg,blockSize=blocksize_A) # Added Oct 19 2022

    # G = A @ Omg
    # H = A.T @ G
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))

    s =np.floor(k/b).astype(int)
    for i in range(s):
        temp = B @ Omg[:, range(i*b,(i+1)*b)]
        Yi = G[:, range(i*b,(i+1)*b)] - Q @ temp

        Qi, Ri = np.linalg.qr(Yi)
        Qi, Rit= np.linalg.qr(Qi-Q@(Q.T@Qi))
        Ri = Rit@ Ri
        #Bi = Ri.T 

        
        # Bi,_,_,_ = np.linalg.lstsq(Ri.T, H[:, range(i*b, (i+1)*b)].T-Yi.T@Q@B-temp.T@B)
        # Bi = np.linalg.solve(Ri.T, H[:, range(i*b, (i+1)*b)].T-np.dot(np.dot(Yi.T,Q),B)-np.dot(temp.T,B))
        Bi = np.linalg.inv(Ri.T)@(H[:, range(i*b, (i+1)*b)].T-Yi.T@Q@B-temp.T@B)


        Q = np.hstack((Q, Qi))
        B = np.vstack((B, Bi))
    U1, S, V = np.linalg.svd(B, full_matrices=False)

    U = Q@U1

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]


    return U, S, V

def rSVDsp_unblock_streaming(A, k, rng = default_rng(), blocksize_A = 1  ):
    """
    Description: Modified from the Single-pass Randomized Blocked SVD From Yu et al (2017)
    
    Notes: 1. Each row in the input matrix A represents the data from one time step
           2. The matrix A is read row by row 
           3. The SVD is performed not blocked
    
    blocksize_A is the blocksize used for reading in rows of A for one-pass computation
    """
    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = rng.standard_normal( size=(n,k))

#     G = np.zeros((0,k))
#     H = np.zeros((n,k))

#     for row in range(m):
#         a = A[row, :].reshape((1,n))
#         g = a @ Omg
#         G = np.vstack((G, g))
#         H = H + a.T@g
    G, H = onepass_update(A,Omg,blockSize=blocksize_A) # Added Oct 19 2022

    # G = A @ Omg
    # H = A.T @ G

    Q, R = np.linalg.qr(G)
    # B = H @ np.linalg.inv(R)
    # B = np.linalg.solve(R.T, H.T)
    # B, _, _, _ = np.linalg.lstsq(R.T, H.T, rcond=None)

    roughConditionNumber = R[0,0] / R[-1,-1]
    if roughConditionNumber > 1e8:
        B = H @ np.linalg.pinv(R) # regularizes a bit...  
    else:
        B = scipy.linalg.solve_triangular( R, H.T, trans='T').T

    U1, S, V = np.linalg.svd(B.T, full_matrices=False)

    U = Q@U1

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]


    return U, S, V

def rSVDsp_unblock(A, k, rng = default_rng() ):

    m, n = np.shape(A)
    os = 10
    k = k + os
    # np.random.seed(42)
    
    Omg = rng.standard_normal( size=(n,k))

    Y = A@Omg
    D = A.T@Y
    Q, R = np.linalg.qr(Y)
    B = scipy.linalg.solve_triangular( R, D.T, trans = 'T')
    #B = ( B @ np.linalg.inv(R) ).T

    U1, S, V = np.linalg.svd(B)
    
    U = np.dot(Q,U1)

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]

    return U, S, V

def truncateSVD(A, k):
    """
    Description: Truncated SVD to get the low rank approximation of A with rank k
    
    """
    U, S, V = np.linalg.svd(A)
    U = U[:, range(k)]
    V = V[range(k),:]
    S = S[range(k)]

    return U, S, V

def SVD_update_adaptive(A, k, blockSize_ini, blockSize_add, rng = default_rng() ):

    """
    Description: Adaptive SVD updating implementation (streaming)

    Note: 1. Each COLUMN of the input matrix A represents the data from one time step
          2. The first blockSize_ini column of A is read to get an initial low-rank approximation with rank k 
          3. The SVD updating then read every blockSize_add columns of A and update the SVD 
    
    """

    m, n = np.shape(A)
    
    nBlocks_add = np.ceil((n-blockSize_ini)/blockSize_add).astype(int)

    A_b_ini = A[:,range(0,blockSize_ini)]

    # Ub, Sb, Vb = truncateSVD(A_b_ini, k)
    
    Vbt, Sb, Ubt = rSVDsp_unblock_streaming(A_b_ini.T, k, rng = rng)
    Ub = Ubt.T
    Vb = Vbt.T

    # Ub, Sb, Vb = rSVDsp_streaming(A_b_ini, k)
    # Ub = Ubt.T
    # Vb = Vbt.T

    for b in range(nBlocks_add):
        if (b+1)*blockSize_add + blockSize_ini <= n:
            A_b = A[:, range(blockSize_ini+ b*blockSize_add, blockSize_ini + (b+1)*blockSize_add)]
            tmp_blockSize = blockSize_add
        else:
            A_b = A[:, range(blockSize_ini + b*blockSize_add, n)]
            tmp_blockSize = n-b*blockSize_add - blockSize_ini
        
        # A_b = Ux_reshape[:, tmp_blockSize]
        A_b_ortho = A_b - Ub@(Ub.T@A_b)
        P, R = np.linalg.qr(A_b_ortho.reshape((-1,tmp_blockSize)))
        P_hat = np.hstack((Ub, P)) 

        m_Vb, n_Vb = np.shape(Vb.T)

        # Q11 = Vb.T
        # Q12 = np.zeros((m_Vb,blockSize))
        # Q21 = np.zeros((blockSize,n_Vb))
        # Q22 = np.eye(blockSize)
        # # Qup = np.block([Q11,Q12])
        # # Qbot = np.block([Q21,Q22])
        # Q = np.block([[Q11,Q12],[Q21,Q22]])

        Q = np.block([[Vb.T, np.zeros((m_Vb,tmp_blockSize))],[np.zeros((tmp_blockSize,n_Vb)), np.eye(tmp_blockSize)]])

        # A_b_old = np.hstack((A_b_old, A_b))
        Sb_mat = np.zeros((Sb.size, Sb.size))
        np.fill_diagonal(Sb_mat,Sb)
        B = np.block([[Sb_mat, Ub.T@A_b.reshape((-1,tmp_blockSize))],[np.zeros((tmp_blockSize,k)),R]])

        X_B,S_B,Y_B = np.linalg.svd(B)
        # X_B,S_B,Y_B = rSVDsp(B, k)
        # Vb= Vb_new[range(k),:]

        Ub_new = P_hat @ X_B[:, range(k)]
        Vb_new = Y_B[range(k),:]@Q.T

        Ub = Ub_new
        Vb = Vb_new
        Sb = S_B[range(k)]

    return Ub, Sb, Vb

def SVD_update_adaptive_sketch(A, k, blockSize_ini, blockSize_add, rng = default_rng() ):
    """
    Description: Adaptive SVD updating combined with sketching (NOT finished yet)
    
    """
    m, n = np.shape(A)
    
    nBlocks_add = np.ceil((n-blockSize_ini)/blockSize_add).astype(int)

    A_b_ini = A[:,range(0,blockSize_ini)]

    # Ub, Sb, Vb = truncateSVD(A_b_ini, k)
    
    Vbt, Sb, Ubt = rSVDsp_unblock_streaming(A_b_ini.T, k, rng)
    Ub = Ubt.T
    Vb = Vbt.T

    # Ub, Sb, Vb = rSVDsp_streaming(A_b_ini, k)
    # Ub = Ubt.T
    # Vb = Vbt.T

    for b in range(nBlocks_add):
        if (b+1)*blockSize_add + blockSize_ini <= n:
            A_b = A[:, range(blockSize_ini+ b*blockSize_add, blockSize_ini + (b+1)*blockSize_add)]
            tmp_blockSize = blockSize_add
        else:
            A_b = A[:, range(blockSize_ini + b*blockSize_add, n)]
            tmp_blockSize = n-b*blockSize_add - blockSize_ini
        
        # A_b = Ux_reshape[:, tmp_blockSize]
        A_b_ortho = A_b - Ub@(Ub.T@A_b)
        P, R = np.linalg.qr(A_b_ortho.reshape((-1,tmp_blockSize)))
        P_hat = np.hstack((Ub, P)) 

        m_Vb, n_Vb = np.shape(Vb.T)

        # Q11 = Vb.T
        # Q12 = np.zeros((m_Vb,blockSize))
        # Q21 = np.zeros((blockSize,n_Vb))
        # Q22 = np.eye(blockSize)
        # # Qup = np.block([Q11,Q12])
        # # Qbot = np.block([Q21,Q22])
        # Q = np.block([[Q11,Q12],[Q21,Q22]])

        Q = np.block([[Vb.T, np.zeros((m_Vb,tmp_blockSize))],[np.zeros((tmp_blockSize,n_Vb)), np.eye(tmp_blockSize)]])

        # A_b_old = np.hstack((A_b_old, A_b))
        Sb_mat = np.zeros((Sb.size, Sb.size))
        np.fill_diagonal(Sb_mat,Sb)
        B = np.block([[Sb_mat, Ub.T@A_b.reshape((-1,tmp_blockSize))],[np.zeros((tmp_blockSize,k)),R]])

        X_B,S_B,Y_B = np.linalg.svd(B)
        # X_B,S_B,Y_B = rSVDsp(B, k)
        # Vb= Vb_new[range(k),:]

        Ub_new = P_hat @ X_B[:, range(k)]
        Vb_new = Y_B[range(k),:]@Q.T

        Ub = Ub_new
        Vb = Vb_new
        Sb = S_B[range(k)]

    return Ub, Sb, Vb

def load_JHTDB_data(data_directory = '../test_JHTDB/', ntstep = 1000, nsample = 64, which_component = 'x'):
    ''' Loads channel flow data, either x or y or z component (of velocity?)
    The output matrix is reshaped to be nsample^2 x ntstep
        where it seems nsample is the number of x and z grid samples (guessing that y is fixed at 256)
        and ntstep is the number of time steps.   e.g., output is 4096 x 1000, meaning 1000 time steps
    '''
    import os
    # ! Load testing channel flow

    fname = os.path.join( data_directory, f'channel_x1_{nsample}_y256_z1_{nsample}_t{ntstep}.h5')

    # f = h5py.File("../test_JHTDB/channel_x1_64_y256_z1_64_t1000.h5")
    f=h5py.File(fname)
    list(f.keys())
    U = np.zeros((ntstep, nsample, 1, nsample, 3))
    X = np.zeros((nsample,3))
    list_key = list(f.keys())
    num_keys = len(list_key)
    for i in range(num_keys):
        if i < ntstep:
            U[i, :] = f[list_key[i]]
            # print(U[i,:].shape)
        else:
            X[:, i-ntstep] = f[list_key[i]]

    if which_component.lower() == 'x':
        U = U[:,:,:,:,0].squeeze()
    elif which_component.lower() == 'y':
        U = U[:,:,:,:,1].squeeze()
    elif which_component.lower() == 'z':
        U = U[:,:,:,:,2].squeeze()
    else:
        raise ValueError

    U_reshape = U.reshape([ntstep,-1]).T
    U_reshape = np.array(U_reshape,dtype=np.float64,order="F")
    return U_reshape

def fastFrobeniusNorm(U,Vt,A,nrmA = None ):
    """ computes || A - U*V ||_F using the expansion
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
        nrmA = np.linalg.norm(A,'fro')
    nrm2 = nrmA**2 + np.trace( (U.T@U) @ (Vt@Vt.T) ) - 2*np.trace( (A@Vt.T).T @ U)
    return np.sqrt( nrm2 )

def main():
    ''' Runs a simple test to see if things are working; not exhaustive test! '''
    A = load_JHTDB_data(which_component='x')

    dimReduced = 50 # the "rank" of our approximation

    blockSize = 100
    m, n = np.shape(A)
    print(f'Matrix is {m} x {n}, using rank {dimReduced}')
    nBlocks = np.ceil(n/blockSize).astype(int)

    
    blockSize_ini = 100
    blockSize_add = 1

    rng = default_rng(1)  # make it reproducible (useful for checking for bugs)
    # rng = default_rng()   # not reproducible

    t_start = time.time()
    Ub, Sb, Vb = SVD_update_adaptive(A, dimReduced,blockSize_ini, blockSize_add, rng = rng)
    Vb = Sb.reshape((-1,1))*Vb  # Python usually broadcasts the sizes correctly
    t_end = time.time() - t_start

    A_recon_col_update = Ub@Vb
    nrmA = np.linalg.norm(A,'fro')

    #norm_f = np.linalg.norm(A-A_recon_col_update,'fro')/nrmA
    norm_f = fastFrobeniusNorm(Ub,Vb,A, nrmA)/nrmA  # efficient way
    

    print('Blocked version (update basis col by col), relative error:\t{0:.2e}, Time: {1:.4f} sec'.format(norm_f,t_end))

    t_start = time.time()
    U,S,V = rSVDsp(A, k=dimReduced, rng = rng )
    V = S.reshape((-1,1))*V
    t_end = time.time() - t_start
    norm_f = fastFrobeniusNorm(U,V,A, nrmA)/nrmA
    print('rSVDsp, relative error:\t\t\t\t\t\t{0:.2e}, Time: {1:.4f} sec'.format(norm_f,t_end))

    t_start = time.time()
    U,S,V = rSVDsp_unblock( A, k=dimReduced, rng = rng )
    V = S.reshape((-1,1))*V
    t_end = time.time() - t_start
    norm_f = fastFrobeniusNorm(U,V,A, nrmA)/nrmA
    print('rSVDsp_unblock, relative error:\t\t\t\t\t{0:.2e}, Time: {1:.4f} sec'.format(norm_f,t_end))

    t_start = time.time()
    U,S,V = rSVDsp_unblock_streaming(A, k=dimReduced, rng = rng )
    V = S.reshape((-1,1))*V
    t_end = time.time() - t_start
    norm_f = fastFrobeniusNorm(U,V,A, nrmA)/nrmA
    print('rSVDsp_unblock_streaming, relative error:\t\t\t{0:.2e}, Time: {1:.4f} sec'.format(norm_f,t_end))

    t_start = time.time()
    U,S,V = rSVDsp_streaming(A, k=dimReduced, rng = rng )
    V = S.reshape((-1,1))*V
    t_end = time.time() - t_start
    norm_f = fastFrobeniusNorm(U,V,A, nrmA)/nrmA
    print('rSVDsp_streaming, relative error:\t\t\t\t{0:.2e}, Time: {1:.4f} sec'.format(norm_f,t_end))

if __name__ == "__main__":
    main()


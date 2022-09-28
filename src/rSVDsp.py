import numpy as np
import matplotlib as plt
import time
import h5py

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

def rSVDsp(A, k, b=10):

    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = np.random.randn(n,k)
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

def rSVDsp_streaming(A, k, b=10):

    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = np.random.randn(n,k)

    G = np.zeros((0,k))
    H = np.zeros((n,k))

    for row in range(m):
        a = A[row, :].reshape((1,n))
        g = a @ Omg
        G = np.vstack((G, g))
        H = H + a.T@g

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
        Bi = Ri.T 

        
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

def rSVDsp_unblock_streaming(A, k):

    m, n = np.shape(A)
    os = 10
    if n <= k:
        k=n
    k = k + os

    # np.random.seed(42)
    Omg = np.random.randn(n,k)

    G = np.zeros((0,k))
    H = np.zeros((n,k))

    for row in range(m):
        a = A[row, :].reshape((1,n))
        g = a @ Omg
        G = np.vstack((G, g))
        H = H + a.T@g

    # G = A @ Omg
    # H = A.T @ G

    Q, R = np.linalg.qr(G)
    # B = H @ np.linalg.inv(R)
    # B = np.linalg.solve(R.T, H.T)
    # B, _, _, _ = np.linalg.lstsq(R.T, H.T, rcond=None)

    B = H @ np.linalg.pinv(R)

    U1, S, V = np.linalg.svd(B.T, full_matrices=False)

    U = Q@U1

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]


    return U, S, V

def rSVDsp_unblock(A, k):

    m, n = np.shape(A)
    os = 10
    k = k + os
    # np.random.seed(42)
    
    Omg = np.random.randn(n,k)

    Y = A@Omg
    B = A.T@Y
    Q, R = np.linalg.qr(Y)
    B = B @ np.linalg.inv(R)

    U1, S, V = np.linalg.svd(B.T)
    
    U = np.dot(Q,U1)

    # S= np.diag(S);
    U= U[:, range(k-os)]
    V= V[range(k-os),:]
    S= S[range(k-os)]

    return U, S, V

def truncateSVD(A, k):
    U, S, V = np.linalg.svd(A)
    U = U[:, range(k)]
    V = V[range(k),:]
    S = S[range(k)]

    return U, S, V

def SVD_update_adaptive(A, k, blockSize_ini, blockSize_add):
    m, n = np.shape(A)
    
    nBlocks_add = np.ceil((n-blockSize_ini)/blockSize_add).astype(int)

    A_b_ini = A[:,range(0,blockSize_ini)]

    # Ub, Sb, Vb = truncateSVD(A_b_ini, k)
    
    Vbt, Sb, Ubt = rSVDsp_unblock_streaming(A_b_ini.T, k)
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

def SVD_update_adaptive_sketch(A, k, blockSize_ini, blockSize_add):
    m, n = np.shape(A)
    
    nBlocks_add = np.ceil((n-blockSize_ini)/blockSize_add).astype(int)

    A_b_ini = A[:,range(0,blockSize_ini)]

    # Ub, Sb, Vb = truncateSVD(A_b_ini, k)
    
    Vbt, Sb, Ubt = rSVDsp_unblock_streaming(A_b_ini.T, k)
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


def main():
    # ! Load testing channel flow
    ntstep = 1000
    nsample = 64

    fname = '../test_JHTDB/channel_x1_{ns}_y256_z1_{ns}_t{nt}.h5'.format(ns=nsample, nt =ntstep)

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


    Ux = U[:,:,:,:,0].squeeze()
    Uy = U[:,:,:,:,1].squeeze()
    Uz = U[:,:,:,:,2].squeeze()

    Ux_reshape = Ux.reshape([ntstep,-1]).T
    Uy_reshape = Uy.reshape([ntstep,-1]).T
    Uz_reshape = Uz.reshape([ntstep,-1]).T

    Ux_reshape = np.array(Ux_reshape,dtype=np.float64,order="F")

    dimReduced = 99

    blockSize = 100
    m, n = np.shape(Ux_reshape)
    nBlocks = np.ceil(n/blockSize).astype(int)

    t_start = time.time()
    blockSize_ini = 100
    blockSize_add = 1

    Ub, Sb, Vb = SVD_update_adaptive(Ux_reshape, dimReduced,blockSize_ini, blockSize_add)

    t_end = time.time()
    Sb_mat_update = np.zeros((dimReduced, dimReduced))
    np.fill_diagonal(Sb_mat_update, Sb)
    Ux_recon_col_update = Ub@Sb_mat_update@Vb
    norm_f = np.linalg.norm(Ux_reshape-Ux_recon_col_update,'fro')/np.linalg.norm(Ux_reshape,'fro')

    print('Blocked version (update basis col by col):{0} Time:{1}'.format(norm_f,t_end-t_start))

if __name__ == "__main__":
    main()


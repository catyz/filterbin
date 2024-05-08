import numpy as np
import scipy as sp

def try_gep(A, B, k, cut):
    try:
        print(f'attempting sparse gep with k={k}')
        e, v = sp.sparse.linalg.eigsh(A, k, B)
    except:        
        new_k = k-100
        print(f'exception!')
        try_gep(A, B, new_k, cut)
    else:
        if e[0] > cut:
            new_k = k+10
            print(f'smallest eig too big, {e[0]}')
            try_gep(A, B, new_k, cut)
        else:
            print('all good, saving')
            np.save('eigenvalues', e)
            np.save('eigenvectors', v)
            print('DONE')
        
def main():    
    nside = 64
    print('loading obs cov')
    obs_C_E = sp.sparse.load_npz(f'obs_C_E_{nside}.npz')
    obs_C_B = sp.sparse.load_npz(f'obs_C_B_{nside}.npz')

    print('regularizing')
    diag_E = obs_C_E.diagonal()
    diag_B = obs_C_B.diagonal()
    sigma = (np.mean(diag_E[diag_E!=0]) + np.mean(diag_B[diag_B!=0])) / 2 /1000
    factor = sp.sparse.identity(obs_C_E.shape[0])*sigma**2
    
    cut = 1.02
    try_gep(obs_C_B+factor, obs_C_E+factor, 2010, cut)

if __name__ == "__main__":
    main()

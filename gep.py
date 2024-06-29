import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky
        
def main():    
    nside = 128
    prefix = 'apo8'
    print('loading obs cov')
    obs_C_E = sp.sparse.load_npz(f'{prefix}/obs_C_E_{nside}.npz')
    obs_C_B = sp.sparse.load_npz(f'{prefix}/obs_C_B_{nside}.npz')

    print('regularizing')
    diag_E = obs_C_E.diagonal()
    diag_B = obs_C_B.diagonal()
    sigma = (np.mean(diag_E[diag_E!=0]) + np.mean(diag_B[diag_B!=0])) / 2 /1000
    factor = sp.sparse.identity(obs_C_E.shape[0])*sigma**2
    
    # cut = 1.02
    print('cholesky')
    cho = cholesky(obs_C_E+factor)
    Minv = sp.sparse.linalg.LinearOperator(matvec=lambda b: cho(b), shape=obs_C_E.shape, dtype=obs_C_E.dtype)
    print('attempting gep')
    e, v = sp.sparse.linalg.eigsh( obs_C_B+factor, 5000, obs_C_E+factor, Minv=Minv)
    np.save(f'{prefix}/eigenvalues', e)
    np.save(f'{prefix}/eigenvectors', v)
    print('done')

if __name__ == "__main__":
    main()

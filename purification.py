import numpy as np
import scipy as sp
import healpy as hp

def main():
    prefix=f'apo10'
    print('loading gep results')
    eigs = np.load(f'{prefix}/eigenvalues.npy')
    v = np.load(f'{prefix}/eigenvectors.npy')
    # cut = 1.02
    # n_v = np.where(eigs >= cut)[0]
    n_v = 7000
    pure_b = v[:, -n_v:]
    print(f'using {pure_b.shape[1]} out of {v.shape[1]} eigenvectors, smallest eig {eigs[-n_v]}')
    
    print('normalizing eigenvectors')
    nside=128
    hits = hp.read_map(f'obsmat_nside{nside}/out/0/filterbin_hits.fits')
    mask = np.zeros_like(hits)
    mask[hits>0]=1
    mask_QU = np.concatenate([mask, mask])
    zeros = np.where(mask_QU==0)[0]
    for i in range(pure_b.shape[1]):    
        pure_b[:,i][zeros] = 0
        pure_b[:,i] /= np.linalg.norm(pure_b[:,i], axis=0)

    print('constructing purification matrix')
    pure_b = sp.sparse.csr_array(pure_b)
    pi_b = pure_b @ sp.sparse.linalg.inv(pure_b.T @ pure_b) @ pure_b.T
    sp.sparse.save_npz(f'{prefix}/pi_b_{pure_b.shape[1]}', pi_b)
    print('DONE')
    
if __name__ == '__main__':
    main()
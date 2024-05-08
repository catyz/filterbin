import numpy as np
import scipy as sp

def main():
    print('loading gep results')
    eigs = np.load(f'eigenvalues.npy')
    v = np.load(f'eigenvectors.npy')

    pass_cut = np.where(eigs >= 1.02)[0]
    pure_b = v[:, pass_cut]
    print(f'{pure_b.shape[1]} eigenvectors survive the cut')

    print('normalizing eigenvectors')
    nside=128
    hits = hp.read_map(f'obsmat_nside{nside}/out/0/filterbin_hits.fits')
    non_zero = np.where(hits!=0)[0]
    mask = np.zeros_like(hits)
    mask_QU = np.concatenate([mask, mask])
    zeros = np.where(mask_QU==0)[0]
    
    for i in range(pure_b.shape[1]):    
        pure_b[:,i][zeros] = 0
        pure_b[:,i] /= np.linalg.norm(pure_b[:,i], axis=0)

    print('constructing purification matrix')
    pure_b = sp.sparse.csr_array(pure_b)
    pi_b = pure_b @ sp.sparse.linalg.inv(pure_b.T @ pure_b) @ pure_b.T

    print('saving')
    sp.sparse.save_npz(f'pi_b_{nside}', pi_b)
    print('DONE')
    
if __name__ == '__main()__':
    main()
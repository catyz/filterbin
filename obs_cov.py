import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

def get_Cl(nside, mode):
    lmax = 3*nside-1
    l = np.arange(lmax+1)
    C_l = 1/l**2
    C_l[0:2] = 0
    
    # sigmab = hp.nside2resol(nside) 
    # fwhm = (8*np.log(2))**0.5 * sigmab
    # bl = hp.gauss_beam(fwhm, lmax)
    
    if mode == 'EE':
        C_l = np.array([np.zeros_like(C_l), C_l, np.zeros_like(C_l), np.zeros_like(C_l)]) # TT EE BB TE
    if mode == 'BB':
        C_l = np.array([np.zeros_like(C_l), np.zeros_like(C_l), C_l, np.zeros_like(C_l)])
    return C_l #* bl**2

def P_l2(ll, z):
    P2 = 3*(1-z**2)
    P3 = 5*z*P2
    P = [0,0,P2,P3]
    if len(ll) <= 4:
        P = P[:ll.max()+1]
    else:
        for l in range(4, ll.max()+1):
            P.append( ( (2*l-1) * z * P[l-1] - (l+1) * P[l-2] ) / (l-2) )
    return np.pad(np.array(P), (np.abs(ll.min()),0))

#bicep implementation
def get_a_ij(r_i, r_j):    
    z_hat = np.array([0,0,1])

    n_ij = np.cross(r_i, r_j)#/np.linalg.norm(np.cross(r_i, r_j))
    n_i = np.cross(r_i, z_hat)#/np.linalg.norm(np.cross(r_i, z_hat))

    a_ij = np.arctan2( np.dot(np.cross(n_ij, n_i), r_i), np.dot(n_ij, n_i) )
        
    return a_ij

def get_R(a):
    return np.array([
        [np.cos(2*a), -np.sin(2*a)],
        [np.sin(2*a), np.cos(2*a)]
    ])


def F_12(l, z):
    if np.round(z, 15) == 1:
        return 0.5*np.ones_like(l) 
    if np.round(z, 15) == -1:
        # print('z=-1')
        return 0.5*(-1)**l
    else:
        F = 2 * ( ((l+2)*z)/(1-z**2) * P_l2(l-1,z) - ((l-4)/(1-z**2) + l*(l-1)/2) * P_l2(l,z) ) / ((l-1)*l*(l+1)*(l+2))
        F[0:2] = 0     
        return F 
        
def F_22(l, z):
    if np.round(z, 15) == 1:
        return -0.5*np.ones_like(l) 
    if np.round(z, 15) == -1:
        # print('z=-1')
        return 0.5*(-1)**l
    else:
        F = 4 * ( (l+2)*P_l2(l-1,z) - (l-1)*z*P_l2(l,z) ) / ( (l-1)*l*(l+1)*(l+2)*(1-z**2) )
        F[0:2] = 0
        return F

def QQ_ij(l, Cl, z):
    return np.sum( (2*l+1) /(4*np.pi) * (F_12(l,z)*Cl[1] - F_22(l,z)*Cl[2]) )

def UU_ij(l, Cl, z):
    return np.sum( (2*l+1) /(4*np.pi) * (F_12(l,z)*Cl[2] - F_22(l,z)*Cl[1]) )

def get_M_ij(l, Cl, z):
    QQ = QQ_ij(l, Cl, z)
    UU = UU_ij(l, Cl, z)
    return np.array([
        [QQ, np.zeros_like(QQ)],
        [np.zeros_like(UU), UU]
    ])


def C_ana(nside, Cl, mask=None):
    lmax = 3*nside-1
    npix = 12*nside**2
    l = np.arange(lmax+1)

    if mask is None:
        pix = range(npix)    
        C = np.zeros((2*npix, 2*npix))
        
    else:
        pix = np.where(mask!=0)[0]
        row = []
        col = []
        data = []
    
    for i in tqdm(pix):
        for j in pix:
            if i == j:
                a_ij = 0
                a_ji = 0
                z = 1
            else:
                r_i = hp.pix2vec(nside, i)  
                r_j = hp.pix2vec(nside, j)
                z = np.dot(r_i, r_j)
                # print(i, j)
                a_ij = get_a_ij(r_i, r_j)
                a_ji = get_a_ij(r_j, r_i)
                
            R_ij = get_R(a_ij)
            R_ji = get_R(a_ji)
            
            M = get_M_ij(l, Cl, z)
            cov = R_ij.T @ M @ R_ji

            if mask is None:
                C[i][j] = cov[0][0]
                C[i][j+npix] = cov[0][1]
                C[i+npix][j] = cov[1][0]
                C[i+npix][j+npix] = cov[1][1]   
                
            else:
                row.append(i)
                col.append(j)
                data.append(cov[0][0])
    
                row.append(i)
                col.append(j+npix)
                data.append(cov[0][1])
    
                row.append(i+npix)
                col.append(j)
                data.append(cov[1][0])
    
                row.append(i+npix)
                col.append(j+npix)
                data.append(cov[1][1])

    if mask is not None:
        C = sp.sparse.coo_array((data, (row, col)), shape=(2*npix, 2*npix))
        
    assert np.abs(C-C.T).max() < 1e-15
    
    return C

def main():
    nside = 64
    hits = hp.ud_grade(hp.read_map('obsmat_nside128/out/0/filterbin_hits.fits'), nside)
    non_zero = np.where(hits!=0)[0]
    
    mask = np.zeros_like(hits)
    mask[non_zero] = 1
    
    Cl_EEonly = get_Cl(nside, 'EE')
    Cl_BBonly = get_Cl(nside, 'BB')
    
    C_E = C_ana(nside, Cl_EEonly, mask)
    C_B = C_ana(nside, Cl_BBonly, mask)
    
    sp.sparse.save_npz(f'C_E_{nside}', C_E)
    sp.sparse.save_npz(f'C_B_{nside}', C_B)
    
    print('DONE')

if __name__ == "__main__":
    main()
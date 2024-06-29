import numpy as np
import healpy as hp
import scipy as sp
from tqdm import tqdm
import os
import argparse
import pymaster as nmt

def get_Cl(nside, mode):
    lmax = 3*nside-1
    l = np.arange(lmax+1)
    C_l = 1/l**2
    C_l[0:2] = 0
    
    if mode == 'E':
        C_l = np.array([np.zeros_like(C_l), C_l, np.zeros_like(C_l), np.zeros_like(C_l)]) # TT EE BB TE
    if mode == 'B':
        C_l = np.array([np.zeros_like(C_l), np.zeros_like(C_l), C_l, np.zeros_like(C_l)])
    return C_l 

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nside',
        required=False,
        default=128
    )
    parser.add_argument(
        '--mode',
        required=True,
        type=str
    )
    parser.add_argument(
        '--aposize',
        required=False,
        default=10, 
        type=int
    )
    args = parser.parse_args()
    
    print('loading and apodizing R')
    npix = 12*args.nside**2
    qq, qu, uu = hp.read_map(f'obsmat_nside{args.nside}/out/0/filterbin_invcov.fits', field=[3,4,5])
    mask = np.zeros(npix)
    mask[qq!=0] = 1
    tr = qq + uu
    det = qq * uu - qu * qu
    weight = 0.5 * (tr - np.sqrt(tr ** 2 - 4 * det) ) 
    mask_apo = nmt.mask_apodization(weight, args.aposize, 'C2')
    mask_apo /= np.sqrt(np.mean(mask_apo**2)) 
    
    R_unapo = sp.sparse.load_npz(f'obsmat_nside{args.nside}/obsmat.npz')
    R_QU_unapo = R_unapo[npix:, npix:]
    
    Z = sp.sparse.diags_array(mask_apo)
    ZZ = sp.sparse.block_diag([Z, Z, Z])
    R = ZZ @ R_unapo
    R_QU = R[npix:, npix:]
    
    C_path = f'C_{args.mode}_{args.nside}.npz'
    if os.path.exists(C_path):        
        print(f'loading {C_path}')
        C = sp.sparse.load_npz(C_path)        
    else:
        Cl = get_Cl(args.nside, args.mode)
        C = C_ana(args.nside, Cl, mask)
        sp.sparse.save_npz(C_path, C)
        print(f'wrote {C_path}')
        
    print('observing cov')
    obs_C = R_QU @ C @ R_QU.T    
    sp.sparse.save_npz(f'apo{args.aposize}/obs_{C_path}', obs_C)    
    print(f'DONE: saved obs_{C_path}')

if __name__ == "__main__":
    main()
import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix, coo_matrix
import torch


from scipy.sparse.linalg import svds
import os
import json
import sys
from tqdm import tqdm

def inverseEntropy(wordBydoc):
    wordBydoc.data = np.log2(wordBydoc.data + 1)
    V = wordBydoc.shape[0]
    D = wordBydoc.shape[1]
    invLogD = 1/np.log2(D)
    word_freqs = np.array(wordBydoc.sum(axis=1).T)[0]

    word_entropy = -invLogD*np.array([float((wordBydoc[i, :]/word_freqs[i])*np.log2(np.array((wordBydoc[i, :]/word_freqs[i]).todense())[0] + 1e-6)) for i in range(V)])
    
    return diags(1 - word_entropy).dot(wordBydoc) #use sparse diag to avoid high memory load


def normalize_weights(C, alpha, k):
    V = C.shape[0]
    Ci = np.array(C.sum(axis=1).T)[0] + alpha*V
    Cj = np.array(C.sum(axis=0))[0] + alpha*V
    T = C.sum()
    Cj_hat = Cj**alpha
    Cj_hat_sum = Cj_hat.sum()
    W = lil_matrix(C.shape)
    print("Normalizing weight matrix...")
    for i in tqdm(range(V)):
        W[i, :] = (np.log2( ((Cj_hat_sum*C[i, :].todense() + alpha) / (Ci[i]*Cj_hat + 1e-8) ) + 1e-8) - np.log(k)).clip(min=0)
    return csr_matrix(W)



def normalize_weights_BandL(C, alpha, k):
    V = C.shape[0]
    Ci = np.array(C.sum(axis=1).T)[0] + alpha*V
    Cj = np.array(C.sum(axis=0))[0] + alpha*V
    T = C.sum()
    Cj_hat = Cj**alpha
    Cj_hat_sum = Cj_hat.sum()
    W = lil_matrix(C.shape)
    print("Normalizing weight matrix...")
    for i in tqdm(range(V)):
        W[i, :] = (Cj_hat_sum*C[i, :].todense() + alpha) / (Ci[i]*Cj_hat + 1e-8) 
    return csr_matrix(W)





















































































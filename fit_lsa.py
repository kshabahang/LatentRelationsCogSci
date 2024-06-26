import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix, coo_matrix
import torch
import argparse

from scipy.sparse.linalg import svds
import os
import json
import sys
from tqdm import tqdm

from normalization import *

def load_csr(fpath, shape, dtype=np.float64, itype=np.int32):
    data = np.fromfile(fpath + "_{}.data".format(str(dtype).split(".")[-1].strip("\'>")), dtype)
    indptr = np.fromfile(fpath + "_{}.indptr".format(str(itype).split(".")[-1].strip("\'>")), itype)
    indices = np.fromfile(fpath + "_{}.indices".format(str(itype).split(".")[-1].strip("\'>")), itype)

    return csr_matrix((data, indices, indptr), shape = shape)

with open("config.json", "r") as f:
    config = json.load(f)

config_params = {"CORPUS_NAME": "TASA2",
                 "DATA_PATH": "/home/kevin/GitRepos/LatentRelationsCogSci/rsc/",
                 "D_DIM": 1024,
                 "M_CONTEXT": 13,
                 "UseWordByDoc": False,
                 "UseWordByWord": True,
                 "ALPHA": 0.5,
                 "K_NEGATIVE": 15,
                 "Overwrite": False}
parser = argparse.ArgumentParser()
for param in config_params:
    parser.add_argument(f"--{param}", type=type(config_params[param]), default=None)
args = parser.parse_args()

if args.CORPUS_NAME is not None:
    CORPUS_NAME = args.CORPUS_NAME
else:
    CORPUS_NAME = config["CORPUS_NAME"]
if args.DATA_PATH is not None:
    DATA_PATH = args.DATA_PATH
else:
    DATA_PATH = config["DATA_PATH"]
if args.D_DIM is not None:
    D_DIM = args.D_DIM
else:
    D_DIM = config["MODELS"]["LSA"]["D_DIM"]
if args.M_CONTEXT is not None:
    M_CONTEXT = args.M_CONTEXT
else:
    M_CONTEXT = config["MODELS"]["LSA"]["M_CONTEXT"]
if args.UseWordByDoc is not None:
    UseWordByDoc = args.UseWordByDoc
else:
    UseWordByDoc = config["MODELS"]["LSA"]["UseWordByDoc"]
if args.UseWordByWord is not None:
    UseWordByWord = args.UseWordByWord
else:
    UseWordByWord = config["MODELS"]["LSA"]["UseWordByWord"]
if args.ALPHA is not None:
    ALPHA = args.ALPHA
else:
    ALPHA = config["MODELS"]["LSA"]["ALPHA"]
if args.K_NEGATIVE is not None:
    K_NEGATIVE = args.K_NEGATIVE
else:
    K_NEGATIVE = config["MODELS"]["LSA"]["K_NEGATIVE"]
if args.Overwrite is not None:
    Overwrite = args.Overwrite
else:
    Overwrite = config["MODELS"]["LSA"]["Overwrite"]




print(f"Using the following configuration\n CORPUS: {CORPUS_NAME}\n DATA_PATH: {DATA_PATH}\n D_DIM: {D_DIM}\n M_CONTEXT: {M_CONTEXT}\n UseWordByDoc: {UseWordByDoc}\n UseWordByWord: {UseWordByWord}\n ALPHA: {ALPHA}\n K_NEGATIVE: {K_NEGATIVE}")

if os.path.exists(f"{DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy") and not Overwrite:
    print("LSA model already exists. Set Overwrite to True in config.json to overwrite.")
    os.system(f"ls {DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy")
else:
    with open(DATA_PATH + f"{CORPUS_NAME}/vocab.txt", "r") as f:
        vocab = f.read().split('\n')
    
    if UseWordByDoc:
        with open(DATA_PATH + f"{CORPUS_NAME}/wXd_dims.dat", "r") as f:
            dims = f.read()
        
        
        
        dims = dims.split()
        dims = (int(dims[0]), int(dims[1]))
        print("Loading word-by-document matrix...")
        wordXdoc = load_csr(DATA_PATH + f"{CORPUS_NAME}/C_wbd", shape=dims, dtype=np.int64)
    
    
        #M = inverseEntropy(wordXdoc)
        #M  = normalize_weights(wordXdoc, 0.5, 1)
        M = normalize_weights_BandL(wordXdoc, ALPHA, K_NEGATIVE)
    
    elif UseWordByWord:
        wordXword = load_csr(DATA_PATH + f"{CORPUS_NAME}/C2_syntagmatic_{M_CONTEXT}_0_1", shape=(len(vocab), len(vocab)), dtype=np.int64)
    
    
        M = normalize_weights(wordXword, ALPHA, K_NEGATIVE)
    
    
    
    UseTorch = True
    
    print("Decomposing word-by-document matrix using SVD (this will take some time so go have a coffee)...")
    
    if not UseTorch:
        U, S, VT = svds(M.tocsr(), k = D_DIM)
    else:
        W = coo_matrix(M)
        values = W.data
        indices = np.vstack((W.row, W.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape=W.shape
        
        W_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        if torch.cuda.is_available():
            W_torch = W_torch.cuda()
     
        U, S, V = torch.svd_lowrank(W_torch, q=D_DIM, niter=100, M=None)
       
        if torch.cuda.is_available():
            U = U.detach().cpu().numpy()
            S = S.detach().cpu().numpy()
            VT = V.detach().cpu().numpy().T
        else:
            U = U.numpy()
            S = S.numpy()
            VT = V.numpy()
    
    
    print("Computing lower-dimensional word-space...")
    LSA = U.dot(np.sqrt(np.diag(S)))
    
    
    
    
    print("Saving to disk...")
    if not os.path.isdir(f"{DATA_PATH}{CORPUS_NAME}/LSA"):
        os.system(f"mkdir {DATA_PATH}/{CORPUS_NAME}/LSA")
    
    np.save(f"{DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}", LSA)






import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix, coo_matrix
import sparselinear as sl
import torch


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

CORPUS_NAME = config["CORPUS_NAME"]
DATA_PATH= config["DATA_PATH"]
D_DIM = config["MODELS"]["LSA"]["D_DIM"]
M_CONTEXT = config["MODELS"]["LSA"]["M_CONTEXT"]    
UseWordByDoc = config["MODELS"]["LSA"]["UseWordByDoc"]
UseWordByWord = config["MODELS"]["LSA"]["UseWordByWord"]
ALPHA = config["MODELS"]["LSA"]["ALPHA"]
K_NEGATIVE = config["MODELS"]["LSA"]["K_NEGATIVE"]
ETA = 0.8
BETA = 0.01


class ANet(torch.nn.Module):
    def __init__(self, V, M, lambda_max, e_max, ETA, BETA):
        super().__init__()
        self.Q = sl.SparseLinear(V, V)
        self.M = sl.SparseLinear(V, V)
        self.M.weight = torch.nn.Parameter(M)
        self.lambda_max = lambda_max
        self.e_max = e_max
        self.ETA = ETA
        self.BETA = BETA

    def DEN(self, x0):
        x1 = 1*x0
        xi = self.M(x1) - (self.lambda_max*self.ETA*(x1 * self.e_max).sum())*self.e_max.T + ((self.lambda_max + self.BETA)*(x0 * x1).sum())*x0.T
        xi /= torch.norm(xi)
        while(torch.dist(xi, x1) > eps):
            x1 = 1*xi
            xi = self.M(x1) - (self.lambda_max*self.ETA*(x1 * self.e_max).sum())*self.e_max.T + ((self.lambda_max + self.BETA)*(x0 * x1).sum())*x0.T 
            xi /= torch.norm(xi)

        return xi

    def forward(self, x):
        return self.DEN(self.Q(x) + x)



with open(DATA_PATH + f"{CORPUS_NAME}/vocab.txt", "r") as f:
    vocab = f.read().split('\n')


wordXword = load_csr(DATA_PATH + f"{CORPUS_NAME}/C2_syntagmatic_{M_CONTEXT}_0_1", shape=(len(vocab), len(vocab)), dtype=np.int64)

print("Normalizing counts using inverse entropy...")
M = normalize_weights(wordXword, ALPHA, K_NEGATIVE)


M = coo_matrix(M)
values = M.data
indices = np.vstack((M.row, M.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape=M.shape

M = torch.sparse.FloatTensor(i, v, torch.Size(shape))


V = len(vocab)
I = {vocab[i]:i for i in range(V)}

x0 = torch.normal(mean=torch.zeros(M.shape[0]), std=1/np.sqrt(M.shape[0]))
if torch.cuda.is_available():
    M = M.cuda()
    x0 = x0.cuda()

eps = 1e-6
x0 /= torch.norm(x0)
xi = x0.T @ M
xi /= torch.norm(xi)
while(torch.dist(xi, x0) > eps):
    x0 = 1*xi
    xi = x0.T @ M
    xi /= torch.norm(xi)

lambda_max = torch.norm(xi.T @ M)
e_max = 1*xi


net = ANet(V, M, lambda_max, e_max, ETA, BETA).cuda()

#### single probe
##initialize state
x0 *= 0
probe = "thomas declaration"
x0[torch.tensor([I[pi] for pi in probe.split() if pi in I])] = 1
x0 = x0.cuda()
x0 /= torch.norm(x0)
xi = net(x0)
#x1 = 1*x0
#xi = x1.T @ M - (lambda_max*ETA*(x1 * e_max).sum())*e_max.T + ((lambda_max + BETA)*(x0 * x1).sum())*x0.T
#xi /= torch.norm(xi)
#while(torch.dist(xi, x1) > eps):
#    x1 = 1*xi
#    xi = x1.T @ M - (lambda_max*ETA*(x1 * e_max).sum())*e_max.T + ((lambda_max + BETA)*(x0 * x1).sum())*x0.T 
#    xi /= torch.norm(xi)
#    
#
print( sorted(zip(xi.detach().cpu().numpy(), vocab), reverse=True)[:10] )

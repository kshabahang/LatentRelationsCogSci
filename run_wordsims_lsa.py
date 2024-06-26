import numpy as np
import json
import pandas as pd
from scipy.stats import spearmanr
import torch
import argparse
import sys, os 

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



print(f"Using the following configuration\n CORPUS: {CORPUS_NAME}\n DATA_PATH: {DATA_PATH}\n D_DIM: {D_DIM}\n M_CONTEXT: {M_CONTEXT}\n UseWordByDoc: {UseWordByDoc}\n UseWordByWord: {UseWordByWord}\n ALPHA: {ALPHA}\n K_NEGATIVE: {K_NEGATIVE}")




if not os.path.exists(f"{DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy"):
    raise ValueError(f"LSA model not found at {DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy. Fit the model using fit_lsa.py") 



with open(f"{DATA_PATH}/{CORPUS_NAME}/vocab.txt", "r") as f:
    vocab = f.read().split('\n')



I = {vocab[i]:i for i in range(len(vocab))}

vectors = np.load(f"{DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy") #TODO fully switch to torch
vectors = torch.from_numpy(vectors)
if torch.cuda.is_available():
    vectors = vectors.cuda()
vector_norms = torch.norm(vectors, dim=1)
vectors = torch.diag(1/vector_norms) @ vectors # normalize vectors

xls = pd.ExcelFile(f"{DATA_PATH}WordSims_Ready.xlsx", engine='openpyxl')
rs = {}
norm_sets = xls.sheet_names[:-1]
with open(f"{DATA_PATH}/{CORPUS_NAME}/LSA/wordsims_LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.dat", "w") as f:
    f.write("NormSet WordA WordB CosineSim HumanSim\n")
    for k in range(len(norm_sets)):
        df_k = pd.read_excel(xls, norm_sets[k])
        wsA = df_k.iloc[:, 0].values
        wsB = df_k.iloc[:, 1].values
        sims = df_k.iloc[:, 2].values
        out = []
        sims_keep = []
        for i in range(len(wsA)):
            wA = wsA[i]
            wB = wsB[i]
            if wA in I and wB in I:
                vcos = vectors[I[wA]].dot(vectors[I[wB]])
                out.append(float(vcos.detach().cpu().numpy()))
                sims_keep.append(sims[i])
                f.write(f"{norm_sets[k]} {wA} {wB} {vcos} {sims[i]}\n")
        (r, p) = spearmanr(sims_keep, out)
        rs[norm_sets[k]] = r
        print(f"{norm_sets[k]}: {r} (p={p})")

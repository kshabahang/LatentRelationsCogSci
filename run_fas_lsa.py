from utilities import prepareFAs, vecs2cos
import pickle
import numpy as np
import json
import sys, os
import argparse
from tqdm import tqdm
from functools import reduce

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



with open(DATA_PATH + "usf_fa_set.pkl", "rb") as f:
    fas = pickle.load(f)


with open(f"{DATA_PATH}/{CORPUS_NAME}/vocab.txt", "r") as f:
    vocab = f.read().split('\n')



I = {vocab[i]:i for i in range(len(vocab))}
fa_vocab, cue_resps_try, cue_resps = prepareFAs(fas, I)
idxs = np.array([I[fa_vocab[i]] for i in range(len(fa_vocab))])
I = {fa_vocab[i]:i for i in range(len(fa_vocab))}

if not os.path.exists(f"{DATA_PATH}/{CORPUS_NAME}/LSA/wXw_fas_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy"):
    print("Computing wXw for LSA")

    vectors = np.load(f"{DATA_PATH}/{CORPUS_NAME}/LSA/LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy")
    wXw = vecs2cos(vectors[idxs]) - np.eye(len(fa_vocab))
    
    ###sanity check
    I_fa = {fa_vocab[i]:i for i in range(len(fa_vocab))}
    
    top_st, top_w = zip(*sorted(zip(wXw[I_fa['halt']], fa_vocab), reverse=True)[:10])
    print("Cue: halt")
    for i in range(10):
        print(top_w[i], top_st[i])
    
    
    ###TODO check if we can share the same vocab across models to simplify the code
    with open("{}/{}/LSA/fa_vocab.txt".format(DATA_PATH, CORPUS_NAME), "w") as f:
        f.write('\n'.join(fa_vocab))
    
    
    np.save(f"{DATA_PATH}/{CORPUS_NAME}/LSA/wXw_fas_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy", wXw)
else:
    print("Loading wXw for LSA")
    wXw = np.load(f"{DATA_PATH}/{CORPUS_NAME}/LSA/wXw_fas_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.npy")


with open(f"{DATA_PATH}/{CORPUS_NAME}/LSA/fas_LSA_{M_CONTEXT}_{D_DIM}_{int(UseWordByDoc)}_{int(UseWordByWord)}_{ALPHA}_{K_NEGATIVE}.dat", "w") as f:
    f.write("model item pRespGivenCue rankActivations rankProbs cue resp\n")
    #for targ_rank in range(1,21):
    for targ_rank in [1]:
        ranks_dir=[]
        ranks_den = []
        ranks_w2v = []
        ranks_lsa = []
        ranks_topics= []
    
        firstps = []
        cue_resps = []
        for i in tqdm(range(len(cue_resps_try))):
            (p, cue, resp1, resps, ps) = cue_resps_try[i]        
            firstps.append(p)

            if len(resps) > targ_rank - 1:
                if targ_rank == 1:
                    resp_i = resp1
                else:
                    resp_i = resps[targ_rank-1]        
                r_lsa = list(np.argsort(wXw[I[cue]])[::-1]).index(I[resp_i]) + 1
                ranks_lsa.append(r_lsa)
            else:
                print(cue, resp1)
        
            #cue_resps.append("{}->{}".format(cue, resp1))
            cue_resps.append((cue, resp1))
        
        models = ["LSA"]#, "Topics", "Direct", "DEN", "w2v"]
        ranks = [ranks_lsa]#, ranks_topics, ranks_dir, ranks_den, ranks_w2v]
        for i in range(len(models)):
            ranks_i = np.array(ranks[i])
            print(models[i], np.median(ranks_i), round(100*sum(ranks_i == 1)/len(ranks_i), 2))
        
        
        model = reduce(lambda a,b:a+b, [[models[i]]*len(ranks[0]) for i in range(len(models))])
        rank = reduce(lambda a,b : a+b, ranks)
        p = firstps*len(models)
            
        for i in range(len(rank)):
            f.write(f"{model[i]} {i % len(ranks[0])} {p[i]} {rank[i]} {targ_rank} {cue_resps[i][0]} {cue_resps[i][1]}\n")



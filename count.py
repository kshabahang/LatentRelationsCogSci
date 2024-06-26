import sys, os
from collections import Counter
from itertools import combinations
from functools import reduce
import pickle


from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
import numpy as np


def save_csr(matrix, filename):
    matrix.data.tofile(filename + "_"+ str(type(matrix.data[0])).split(".")[-1].strip("\'>") + ".data")
    matrix.indptr.tofile(filename + "_"+ str(type(matrix.indptr[0])).split(".")[-1].strip("\'>") + ".indptr")
    matrix.indices.tofile(filename + "_" +str(type(matrix.indices[0])).split(".")[-1].strip("\'>") + ".indices")



CORPUS_NAME="TASA2"
DATA_PATH="/home/kevin/GitRepos/LatentRelationsCogSci/rsc/"
MIN_FREQ = 10
K_BANKS = 2
IDX=0 #Index of process (for parallelization)
CHU=1 #number of processes
M_CONTEXT=31

CountOrder = False
CountBoG = True
CountWordByDoc = False
CountWordByWord = True
SaveCounts = True

with open(DATA_PATH + f"/{CORPUS_NAME}/{CORPUS_NAME}.txt", "r") as f:
    corpus = f.read()#.splitlines()
if not os.path.exists(f"{DATA_PATH}{CORPUS_NAME}/vocab.txt"):
    print("Creating vocab")
    wf = Counter(corpus.split()) # word frequency
    vocab, _ = zip(*list(filter(lambda x:x[1] >= MIN_FREQ, wf.items())))
    vocab = list(vocab) + ['<LOWFREQ>'] 
    f = open(f"/{DATA_PATH}/vocab.txt", "w")
    f.write("\n".join(vocab))
    f.close()
else:
    print("Loading vocab")
    with open(DATA_PATH + f"/{CORPUS_NAME}/vocab.txt", "r") as f:
        vocab = f.read().splitlines()


I = {vocab[i]:i for i in range(len(vocab))}
V = len(vocab)
print("Vocab size: ", V)

episodes = corpus.splitlines()
H = len(episodes)/CHU
episodes = episodes[int(IDX*H):int((IDX+1)*H)]


#####MAP WORDS TO INDEXES####
episodes_int = []
episodes_int_by_doc = []

print("Loading corpus...")
for i in range(len(episodes)):
    episode = episodes[i].strip().split()
    if len(episode) > 1:
        episode_int = []
        for j in range(len(episode)):
            if episode[j] != '_' and episode[j] in I:
                episode_int.append(I[episode[j]])
            else:
                episode_int.append(len(vocab)-1) #<LOWFREQ> dummy token
        episodes_int += episode_int
        episodes_int_by_doc.append(episode_int)
    else:
        print(episode)

if CountOrder:
    print("Computing ngram counts. Number of banks: {}".format(K_BANKS))
    C = lil_matrix((K_BANKS*V, K_BANKS*V), dtype=np.int64)
    for k in range(K_BANKS):
        for l in range(k,K_BANKS):
            c = Counter([(episodes_int[i+k], episodes_int[i+l]) for i in range(len(episodes_int) - max(k, l) - 1)])
            idx = c.keys()
            Ckl = coo_matrix( (list( map( lambda i : c[i], idx) ), list(zip(*idx)) ), shape = (V, V)).tolil()
            for i in range(V):
                for j in range(len(Ckl.rows[i])):
                    C[int(k*V)+i, int(l*V) + Ckl.rows[i][j]] = Ckl.data[i][j]
                    C[int(l*V) + Ckl.rows[i][j], int(k*V)+i] = Ckl.data[i][j]
            if l >= k and not os.path.exists(f"FW{l-k}_{IDX}_{CHU}_int32.indptr"):
                save_csr(csr_matrix(Ckl), f"FW{l-k}_{IDX}_{CHU}")
    print("Number of windows: {}".format(len(episodes_int)-K_BANKS))
    print("Saving to disk.")
    os.system(f'mv FW* "{DATA_PATH}/{CORPUS_NAME}/"')
    save_csr(csr_matrix(C), f"C_order_{K_BANKS}_{IDX}_{CHU}")
    os.system(f'mv C_order_{K_BANKS}_* "{DATA_PATH}/{CORPUS_NAME}/"')


if CountBoG:
    if CountWordByDoc:

        print("Counting word-by-document co-occurrence")
        ###word-by-doc counts NOTE: not implemented for multiple chunks
        C_wbd = lil_matrix((V, len(episodes_int_by_doc)), dtype=np.int64)
        for i in range(len(episodes_int_by_doc)):
            for w_idx in episodes_int_by_doc[i]:
                C_wbd[w_idx, i] += 1

        save_csr(csr_matrix(C_wbd), "C_wbd")
        os.system(f"mv C_wbd* {DATA_PATH}/{CORPUS_NAME}/")
        f = open(f"{DATA_PATH}/{CORPUS_NAME}/wXd_dims.dat", "w")
        f.write(f"{C_wbd.shape[0]} {C_wbd.shape[1]}")
        f.close()

    if CountWordByWord:
        assert((M_CONTEXT - 1) % 2 == 0)
        print("Computing syntagmatic counts.")
        c = Counter()
        C2 = lil_matrix((V,V), dtype=np.int64) #word2vec style counts
        for k in range(len(episodes_int) - M_CONTEXT):
            window = episodes_int[k:k+M_CONTEXT]
            c.update([c for c in combinations(window, 2)]) #all pair-wise associations between same-window items

            for idx, word in enumerate(window):
                for i in range(1, int((M_CONTEXT - 1)/2) + 1):
                    if idx - i >= 0:
                        C2[word, window[idx - i]] += 1
                    if idx + i < len(window):
                        C2[word, window[idx + i]] += 1

        print("Finished counting. Converting to matrix.")
        if SaveCounts:
            f = open(f"counts_syntagmatic_{M_CONTEXT}_{IDX}_{CHU}.pkl", "wb")
            pickle.dump(c, f)
            f.close()


        idx = c.keys()
        C = coo_matrix( (list( map( lambda i : c[i], idx) ), list(zip(*idx)) ), shape = (V, V)).tolil()
        print("Saving to disk.")
        save_csr(csr_matrix(C), f"C_syntagmatic_{M_CONTEXT}_{IDX}_{CHU}")
        save_csr(csr_matrix(C2), f"C2_syntagmatic_{M_CONTEXT}_{IDX}_{CHU}")
        os.system(f'mv C2_syntagmatic_{M_CONTEXT}_* "{DATA_PATH}/{CORPUS_NAME}/"')
        os.system(f'mv C_syntagmatic_{M_CONTEXT}_* "{DATA_PATH}/{CORPUS_NAME}/"')                                                                                                                                                                                
        os.system(f'mv counts*.pkl "{DATA_PATH}/{CORPUS_NAME}/"')

import numpy as np
from scipy.sparse import csr_matrix  
from functools import reduce


def prepareFAs(fas, I):
    cue_resps = []
    K = 1
    for cue in fas.keys():
        resps, ps = zip(*fas[cue])
        idxs = np.argsort(ps)[::-1]
        idx = np.argmax(ps)
        first_p = ps[idx]
        first_resps = []
        if len(idxs) >= K:
            for j in range(len(resps)):
                first_resps.append(resps[idxs[j]])

            first_resp = resps[idx]
            #fa_vocab.append(first_resp)

            cue_resps.append((first_p, cue, first_resp, first_resps, ps))



    cue_resps = sorted(cue_resps)[::-1]

    cue_resps_try = []
    fa_vocab = []
    k = 0
    i = 0
    while(k < 1000000 and i < len(cue_resps)):
        (p, cue, resp, resps, ps) = cue_resps[i]
        if cue in I and resp in I:
            cue_resps_try.append((p, cue, resp, resps, ps ))
            fa_vocab.append(cue)
            fa_vocab.append(resp)
            k += 1
            i += 1
        else:
            i += 1
    fa_vocab = list(set(fa_vocab))

    return sorted(fa_vocab), cue_resps_try, cue_resps


def vecs2cos(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    return np.diag(1/norms).dot(vectors.dot(vectors.T)).dot(np.diag(1/norms))

def loadGibbsSamples(k_topics, run_num, n_chains, n_samples, mem_path, corpus, ref_vocab):

    tt_files = list(reduce(lambda a,b : a+b, [['tt{}_{}.npy'.format(j+1, i) for i in range(n_samples)] for j in range(n_chains)]))

    tts = np.zeros((k_topics, len(ref_vocab), n_samples*n_chains))

    count_samples = 0
    for chain_num in range(n_chains):
        f = open(mem_path + "{}/Topics/vocab{}.txt".format(corpus, chain_num+1), 'r')
        vocab = f.read().split('\n')
        f.close()
        V = len(vocab)
        I = {vocab[i]:i for i in range(V)}

        idx = np.array([I[ref_vocab[i]] for i in range(len(ref_vocab))])

        tts[:, :, count_samples] = np.load(mem_path + "{}/Topics/tt{}_{}.npy".format(corpus, chain_num+1, 0)).T[:, idx]
        count_samples += 1

        for k in range(1, n_samples):
            tts[:, :, count_samples] = np.load(mem_path + "{}/Topics/tt{}_{}.npy".format(corpus, chain_num+1, k)).T[:, idx]
            count_samples += 1
    return tts


def save_csr(matrix, filename):
    matrix.data.tofile(filename + "_"+ str(type(matrix.data[0])).split(".")[-1].strip("\'>") + ".data")
    matrix.indptr.tofile(filename + "_"+ str(type(matrix.indptr[0])).split(".")[-1].strip("\'>") + ".indptr")
    matrix.indices.tofile(filename + "_" +str(type(matrix.indices[0])).split(".")[-1].strip("\'>") + ".indices")

def load_csr(fpath, shape, dtype=np.float64, itype=np.int32):
    data = np.fromfile(fpath + "_{}.data".format(str(dtype).split(".")[-1].strip("\'>")), dtype)
    indptr = np.fromfile(fpath + "_{}.indptr".format(str(itype).split(".")[-1].strip("\'>")), itype)
    indices = np.fromfile(fpath + "_{}.indices".format(str(itype).split(".")[-1].strip("\'>")), itype)

    return csr_matrix((data, indices, indptr), shape = shape)


def proj(a, v):
    return (a.dot(v)/v.dot(v))*v




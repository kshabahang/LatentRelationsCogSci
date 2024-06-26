import numpy as np
import itertools
import os
import sys
from tqdm import tqdm
import torch
import multiprocessing as mp







def sampler(args):
#def sampler(docs, tokens, topics, N, V, K, D, alpha, beta, burnin, thinning, samples):

    docs, tokens, topics, N, V, K, D, alpha, beta, burnin, thinning, samples = args
    print("Starting Gibbs sampling")
    """
    Perform Gibbs sampling to sample topics, beginning from the seed "topics".

    docs: document identifier
    tokens: token indices
    N, V, K, D, alpha, beta: topic model parameters
    burnin: number of samples to burn in
    thinning: thinning interval
    samples: number of samples to take.

    output is (samples x N) array of topic assignments for each 
    of the N tokens in the data
    """


    tok_topic_mat = np.zeros((V, K))

    tok_topic_agg = np.zeros(K)

    doc_topic_mat = np.zeros((D, K))


    sampled_topics = np.zeros((samples, N))




    iter = 0

    sample = 0

    maxiter = burnin + thinning*samples

    p = np.zeros(K)



    #initialize topic counts

    for i in range(N):
        tok_topic_mat[tokens[i], topics[i]] += 1
        doc_topic_mat[docs[i], topics[i]] += 1

    tok_topic_agg = np.sum(tok_topic_mat, axis=0)

    while iter < maxiter:

        for i in tqdm(range(N)):

            token = tokens[i]
            doc = docs[i]

            old_topic = topics[i]

            tok_topic_mat[token, old_topic] -= 1
            tok_topic_agg[old_topic] -= 1
            doc_topic_mat[doc, old_topic] -= 1

            for j in range(K):
                p[j] = (tok_topic_mat[token, j] + beta) / \
                       (tok_topic_agg[j] + beta*V) * (doc_topic_mat[doc, j] + alpha)
            for j in range(1,K):
                p[j] += p[j-1]

            new_topic = multinomial_sample(p, K)

            tok_topic_mat[token, new_topic] += 1
            tok_topic_agg[new_topic] += 1
            doc_topic_mat[doc, new_topic] += 1

            topics[i] = new_topic

        iter += 1
        if iter - burnin > 0 and (iter - burnin) % thinning == 0: 
            sampled_topics[sample, ::1] = topics
            sample += 1
        if iter % 10 == 0:
            print("Iteration %d of (collapsed) Gibbs sampling" % iter)



    return np.asarray(sampled_topics)

def tt_comp(args):
#def tt_comp(tokens, topics, N, V, K, beta):

    """
    Compute term-topic matrix from topic assignments
    """

    #cdef DTYPE_t [:, ::1] tok_topic_mat = np.zeros((V, K), dtype=DTYPE)
    #cdef DTYPE_t [:] tok_topic_agg = np.zeros(K, dtype=DTYPE)
    #cdef FTYPE_t [:, ::1] tt = np.zeros((V, K), dtype=FTYPE)
    #cdef int i, v, k

    tokens, topics, N, V, K, beta = args
    tok_topic_mat = np.zeros((V, K))
    tok_topic_agg = np.zeros(K)
    tt = np.zeros((V, K))
    


    for i in range(N):
        tok_topic_mat[tokens[i], topics[i]] += 1

    tok_topic_agg = np.sum(tok_topic_mat, axis=0)

    for v in range(V):
        for k in range(K):
            tt[v, k] = (tok_topic_mat[v, k] + beta) / \
                       (tok_topic_agg[k] + V*beta) 

    return np.asarray(tt)



#def dt_comp(docs, topics, N, K, D, alpha):
def dt_comp(args):


    """
    Compute document-topic matrix from topic assignments
    """
    docs, topics, N, K, D, alpha = args

    #cdef DTYPE_t [:, ::1] doc_topic_mat = np.zeros((D, K), dtype=DTYPE)
    #cdef DTYPE_t [:] doc_topic_agg = np.zeros(D, dtype=DTYPE)
    #cdef FTYPE_t [:, ::1] dt = np.zeros((D, K), dtype=FTYPE)
    #cdef int i, d, k

    doc_topic_mat = np.zeros((D, K))
    dt = np.zeros((D, K))


    for i in range(N):
        doc_topic_mat[docs[i], topics[i]] += 1

    doc_topic_agg = np.sum(doc_topic_mat, axis=1)

    for d in range(D):
        for k in range(K):
            dt[d, k] = (doc_topic_mat[d, k] + alpha) / \
                       (doc_topic_agg[d] + K*alpha) 

    return np.asarray(dt)



def multinomial_sample(p, K ):

    """
    Sample from multinomial distribution with probabilities p and length K
    """
    rnd = np.random.random_sample()*p[K-1]
    for new_topic in range(K):
        if p[new_topic] > rnd:
            break

    return new_topic







CORPUS_NAME="TASA2"
DATA_PATH="/home/kevin/GitRepos/LatentRelationsCogSci/rsc/"


with open(DATA_PATH + f"{CORPUS_NAME}/{CORPUS_NAME}.txt", "r") as f:
    docs = f.readlines()


docs = [docs[i].split() for i in range(len(docs))]


K = 1700
D = len(docs)
docs = docs

doc_list = list(itertools.chain(*docs))
token_key = {}
for i, v in enumerate(set(doc_list)):
    token_key[v] = i
V = len(token_key)

tokens = np.array([token_key[t] for t in doc_list])
N = tokens.shape[0]
topic_seed = np.random.random_integers(0, K-1, N)

docid = [[i]*len(d) for i, d in enumerate(docs)]
docid = np.array(list(itertools.chain(*docid)))

alpha = 50/K
beta = 200/V


burnin=800
thinning=100
samples=8
append=True


# Estimate topics via Gibbs sampling.
# burnin: number of iterations to allow chain to burn in before sampling.
# thinning: thinning interval between samples.
# samples: number of samples to take.
# Total number of samples = burnin + thinning * samples
# If sampled topics already exist and append = True, extend chain from
# last sample.
# If append = False, start new chain from the seed.
chain_num = 0#sys.argv[1]
outpath = DATA_PATH
vocab = [w for w in token_key.keys()]
if not os.path.isdir(f"{DATA_PATH}{CORPUS_NAME}/Topics"):
    os.system(f"mkdir {DATA_PATH}{CORPUS_NAME}/Topics")
with open(f"{DATA_PATH}{CORPUS_NAME}/Topics/vocab{chain_num}.txt", "w") as f:
    f.write("\n".join(vocab))


runGibbs = True
if runGibbs:

    seed = np.copy(topic_seed)
    print("Starting Gibbs sampling")
    args = [docid, tokens, seed, N, V, K, D, alpha, beta, burnin, thinning, 1]
    with mp.Pool() as p:
        sampled_topics = np.asarray(p.map(sampler, [args]*samples)).astype(int).squeeze()

    print("Computing term-topic and document-topic matrices")
    args = []
    for s in range(samples):
        args.append([tokens, sampled_topics[s, :], N, V, K, beta])
    with mp.Pool() as p:
        tts = p.map(tt_comp, args)
    for tt in tts:
        tt = np.asarray(tt)
        np.save(f"{DATA_PATH}{CORPUS_NAME}/Topics/tt{chain_num}_{s}.npy", tt[:, :])

    args = []
    for s in range(samples):
        args.append([docid, sampled_topics[s, :], N, K, D, alpha])
    with mp.Pool() as p:
        dts = p.map(dt_comp, args)
    for dt in dts:
        dt = np.asarray(dt)
        np.save(f"{DATA_PATH}{CORPUS_NAME}/Topics/dt{chain_num}_{s}.npy", dt[:, :])

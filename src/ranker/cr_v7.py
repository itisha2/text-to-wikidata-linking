# version 7: graph embedding based coherence.
# node embeddings: NetMF-style (normalized adjacency walks + log, SVD), numpy
#
# like v2 the ranking is v1's text ranking plus a graph coherence score, but
# the coherence is structural instead of verbalized: nodes of the candidate
# graph get embedded (degree-normalized walk matrix, log damped, SVD) and each
# candidate is scored by cosine between its node vector and the centroid of
# the other mentions' candidate vectors. the normalization and log damp hub
# degrees, which was the popularity bias in v2/v4's graph bonus.
import networkx as nx
import numpy as np

from cr_v1 import candidate_text
from utils import W_GRAPH, candidate_graph, embed_rank, evaluate

DIM, WINDOW = 32, 5
_embeddings = {}  # id(G) -> (node -> row index, vectors); graphs are cached in utils


def node_embeddings(G):
    """NetMF-lite: M = mean of the first WINDOW powers of the degree-normalized
    adjacency (random walk co-occurrence), log damped, then SVD."""
    nodes = list(G)
    A = nx.to_numpy_array(G, nodelist=nodes)
    P = A / np.maximum(A.sum(1, keepdims=True), 1)
    M, Pk = np.zeros_like(P), np.eye(len(nodes))
    for _ in range(WINDOW):
        Pk = Pk @ P
        M += Pk / WINDOW
    M = np.log1p(M * len(nodes))
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    d = min(DIM, len(S))
    vecs = U[:, :d] * np.sqrt(S[:d])
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    return {n: i for i, n in enumerate(nodes)}, vecs


def graph_embed_bonus(mention, context, candidates):
    """W_GRAPH * cosine(candidate's node vector, centroid of the other
    mentions' candidate vectors). 0 for candidates not in the graph."""
    G, others = candidate_graph(mention, context)
    if not len(G) or not others:
        return np.zeros(len(candidates))
    if id(G) not in _embeddings:
        _embeddings[id(G)] = node_embeddings(G)
    index, vecs = _embeddings[id(G)]
    other_vecs = [vecs[index[q]] for qids in others for q in qids if q in index]
    if not other_vecs:
        return np.zeros(len(candidates))
    centroid = np.mean(other_vecs, axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)
    return W_GRAPH * np.array(
        [vecs[index[c["qid"]]] @ centroid if c["qid"] in index else 0.0
         for c in candidates])


def rank(mention, context, candidates):
    return embed_rank(f"{mention}. {context}", candidates, candidate_text, mention,
                      extra=graph_embed_bonus(mention, context, candidates))


if __name__ == "__main__":
    evaluate(rank)

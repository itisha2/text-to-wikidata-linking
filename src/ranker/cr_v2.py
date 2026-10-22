# version 2: semantic similarity with graph context.
# embedding model: embeddinggemma:latest, graph: networkx
#
# like v1 (label + description + aliases), but each candidate's evidence is
# retrieved from the networkx candidate graph (utils.graph_facts): its
# neighbourhood as facts, sentence-relevant neighbours first. document level
# graphs only (doc_only), so on sentence-grouped stores (no doc_id) v2 ranks
# exactly like v1.
from cr_v1 import candidate_text as base_text
from utils import embed_rank, evaluate, graph_bonus, graph_candidate_text


def candidate_text(c):
    # flat 1-hop triples, the non-graph version of the text (v5 uses this)
    facts = "; ".join(f"{p[0]} {o[0]}" for s, p, o in c["triples"]
                      if p[0] not in ("label", "description", "alias"))
    return f"{base_text(c)}. {facts}"


def rank(mention, context, candidates):
    text = graph_candidate_text(mention, context, base_text, doc_only=True)
    return embed_rank(f"{mention}. {context}", candidates, text, mention,
                      extra=graph_bonus(mention, context, candidates, doc_only=True))


if __name__ == "__main__":
    evaluate(rank)

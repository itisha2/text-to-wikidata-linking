# version 5: AMR graph based.
# amr parser: amrlib parse_xfm_bart_base, embedding model: embeddinggemma
#
# parses the sentence into an AMR graph, renders the graph as
# "concept relation concept" facts, and embeds that against each candidate's
# graph text (label + description + aliases + triples, same as v2). so it is
# graph to graph comparison instead of sentence to graph.
from functools import lru_cache

import penman

from cr_v2 import candidate_text
from utils import embed_rank, evaluate

_stog = None


@lru_cache(maxsize=None)
def amr_facts(sentence):
    """sentence -> AMR graph -> 'concept relation concept' facts."""
    global _stog
    if _stog is None:
        import amrlib
        _stog = amrlib.load_stog_model()
    g = penman.decode(_stog.parse_sents([sentence])[0])
    concept = {v: c for v, _, c in g.instances()}
    facts = [f"{concept.get(s, s)} {r.strip(':')} {concept.get(t, t)}"
             for s, r, t in g.edges() + g.attributes()]
    return "; ".join(facts)


def rank(mention, context, candidates):
    return embed_rank(f"{mention}. {amr_facts(context)}", candidates, candidate_text, mention)


if __name__ == "__main__":
    evaluate(rank)

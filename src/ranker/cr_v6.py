# version 6: AMR graph based, d1 ranked, with graph context.
# amr parser: amrlib parse_xfm_bart_base, decision model: d1-omni:600m
#
# like v5 the sentence is parsed into AMR "concept relation concept" facts,
# but the selection is a d1 choice question, and each candidate option carries
# the evidence retrieved from the networkx sentence graph (same as v2/v4).
from cr_v1 import candidate_text as base_text
from cr_v5 import amr_facts
from utils import d1_rank, evaluate, graph_bonus, graph_candidate_text


def rank(mention, context, candidates):
    text = graph_candidate_text(mention, context, base_text)
    return d1_rank(mention, amr_facts(context), candidates, text,
                   extra=graph_bonus(mention, context, candidates))


if __name__ == "__main__":
    evaluate(rank)

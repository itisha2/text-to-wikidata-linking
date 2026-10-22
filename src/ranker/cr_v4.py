# version 4: decision model based ranker with graph context.
# decision model: d1-omni:600m (utils.D1_MODEL), graph: networkx
#
# like v3, but each candidate option carries the evidence retrieved from the
# networkx sentence graph (utils.graph_facts), so d1 sees the graph
# neighbourhood too. same graph context as v2.
from cr_v1 import candidate_text as base_text
from utils import d1_rank, evaluate, graph_bonus, graph_candidate_text


def rank(mention, context, candidates):
    text = graph_candidate_text(mention, context, base_text)
    return d1_rank(mention, context, candidates, text,
                   extra=graph_bonus(mention, context, candidates))


if __name__ == "__main__":
    evaluate(rank)

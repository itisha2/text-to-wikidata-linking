# simple version 1: semantic similarity based.
# embedding model: embeddinggemma:latest
#
# Embeds the mention sentence and each candidate's "label: description (aliases)",
# ranks candidates by cosine similarity.
# Candidates come as the JSON list the candidate generator produces
# (qid, label, description, aliases, triples, triple_text).
from utils import embed_rank, evaluate


def candidate_text(c):
    aliases = f" (also known as {', '.join(c['aliases'])})" if c["aliases"] else ""
    return f"{c['label']}: {c['description']}{aliases}"


def rank(mention, context, candidates):
    return embed_rank(f"{mention}. {context}", candidates, candidate_text, mention)


if __name__ == "__main__":
    evaluate(rank)

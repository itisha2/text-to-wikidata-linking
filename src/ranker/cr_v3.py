# version 3: decision model based ranker.
# decision model: d1-omni:600m (utils.D1_MODEL), via Ollama /v1/systemone
#
# one choice question per mention: state = mention + sentence, options = the
# candidates (label + description + aliases, same text as v1). d1 gives a
# calibrated probability per candidate in a single forward pass, which is
# used directly as the ranking score.
from cr_v1 import candidate_text
from utils import d1_rank, evaluate


def rank(mention, context, candidates):
    return d1_rank(mention, context, candidates, candidate_text)


if __name__ == "__main__":
    evaluate(rank)

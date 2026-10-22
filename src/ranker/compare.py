# compare all the rankers on the same snapshot of candidate store rows,
# each run with every model in its list.
# CANDIDATE_DB=candidate_store_aida.db switches to the AIDA store, an optional
# cli argument limits the run to the first N rows (compare.py 4).
# prints top-1 accuracy / MRR, overall and split by ambiguity.
import json
import sqlite3
import sys
import time

import cr_v1
import cr_v2
import cr_v3
import cr_v4
import cr_v5
import cr_v6
import cr_v7
import utils

EMBED_MODELS = ["embeddinggemma"]
MODELS = {"embed": EMBED_MODELS, "d1": [utils.D1_MODEL]}

# (name, module, which model knob the ranker uses)
RANKERS = [("v1 embed", cr_v1, "embed"), ("v2 embed+graph", cr_v2, "embed"),
           ("v3 d1", cr_v3, "d1"), 
           ("v4 d1+graph", cr_v4, "d1"),
           ("v5 amr graph", cr_v5, "embed"), ("v6 amr+d1", cr_v6, "d1"),
           ("v7 graph-embed", cr_v7, "embed")]


def cell(ranks):
    n = len(ranks)
    if not n:
        return "-"
    acc = sum(r == 1 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return f"{acc:.1%} / {mrr:.2f}"


def main():
    # cli: a number = first N rows, 'amb' / 'non' = only that use case
    rows = utils.load_rows()
    print(f"{utils.DB.name}: {len(rows)} rows (top-1 accuracy / MRR)\n")
    print(f"{'ranker':36} {'overall':>14} {'ambiguous':>14} {'non-ambiguous':>14}")

    for name, mod, kind in RANKERS:
        for model in MODELS[kind]:
            if kind == "embed":
                utils.EMBED_MODEL = model
            t0 = time.time()
            results = []  # (ambiguous, rank of gold or None)
            for mention, context, gold, amb, cands_json in rows:
                ranked = mod.rank(mention, context, json.loads(cands_json))["ranked"]
                gold_rank = next((i + 1 for i, c in enumerate(ranked) if c["qid"] == gold), None)
                results.append((amb, gold_rank))
            overall = [r for _, r in results]
            amb_t = [r for a, r in results if a == "True"]
            amb_f = [r for a, r in results if a == "False"]
            print(f"{f'{name} [{model}]':36} {cell(overall):>14} {cell(amb_t):>14}"
                  f" {cell(amb_f):>14}   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

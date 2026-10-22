# compare v2 (embed+triples) and v5 (amr graph) with embeddinggemma on the
# current snapshot of candidate_store.db (the 200-row sample).
# prints top-1 accuracy / MRR, overall and split by ambiguity.
import json
import sqlite3
import time

import pandas as pd

import cr_v2
import cr_v5
import utils

RANKERS = [("v2 embed+triples", cr_v2), ("v5 amr graph", cr_v5)]
utils.EMBED_MODEL = "embeddinggemma"
DATASET = utils.ROOT / "evaluation/dataset_3_sen_level/dataset_3_sampled_100_v2.csv"


def cell(ranks):
    n = len(ranks)
    acc = sum(r == 1 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return f"{acc:.1%} / {mrr:.2f}"


def main():
    # only the current sample's rows, the table also has rows from earlier samples
    df = pd.read_csv(DATASET)
    keys = {f"{m}#{t[:50]}" for m, t in zip(df["string"], df["text"])}
    rows = [r[1:] for r in sqlite3.connect(utils.DB).execute(
        "SELECT key, string, text, correct_id, ambiguous, wiki_candidates FROM candidates"
    ) if r[0] in keys]
    print(f"{len(rows)} rows (top-1 accuracy / MRR)\n")
    print(f"{'ranker':36} {'overall':>14} {'ambiguous':>14} {'non-ambiguous':>14}")

    for name, mod in RANKERS:
        t0 = time.time()
        results = []  # (ambiguous, rank of gold or None)
        for i, (mention, context, gold, amb, cands_json) in enumerate(rows, 1):
            ranked = mod.rank(mention, context, json.loads(cands_json))["ranked"]
            gold_rank = next((j + 1 for j, c in enumerate(ranked) if c["qid"] == gold), None)
            results.append((amb, gold_rank))
            if i % 20 == 0:
                print(f"  [{name}] {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        overall = [r for _, r in results]
        amb_t = [r for a, r in results if a == "True"]
        amb_f = [r for a, r in results if a == "False"]
        print(f"{f'{name} [{utils.EMBED_MODEL}]':36} {cell(overall):>14} {cell(amb_t):>14}"
              f" {cell(amb_f):>14}   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

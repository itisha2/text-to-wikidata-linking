# runs the candidate generator over the sampled dataset and stores everything
# in sqlite. same columns as the input csv, plus:
#   wiki_candidates            full candidate list (qid, label, description,
#                              aliases, triples, triple_text) as json
#   correct_qid_in_candidate   1 if the gold QID was retrieved, else 0
# table `candidates`, key = "<mention>#<context[:50]>" (primary key).
# commits row by row so the run can be resumed, already stored keys get skipped.
# prints candidate retrieval recall at the end, overall and per stratum.
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from candidate_generator import candidate_generator

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/dataset_3_sen_level/dataset_3_sampled_100_v2.csv"
DB = ROOT / "candidate_store.db"


def make_key(mention, context):
    return f"{mention}#{context[:50]}"


def ensure_table(con, df, table="candidates"):
    cols = ", ".join(f'"{c}" TEXT' for c in df.columns)
    con.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
        key TEXT PRIMARY KEY,
        {cols},
        wiki_candidates TEXT,
        correct_qid_in_candidate INTEGER)""")
    con.commit()


def candidates_as_json(result):
    """candidate_generator output -> candidate list we can dump as json."""
    candidates = []
    for qid, bundle in result.items():
        if not bundle:
            continue
        (triple_text, _), item = bundle
        candidates.append({
            "qid":         qid,
            "label":       item["label"],
            "description": item["description"],
            "aliases":     item["aliases"],
            "instance_of": [list(t) for t in item.get("instance_of", [])],
            "triples":     [[list(part) for part in t] for t in item["triples"]],
            # [[subj_label, qid], [pred_label, pid], [obj_label, obj_qid|None]]
            "triple_text": triple_text,
        })
    return candidates


def report_recall(con, keys, table="candidates"):
    # only count the current dataset's keys, the table also keeps rows from
    # earlier samples
    ph = ",".join("?" * len(keys))
    n, hits = con.execute(
        f"""SELECT COUNT(*), COALESCE(SUM(correct_qid_in_candidate), 0)
           FROM {table} WHERE key IN ({ph})""", keys).fetchone()
    if n:
        print(f"\nCandidate retrieval recall = {hits}/{n} = {hits/n:.1%}")
    for amb, n_s, hits_s in con.execute(
        f"""SELECT ambiguous, COUNT(*), SUM(correct_qid_in_candidate)
           FROM {table} WHERE key IN ({ph}) GROUP BY ambiguous""", keys):
        print(f"  ambiguous={amb}: {hits_s}/{n_s} = {hits_s/n_s:.1%}")


def build(limit=None):
    df = pd.read_csv(DATASET)
    keys = [make_key(m, t) for m, t in zip(df["string"], df["text"])]
    con = sqlite3.connect(DB)
    ensure_table(con, df)
    have = {k for (k,) in con.execute("SELECT key FROM candidates")}
    mention_cache = {}   # same mention -> same candidates, fetch only once
    stored = sum(k in have for k in keys)
    print(f"{len(df)} rows, {stored} already stored")

    done = 0
    for _, row in df.iterrows():
        mention, context = row["string"], row["text"]
        key = make_key(mention, context)
        if key in have:
            continue

        t0 = time.time()
        if mention in mention_cache:
            candidates = mention_cache[mention]
        else:
            result = candidate_generator(mention, context)
            candidates = candidates_as_json(result)
            mention_cache[mention] = candidates

        hit = int(any(c["qid"] == row["correct_id"] for c in candidates))
        values = ([key]
                  + [str(row[c]) for c in df.columns]
                  + [json.dumps(candidates, ensure_ascii=False), hit])
        placeholders = ",".join("?" * len(values))
        con.execute(f"INSERT OR REPLACE INTO candidates VALUES ({placeholders})", values)
        con.commit()
        have.add(key)
        done += 1
        stored += 1
        print(f"[{stored}/{len(df)}] {key!r}: "
              f"{len(candidates)} candidates, hit={hit}, {time.time()-t0:.1f}s",
              flush=True)
        if limit is not None and done >= limit:
            break

    report_recall(con, keys)
    con.close()


if __name__ == "__main__":
    build()

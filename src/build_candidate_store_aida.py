# builds the AIDA candidate store from el_final_dataset.csv.
# context-aware search this time (use_context=True -> cirrus full-text over
# mention form + sentence lemmas). reuses each row's stored expanded_mention
# so the preprocessing doesn't run again. ranking context = the mention's
# parent sentence, doc_id is kept so document level candidate graphs can be
# built later. same schema/recall reporting as build_candidate_store, same DB
# but its own table (candidates_ctx).
import ast
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from build_candidate_store import (candidates_as_json, ensure_table, make_key,
                                   report_recall)
from candidate_generator import candidate_generator

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/dataset_2_more_entity_types/el_final_dataset.csv"
DB = ROOT / "candidate_store_aida.db"
TABLE = "candidates_ctx"


def build(limit=None):
    df = pd.read_csv(DATASET)
    # rename to the column names the rankers expect: string = mention,
    # text = context sentence, correct_id = gold QID
    df = df.rename(columns={"text": "document_text"})
    df = df.rename(columns={"mention": "string", "parent_sentence": "text",
                            "entity_id": "correct_id"})
    keys = [make_key(m, t) for m, t in zip(df["string"], df["text"])]
    con = sqlite3.connect(DB)
    ensure_table(con, df, table=TABLE)
    have = {k for (k,) in con.execute(f"SELECT key FROM {TABLE}")}
    stored = sum(k in have for k in keys)
    print(f"{len(df)} rows, {stored} already stored")

    done = 0
    for _, row in df.iterrows():
        mention, context = row["string"], row["text"]
        key = make_key(mention, context)
        if key in have:
            continue

        t0 = time.time()
        preprocessed = {"expanded_mention": ast.literal_eval(row["expanded_mention"])}
        result = candidate_generator(mention, context, preprocessed=preprocessed, use_context=True)
        candidates = candidates_as_json(result)

        hit = int(any(c["qid"] == row["correct_id"] for c in candidates))
        values = ([key]
                  + [str(row[c]) for c in df.columns]
                  + [json.dumps(candidates, ensure_ascii=False), hit])
        placeholders = ",".join("?" * len(values))
        con.execute(f"INSERT OR REPLACE INTO {TABLE} VALUES ({placeholders})", values)
        con.commit()
        have.add(key)
        done += 1
        stored += 1
        print(f"[{stored}/{len(df)}] {key!r}: "
              f"{len(candidates)} candidates, hit={hit}, {time.time()-t0:.1f}s",
              flush=True)
        if limit is not None and done >= limit:
            break

    report_recall(con, keys, table=TABLE)
    con.close()


if __name__ == "__main__":
    build()

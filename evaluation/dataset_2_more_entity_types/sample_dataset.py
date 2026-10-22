# sample the stratified AIDA dataset down to the documents with the most
# mentions (~100 mentions total), keeping all mentions of each chosen document.
# ties broken by whichever doc keeps the running ambiguous/non-ambiguous
# split closest to even.
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
N_DOCS = 22

df = pd.read_csv(HERE / "el_stratified_dataset.csv")

# sequential id per unique document + mention count per document
if "doc_id" not in df:
    df["doc_id"] = pd.factorize(df["document_id"])[0] + 1
    df["mentions_in_doc"] = df.groupby("document_id")["mention"].transform("size")

per = (df.groupby("doc_id").agg(mentions=("mention", "size"), amb=("ambiguous", "sum"))
         .assign(non=lambda d: d.mentions - d.amb))

chosen, a, n = [], 0, 0
pool = per.copy()
for _ in range(N_DOCS):
    top = pool[pool.mentions == pool.mentions.max()]
    pick = ((a + top.amb) - (n + top.non)).abs().idxmin()
    chosen.append(pick)
    a += per.loc[pick, "amb"]
    n += per.loc[pick, "non"]
    pool = pool.drop(pick)

out = df[df.doc_id.isin(chosen)]
out.to_csv(HERE / "el_sampled_docs.csv", index=False)
print(f"{out.doc_id.nunique()} documents, {len(out)} mentions "
      f"({out.ambiguous.sum()} ambiguous / {(~out.ambiguous).sum()} non-ambiguous)")
print("mentions per doc:", out.groupby("doc_id").size().value_counts().sort_index().to_dict())

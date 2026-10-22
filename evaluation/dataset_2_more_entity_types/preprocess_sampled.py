# preprocess the sampled AIDA documents: parse each document once, then add
# the text processing features per mention row, including the sentence split
# and the parent sentence the mention belongs to (found by its char span).
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from text_preprocess import nlp_sm, text_preprocess

df = pd.read_csv(HERE / "el_sampled_docs.csv")

docs = {text: nlp_sm(text) for text in df.text.unique()}  # parse each doc once
feats = [text_preprocess(docs[r.text], mention=r.mention, span=(r.start, r.end))
         for r in df.itertuples()]
feats = pd.DataFrame(feats).drop(columns=["text"])  # text is already a column

out = pd.concat([df.reset_index(drop=True), feats], axis=1)

# stratify by ambiguity x NER coverage
out["stratum"] = (out.ambiguous.map({True: "ambiguous", False: "non_ambiguous"})
                  + "_" + out.ner_extracted.map({True: "ner", False: "no_ner"}))

out.to_csv(HERE / "el_sampled_docs_preprocessed_ner_stra.csv", index=False)
print(f"wrote el_sampled_docs_preprocessed_ner_stra.csv: {len(out)} rows, "
      f"{out.parent_sentence.notna().sum()} with a parent sentence")
print("strata:", out.stratum.value_counts().to_dict())

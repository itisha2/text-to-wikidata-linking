# Architecture and implementation

The pipeline links mentions in a sentence to Wikidata QIDs in four stages: preprocess the
text, retrieve candidates from the Wikidata API, enrich each candidate with its 1-hop
graph neighbourhood, and rank the candidates against the sentence context. Everything is
lightweight and local: no Wikidata dump, a SQLite cache for candidates, and small
open models served through Ollama for embeddings and LLM ranking.

```
sentence + mention
      │
      ▼
1. text processing & mention expansion        (spaCy, NLTK)
      │  {surface, lemma, stem, synonyms}
      ▼
2. candidate retrieval                        (Wikidata API, parallel)
      │  deduplicated candidates: QID, label, description, aliases
      ▼
3. knowledge graph enrichment                 (wbgetentities claims → triples)
      │  candidates + verbalized 1-hop graph
      ▼
4. GraphRAG-based ranking                     (embeddings / LLM, Ollama)
      │
      ▼
5. structured JSON output + evaluation
```

## 1. Text processing and term extraction (`src/text_preprocess.py`)

Each sentence runs through spaCy (`en_core_web_sm`). The preprocessor keeps the original
text and extracts named entities with labels, noun phrases, verb phrases, POS tags,
lemmas, stems, and stop-word-filtered context lemmas (used later for context-aware
retrieval). For a given mention it records whether NER covered it and with which type.

The key step for recall is **mention expansion**: each mention becomes a set of query
forms — the normalized surface form, its lemma, its Porter stem, and WordNet synonyms —
each tagged with its type and a per-form search limit (7 for surface/lemma/stem, 1 for
synonyms). Querying all forms instead of just the surface string recovers concepts whose
Wikidata label differs morphologically or lexically from the mention.

## 2. Candidate retrieval (`src/candidate_generator.py`)

Every expanded form is queried against the Wikidata API in parallel (multiprocessing
pool), with retry/backoff handling for timeouts, 5xx errors and rate limits. Two modes:

- **Mention-only** (default): `wbsearchentities`, Wikidata's own prefix/alias search.
- **Context-aware**: Cirrus full-text search (`list=search`) with `inlabel:"form"` plus
  the sentence's content lemmas ORed, so items whose descriptions and statements match
  the context rank higher. Because this endpoint returns QIDs without labels, hits are
  verified after enrichment and dropped if the form is not actually in the label/aliases.

Results from all forms are deduplicated by QID. Each candidate carries retrieval
features: which form and form-type found it, Wikidata's own rank, whether it matched by
label or alias, Levenshtein distance to the mention, and an exact-match flag. This stage
deliberately optimizes recall — finding the right concept among the candidates — and
leaves selection to the ranker.

## 3. Knowledge graph enrichment (`candidate_generator.py`)

One batched `wbgetentities` call per candidate set fetches labels, descriptions, aliases
*and* claims together. Claims are converted to 1-hop triples
`(subject, predicate, object)`, where each element keeps its English label, its QID/PID,
and the term Wikidata's own RDF export uses (`wd:`/`wdt:` IRIs; `rdfs:label`,
`schema:description`, `skos:altLabel` for the lexical layer). Identifier/media datatypes
and housekeeping properties (Commons category, "described by source", …) are skipped
because they carry no semantic signal. The resulting local graph connects each candidate
to its types, country, occupation, part-of relations, etc. — exactly the context needed
to tell apart candidates with the same label.

Because candidate retrieval is the slow, API-bound part, `src/build_candidate_store.py`
runs it once over the evaluation set and stores the enriched candidates as JSON in
SQLite (`candidate_store.db`, resumable, one table per retrieval mode). All ranking
experiments then run offline against this store.

## 4. GraphRAG-based candidate selection (`src/ranker/`)

The GraphRAG idea is implemented by **verbalizing each candidate's 1-hop graph** into its
evidence text, so the ranker scores the mention context against the graph neighbourhood,
not just the description. Seven ranker variants form a controlled ablation:

| ranker | context side | candidate side | scorer |
|---|---|---|---|
| v1 (baseline) | sentence | label + description + aliases | embedding cosine |
| v2 | sentence | v1 + graph triples | embedding cosine |
| v3 | sentence | label + description + aliases | LLM ranking |
| v4 | sentence | v3 + graph triples | LLM ranking |
| v5 | AMR graph of sentence | v2 text | embedding cosine |
| v6 | AMR graph of sentence | v2 text | LLM ranking |
| v7 | sentence | v1 text + structural node embeddings | embedding cosine + graph-embedding coherence |

v5/v6 parse the sentence into an AMR graph (amrlib) and render it as "concept relation
concept" facts, making the comparison graph-to-graph. v7 replaces the verbalized graph
context of v2/v4 with structural node embeddings of the candidate graph (NetMF-style:
degree-normalized walk matrix, log-dampened, SVD-factorized), scoring each candidate by
cosine to the centroid of the other mentions' candidate vectors — graph coherence without
the hub-degree bias of the verbalized bonus. Models run locally via Ollama:
`embeddinggemma` / `nomic-embed-text` for embeddings, `lfm2.5-thinking` / `deepseek-r1`
for LLM ranking.

**Ranking strategy.** The semantic score (cosine similarity, or 1/position from the
LLM's ranking) is combined with lexical priors:

```
score(c) = sem(c) + 0.15 / wikidata_rank + 0.1 · exact_label_match
           − 0.05 · levenshtein(mention, label) / len(mention)
```

The top-scored candidate is selected; confidence is a softmax over the candidate scores.

## 5. Output and evaluation

Each decision is emitted as structured JSON (`linking_results_*.json`): mention,
sentence, selected concept with QID, rank, score and confidence, plus all alternatives,
each with its evidence text.

The evaluation set is 100 sentence-level mentions stratified into 50 ambiguous
(mentions with multiple plausible QIDs) and 50 non-ambiguous, sampled with a fixed seed.
Two things are measured:

- **Candidate retrieval recall** — is the gold QID among the candidates?
  Mention-only retrieval reaches **91%** overall (100% non-ambiguous, 82% ambiguous).
  The context-aware Cirrus variant reached only 52%, so the mention-only candidates are
  used for ranking: context helps selection, but restricting *retrieval* by context
  hurts recall.
- **Linking accuracy** — top-1 accuracy and MRR per ranker and model
  (`src/ranker/compare.py`), reported overall and split by ambiguity, so the baseline
  (v1/v3, text only) can be compared directly against the graph-informed variants
  (v2/v4/v5/v6) to quantify whether graph context improves selection.

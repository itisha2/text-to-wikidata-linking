# Architecture and implementation

The pipeline links mentions in a sentence to Wikidata QIDs in four stages: preprocess the
text, retrieve candidates from the Wikidata API, enrich each candidate with its 1-hop
graph neighbourhood, and rank the candidates against the sentence context. Everything is
lightweight and local: no Wikidata dump, a SQLite cache for candidates, and small
open models served through Ollama for embeddings and decision-model ranking.

```
sentence + mention
      │
      ▼
1. text processing & mention expansion        (spaCy + d1-omni NER, NLTK)
      │  {surface, lemma, stem, synonyms}
      ▼
2. candidate retrieval                        (Wikidata API, parallel)
      │  deduplicated candidates: QID, label, description, aliases
      ▼
3. knowledge graph enrichment                 (wbgetentities claims → triples,
      │  candidates + verbalized 1-hop graph   per-document NetworkX graph)
      ▼
4. GraphRAG-based ranking                     (embeddings / d1 decision model, Ollama)
      │
      ▼
5. structured JSON output + evaluation
```

## 1. Text processing and term extraction (`src/text_preprocess.py`)

Each sentence runs through spaCy (`en_core_web_sm`). The preprocessor keeps the original
text and extracts named entities with labels, noun phrases, verb phrases, POS tags,
lemmas, and stems. For a given mention it records whether NER covered it and with which
type.

spaCy's small model alone misses non-named-entity mentions, so NER is a two-pass merge:
spaCy first, then a document-level pass with the **d1-omni:600m decision model** (served
via Ollama) that types the spans spaCy missed, answering typed
NER questions with a calibrated probability per option in one forward pass. The merge
raised mention coverage from 61/94 ≈ 65% (spaCy alone) to **90/100 = 90%**.

The key step for recall is **mention expansion**: each mention becomes a set of query
forms — the normalized surface form, its lemma, its Porter stem, and WordNet synonyms —
each tagged with its type and a per-form search limit (7 for surface/lemma/stem, 1 for
synonyms). Querying all forms instead of just the surface string recovers concepts whose
Wikidata label differs morphologically or lexically from the mention.

## 2. Candidate retrieval (`src/candidate_generator.py`)

Every expanded form is queried against the Wikidata API in parallel (multiprocessing
pool), with retry/backoff handling for timeouts, 5xx errors and rate limits. Two modes:

- **Mention-only** (default): `wbsearchentities`, Wikidata's own prefix/alias search.
- **Context-aware**: Cirrus full-text search (`list=search`) with the sentence, so items
  whose descriptions and statements match the context rank higher. Because this endpoint returns QIDs without labels, hits are
  verified after enrichment and dropped if the form is not actually in the label/aliases.

Results from all forms are deduplicated by QID. Each candidate carries retrieval
features: which form and form-type found it, Wikidata's own rank, whether it matched by
label or alias, Levenshtein distance to the mention, and an exact-match flag. This stage
deliberately optimizes recall — finding the right concept among the candidates — and
leaves selection to the ranker.

**Retrieval results** (gold-QID recall on the stratified evaluation set):

| retrieval mode | overall recall | ambiguous | non-ambiguous |
|---|---|---|---|
| `wbsearchentities` | **68/90 = 75.6%** | 27/44 = 61.4% | 41/46 = 89.1% |
| CirrusSearch (full-text) | 35/90 = 38.9% | 19/44 = 43.2% | 16/46 = 34.8% |

`wbsearchentities` wins decisively: Cirrus surfaces less prominent entities that merely
contain the query words, while `wbsearchentities` ranks by prominence, which matches how
mentions are used in text — so the mention-only candidates are used for ranking. The gap
is widest on non-ambiguous mentions (89.1% vs 34.8%): adding the sentence to the query
introduces ambiguity for mentions that a mention-only lookup retrieves easily. Candidate
recall is the ceiling on end-to-end accuracy, and the misses concentrate on ambiguous
mentions (61.4% vs 89.1%): on those, the generator, not the ranker, is the bottleneck.

## 3. Knowledge graph enrichment (`candidate_generator.py`)

One batched `wbgetentities` call per candidate set fetches labels, descriptions, aliases
*and* claims together. Claims are converted to 1-hop triples
`(subject, predicate, object)`, where each element keeps its English label, its QID/PID,
and the term Wikidata's own RDF export uses (`wd:`/`wdt:` IRIs; `rdfs:label`,
`schema:description`, `skos:altLabel` for the lexical layer). Identifier/media datatypes
and housekeeping properties (Commons category, "described by source", …) are skipped
because they carry no semantic signal.

Per document, a NetworkX graph connects all candidates of all mentions: nodes are QIDs,
edges are Wikidata properties. This graph is the disambiguation signal: correct
candidates pull together into a connected cluster — e.g. Belarus, Germany and Ireland
link via "diplomatic relation" — while wrong candidates with the same surface forms
(the element germanium, the person Samuel Ireland) stay isolated.

Because candidate retrieval is the slow, API-bound part, `src/build_candidate_store_aida.py`
runs it once over the evaluation set and stores the enriched candidates as JSON in
SQLite (`candidate_store_aida.db`, resumable, one table per retrieval mode). All ranking
experiments then run offline against this store.

## 4. GraphRAG-based candidate selection (`src/ranker/`)

A RAG pipeline usually retrieves from a dedicated vector or graph database; here Wikidata
itself is the graph database, and retrieval from it happens at the candidate-generation
phase (stages 2–3). GraphRAG enters the ranker in two ways:

- **Retrieval-augmentation**: each candidate's 1-hop graph neighbourhood is verbalized
  into up to 12 facts and appended to the candidate text before scoring.
- **Graph coherence bonus** (weight 0.2): from the per-document candidate graph, a
  candidate directly connected to another mention's candidate scores 1.0, one sharing a
  neighbour 0.5, an isolated candidate 0.

**Base scorer (all variants).** The semantic score (embedding cosine between
"mention + sentence" and each candidate rendered as "label: description (aliases)", or
the d1 decision model's calibrated probability) is combined with lexical priors:

```
score(c) = sem(c) + 0.15 / wikidata_rank + 0.1 · exact_label_match
           − 0.05 · levenshtein(mention, label) / len(mention)
```

The top-scored candidate is selected; confidence is a softmax over the candidate scores.
Models run locally via Ollama: `embeddinggemma` for embeddings, `d1-omni:600m` as the
decision model (ranking phrased as one choice question, candidates as options, a
calibrated probability per candidate in a single forward pass).

Seven ranker variants (`cr_v1`–`cr_v7`) form a controlled ablation:

| ranker | context side | candidate side | scorer |
|---|---|---|---|
| v1 (baseline) | sentence | label + description + aliases | embedding cosine |
| v2 | sentence | v1 + graph facts + coherence bonus | embedding cosine |
| v3 | sentence | label + description + aliases | d1 choice question |
| v4 | sentence | v3 + graph facts + coherence bonus | d1 choice question |
| v5 | AMR graph of sentence (amrlib) | v2 text | embedding cosine |
| v6 | AMR graph of sentence | v2 text | d1 choice question |
| v7 | sentence | v1 text + structural node embeddings | embedding cosine + graph-embedding coherence |

v5/v6 parse the sentence into an AMR graph (amrlib) and render it as "concept relation
concept" facts, making the comparison graph-to-graph. v7 replaces the verbalized graph
context with structural node embeddings of the candidate graph (NetMF-style:
degree-normalized walk matrix, log-dampened, SVD-factorized), scoring each candidate by
cosine to the centroid of the other mentions' candidate vectors.

**Results** (full stratified run, top-1 accuracy / MRR):

| ranker | overall | ambiguous | non-ambiguous | time |
|---|---|---|---|---|
| v1 embed (embeddinggemma) | 57.8% / 0.64 | 40.9% / 0.48 | 73.9% / 0.79 | 14 s |
| v2 embed + graph | **63.3% / 0.67** | **47.7% / 0.52** | **78.3% / 0.82** | 19 s |
| v3 d1 (d1-omni:600m) | 62.2% / 0.67 | 45.5% / 0.52 | **78.3% / 0.82** | 15 s |
| v4 d1 + graph | 60.0% / 0.66 | 45.5% / 0.52 | 73.9% / 0.79 | 14 s |
| v5 AMR graph (amrlib) | 60.0% / 0.64 | 47.7% / 0.51 | 71.7% / 0.76 | 858 s |
| v6 AMR + d1 | 56.7% / 0.63 | 38.6% / 0.47 | 73.9% / 0.79 | 28 s |
| v7 graph-embed (NetMF-style) | 57.8% / 0.64 | 38.6% / 0.47 | 76.1% / 0.80 | 19 s |

**Decisions from these experiments:**

- **Embedding vs decision-model ranking.** The d1 rankers land just behind the embeddings
  (62.2% / 60.0% vs 63.3%), but graph context only helps the embedding family (v1→v2,
  +5.5 points overall); for d1 it slightly hurts (v3→v4, −2.2) — the verbalized facts
  lengthen the options without adding discriminative evidence for the decision model.
- **Verbalized vs structural graph embeddings.** v7 matched the no-graph baseline overall
  (57.8%) but lost to verbalized graph context on ambiguous mentions (38.6% vs 47.7%).
- **Negative results — AMR graphs (v5/v6).** v5 ties v2 on ambiguous mentions (47.7%) but
  trails overall (60.0% vs 63.3%) and takes 858 s — ~45× slower — ruling it out. v6 drops
  to 56.7% overall, joint-worst on ambiguous mentions (38.6%, with v7): stacking
  verbalized graphs on both sides dilutes the evidence instead of sharpening it.
### ⭐ Final ranker: v2 — embeddings + verbalized graph facts + coherence bonus

> Graph context lifts overall accuracy **57.8% → 63.3%** and ambiguous accuracy
> **40.9% → 47.7%** over the no-graph baseline — the gain lands exactly where
> disambiguation is hardest, but only when paired with the embedding ranker. How the
> graph is used matters more than the ranking model: verbalized facts beat both
> structural embeddings (v7) and AMR graphs (v5/v6). MRR above accuracy (0.67 vs 63.3%)
> means that when rank 1 is wrong the gold entity is usually just below it, so the
> surfaced alternatives often contain the right answer.

## 5. Output and evaluation

Each decision is emitted as structured JSON (`linking_results_*.json`): mention,
sentence, selected concept with QID, rank, score and confidence, plus all alternatives,
each with its evidence text.

The evaluation set is built from ELEVANT (EMNLP 2022), chosen because it annotates
entities beyond named entities. Gold QIDs were re-validated against current Wikidata,
and the set was stratified into ambiguous vs non-ambiguous mentions, crossed with the
NER outcome of stage 1 (ambiguous/non-ambiguous × ner/no-ner). Documents were sampled by
taking those with the most mentions, keeping all mentions of a chosen document and
breaking ties to keep the ambiguity split closest to even. Two things are measured:

- **Candidate retrieval recall** — is the gold QID among the candidates? (table in
  stage 2: 75.6% mention-only vs 38.9% context-aware Cirrus — context helps selection,
  but restricting *retrieval* by context hurts recall).
- **Linking accuracy** — top-1 accuracy and MRR per ranker (`src/ranker/compare.py`),
  reported overall and split by ambiguity (table in stage 4), so the no-graph baselines
  (v1/v3) are compared directly against the graph-informed variants to quantify whether
  graph context improves selection. It does: +5.5 points overall, +6.8 on ambiguous
  mentions (v1→v2).

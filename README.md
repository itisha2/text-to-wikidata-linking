# Text-to-Wikidata Entity Linking

A pipeline that takes a plain-text paragraph, finds the meaningful terms in it, and links
each one to the right Wikidata QID. Candidates come from the live Wikidata API, get
enriched with their 1-hop knowledge-graph neighbourhood, and a GraphRAG-style ranker
picks the winner. No Wikidata dump, no training — just a SQLite cache and small open
models served through Ollama.

**Example** — input:

> Marie Curie conducted research on radioactivity in Paris and won the Nobel Prize in Physics.

Output: *Marie Curie → Q7186, Paris → Q90, radioactivity → Q11448, ...* with a
confidence per mention and a JSON file of alternatives and evidence.

## Where to look

| What you want to see | Where it is |
|---|---|
| The pipeline end to end, on a real paragraph | [`inference_pipeline.ipynb`](inference_pipeline.ipynb) |
| Design decisions and the full rationale | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Term extraction (spaCy NER + d1 pass, mention expansion) | [`src/text_preprocess.py`](src/text_preprocess.py) |
| Candidate retrieval + 1-hop graph enrichment | [`src/candidate_generator.py`](src/candidate_generator.py) |
| The seven ranking experiments (v2 is the final one) | [`src/ranker/`](src/ranker/) |
| How the evaluation dataset was built | [`evaluation/`](evaluation/) |

## How it works, briefly

1. **Term extraction** — spaCy NER plus a small decision-model pass for the spans spaCy
   misses; each mention is expanded into lemma/stem/synonym query forms.
2. **Candidate retrieval** — every form is queried against the Wikidata API in parallel;
   this stage goes for recall, selection happens later.
3. **Graph enrichment** — each candidate's Wikidata claims become 1-hop triples, and all
   candidates together form one document-level graph.
4. **Ranking** — the final ranker (v2) embeds each candidate's verbalized graph
   neighbourhood against the mention + sentence, and adds a coherence bonus for
   candidates connected to the other mentions' candidates.

The ranker folder holds seven variants (`cr_v1.py` … `cr_v7.py`) — a controlled ablation
over context, graph evidence and scorer. v2 (embedding + graph) beats the text-only
baseline by ~5pp top-1, with the gain concentrated on ambiguous mentions. Full tables in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Running it

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

Ollama must be running locally with `embeddinggemma` and `d1-omni:600m` pulled.

**Inference:** open [`inference_pipeline.ipynb`](inference_pipeline.ipynb) and run it
top to bottom. Takes a few minutes, mostly Wikidata rate-limit waits.

**Evaluation:** retrieval is slow, so it runs once into a store, then rankers run
offline against it:

```bash
uv run python src/build_candidate_store_aida.py   # -> candidate_store_aida.db
cd src/ranker
uv run python cr_v2.py        # one ranker (prints accuracy/MRR, writes a results JSON)
uv run python compare.py      # all rankers x all models
```

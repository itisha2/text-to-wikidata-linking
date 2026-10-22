# Shared helpers for the rankers: Ollama embedding / LLM calls and evaluation.
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import Levenshtein
import networkx as nx
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
# AIDA store by default, set CANDIDATE_DB=candidate_store.db for the dataset_3
# store. gold QID column is correct_id (= the csv's entity_id, renamed in
# build_candidate_store_aida.py; string = mention, text = parent sentence).
DB = ROOT / os.environ.get("CANDIDATE_DB", "candidate_store_aida.db")
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma" # small model# 621MB

D1_MODEL = "d1-omni:600m" # small decision model


def ollama(endpoint, payload):
    """post to ollama with retries (models loading/unloading, server restarts)."""
    for wait in (5, 30, 120, 300):
        try:
            r = requests.post(f"{OLLAMA}/{endpoint}", json=payload)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(wait)
    r = requests.post(f"{OLLAMA}/{endpoint}", json=payload)
    r.raise_for_status()
    return r.json()


def embed(texts, model=None):
    """embed a list of texts, unit-normalized."""
    v = np.array(ollama("api/embed", {"model": model or EMBED_MODEL,
                                      "input": texts})["embeddings"])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def llm(prompt, model):
    """single LLM completion."""
    return ollama("api/generate", {"model": model,
                                   "prompt": prompt, "stream": False})["response"]


def cosine(M, v):
    """cosine of each row of M with v (normalizes both sides, so it also works
    on embeddings that aren't unit length)."""
    return (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v))


W_RANK, W_EXACT, W_LEV = 0.15, 0.1, 0.05
W_GRAPH = 0.2  # weight of the graph coherence score (v2/v4/v6)


def score_base(mention, candidates, sems):
    """score = sem + w_rank/rank + w_exact*exact_match - w_lev*lev/len(mention).
    rank = position in the stored candidate list (wikidata's own search order)."""
    m = (mention or "").lower().strip()
    scores = []
    for rank, (c, sem) in enumerate(zip(candidates, sems), start=1):
        label = (c["label"] or "").lower()
        s = float(sem) + W_RANK / rank
        if m:
            s += W_EXACT * (label == m)
            s -= W_LEV * Levenshtein.distance(m, label) / max(len(m), 1)
        scores.append(s)
    return np.array(scores)


def embed_rank(query_text, candidates, candidate_text, mention=None, extra=None):
    """rank candidates by cosine to query_text + the lexical priors from
    score_base, plus an optional extra score per candidate (the graph bonus).
    returns the best QID with its evidence and the full ranked list."""
    if not candidates:
        return {"qid": None, "evidence": None, "ranked": []}
    texts = [candidate_text(c) for c in candidates]
    sems = cosine(embed(texts), embed([query_text])[0])
    scores = score_base(mention, candidates, sems)
    if extra is not None:
        scores = scores + extra
    ranked = [{"qid": candidates[i]["qid"], "label": candidates[i]["label"],
               "score": float(scores[i]), "evidence": texts[i]}
              for i in np.argsort(-scores)]
    return {"qid": ranked[0]["qid"], "evidence": ranked[0]["evidence"], "ranked": ranked}






FACTS = 12  # graph facts per candidate
_graphs = None


def rdf_term(part):
    """rdf term for a triple part: the stored one if present, else built from
    the id (wd: for entities, wdt: for properties)."""
    if len(part) > 2 and part[2]:
        return part[2]
    i = part[1]
    if not i:
        return None
    return f"wdt:{i}" if str(i).startswith("P") else f"wd:{i}"


def graphs_from_rows(rows, doc_level):
    """one networkx graph per group, built from (mention, text, group,
    candidates) rows. nodes are QIDs, edges wikidata properties, both keep
    their rdf mapping (wd:/wdt:) in the 'rdf' attribute. returns
    ({group: (graph, {mention: qids})}, {text: group}, doc_level) so a ranking
    context (text column) can find its group's graph. also used on the fly at
    inference: seed _graphs with the result and the rankers pick it up."""
    graphs, by_text = {}, {}
    for mention, text, group, cands in rows:
        G, mentions = graphs.setdefault(group, (nx.Graph(), {}))
        by_text[text] = group
        for c in cands:
            mentions.setdefault(mention, set()).add(c["qid"])
            for s, p, o in c["triples"]:
                if o[1]:  # entity object -> edge between QIDs
                    G.add_node(s[1], label=s[0], rdf=rdf_term(s))
                    G.add_node(o[1], label=o[0], rdf=rdf_term(o))
                    G.add_edge(s[1], o[1], label=p[0], rdf=rdf_term(p))
    return graphs, by_text, doc_level


def candidate_graphs():
    """graphs_from_rows over the whole candidate store. grouped per document
    when the store has doc_id (AIDA), else per sentence."""
    con = sqlite3.connect(DB)
    cols = [c[1] for c in con.execute("PRAGMA table_info(candidates)")]
    group_col = "doc_id" if "doc_id" in cols else "text"
    rows = con.execute(f'SELECT string, text, "{group_col}", wiki_candidates'
                       " FROM candidates").fetchall()
    con.close()
    return graphs_from_rows([(m, t, g, json.loads(j)) for m, t, g, j in rows],
                            group_col == "doc_id")


def candidate_graph(mention, context, doc_only=False):
    """graph + the other mentions' candidate qids for one group (cached).
    doc_only=True gives an empty graph on sentence-grouped stores, so the
    caller only ever gets document level graph context."""
    global _graphs
    if _graphs is None:
        _graphs = candidate_graphs()
    graphs, by_text, doc_level = _graphs
    if doc_only and not doc_level:
        return nx.Graph(), []

    # one G per context or document
    G, mentions = graphs.get(by_text.get(context), (nx.Graph(), {}))
    others = [qids for m, qids in mentions.items() if m != mention]
    return G, others


def graph_facts(G, qid, others, sentence):
    """the candidate's graph neighbourhood as text. neighbours named in the
    sentence or linked to other mentions' candidates sort first, so the most
    relevant facts survive the FACTS cut."""
    if qid not in G:
        return ""
    s = sentence.lower()
    other_qids = set().union(*others) if others else set()
    facts = []
    for n in G.neighbors(qid):
        lab = G.nodes[n].get("label") or ""
        rel = G.edges[qid, n].get("label", "related to")
        relevant = ((len(lab) > 2 and lab.lower() in s) or n in other_qids
                    or bool(other_qids & set(G.neighbors(n))))
        facts.append((not relevant, f"{rel} {lab}"))
    facts.sort()
    return "; ".join(f for _, f in facts[:FACTS])


def graph_evidence(G, qid, others, sentence):
    """0..1. connectivity to the other mentions' candidates (direct edge 1,
    shared neighbour 0.5, best per mention), or neighbour labels in the sentence."""
    if qid not in G:
        return 0.0
    nb = set(G.neighbors(qid))
    links = [max((1.0 if q in nb else
                  0.5 if q in G and nb & set(G.neighbors(q)) else 0.0)
                 for q in qids) for qids in others if qids]
    link = sum(links) / len(links) if links else 0.0
    s = sentence.lower()
    labels = ((G.nodes[n].get("label") or "").lower() for n in nb)
    hits = sum(1 for lab in labels if len(lab) > 2 and lab in s)
    return max(link, min(1.0, hits / 3))


def graph_bonus(mention, context, candidates, doc_only=False):
    """graph coherence score per candidate (graph_evidence), scaled by W_GRAPH."""
    G, others = candidate_graph(mention, context, doc_only)
    return W_GRAPH * np.array(
        [graph_evidence(G, c["qid"], others, context) for c in candidates])


_printed_graphs = set()


def print_graph(G, max_lines=40):
    """print a candidate graph as an ascii tree (networkx write_network_text,
    node labels instead of QIDs)."""
    print(G)  # Graph with N nodes and M edges
    buf = io.StringIO()
    nx.write_network_text(G, path=buf, with_labels="label", max_depth=2)
    lines = buf.getvalue().splitlines()
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"... {len(lines) - max_lines} more lines")


def graph_candidate_text(mention, context, base_text, doc_only=False):
    """candidate_text that appends the graph facts to base_text(c). base_text
    comes back unchanged when there is no graph (doc_only on a sentence-grouped
    store)."""
    G, others = candidate_graph(mention, context, doc_only)
    if not len(G):
        return base_text
    """
    if os.environ.get("GRAPH_DEBUG") and id(G) not in _printed_graphs:
        _printed_graphs.add(id(G))
        print(f"\ncandidate graph for {mention!r} "
              f"(co-mention candidates: {[len(q) for q in others]}):")
        print_graph(G)
        """
    return lambda c: f"{base_text(c)}. {graph_facts(G, c['qid'], others, context)}"


D1_QUESTION = "Which Wikidata candidate does the mention refer to in the sentence?"


def d1_rank(mention, context, candidates, candidate_text, extra=None):
    """ask the d1 decision model to pick the candidate: one choice question,
    the options are the rendered candidates. d1 scores all options in a single
    forward pass, the probabilities then go through score_base (same lexical
    priors as embed_rank). extra (the graph bonus) is scaled by d1's
    uncertainty, so it breaks near-ties but can't overrule a confident pick."""
    if not candidates:
        return {"qid": None, "evidence": None, "ranked": []}
    answer = ollama("v1/systemone", {
        "model": D1_MODEL,
        "state": {"mention": mention, "sentence": context},
        "questions": {"entity": {
            "type": "choice",
            "instructions": D1_QUESTION,
            "criteria": {c["qid"]: candidate_text(c) for c in candidates},
        }},
    })["answers"]["entity"]
    probs = answer["probabilities"]
    d1_scores = np.array([float(probs.get(c["qid"], 0.0)) for c in candidates])

    # same as in embed_rank; coherence weighted by how unsure d1 is
    scores = score_base(mention, candidates, d1_scores)
    if extra is not None:
        scores = scores + (1 - float(answer.get("confidence", 0.0))) * extra
    ranked = [{"qid": candidates[i]["qid"], "label": candidates[i]["label"],
               "score": float(scores[i]), "evidence": candidate_text(candidates[i])}
              for i in np.argsort(-scores)]
    return {"qid": ranked[0]["qid"], "evidence": ranked[0]["evidence"], "ranked": ranked}


def to_json(mention, context, result):
    """one linking decision as json: selected concept + QID with a confidence
    (softmax over the ranking scores), plus alternatives and evidence."""
    ranked = result["ranked"]
    # embed rankers carry a cosine score; LLM rankers only an order -> use 1/rank
    scores = np.array([c.get("score", 1 / (i + 1)) for i, c in enumerate(ranked)])
    conf = np.exp(scores) / np.exp(scores).sum() if len(ranked) else np.array([])
    cands = [{"concept": c["label"], "qid": c["qid"], "rank": i + 1,
              "score": round(float(scores[i]), 4),
              "confidence": round(float(conf[i]), 4),
              "evidence": c["evidence"]}
             for i, c in enumerate(ranked)]
    return {"mention": mention, "sentence": context,
            "selected": cands[0] if cands else None,
            "alternatives": cands[1:]}


def load_rows():
    """candidate store rows (string, text, correct_id, ambiguous, candidates).
    cli args filter by use case: 'amb' or 'non' keeps only that case, a number
    keeps the first N rows (e.g. cr_v2.py amb 4). stores without a category
    column (dataset_3) derive it from the ambiguous flag."""
    con = sqlite3.connect(DB)
    cols = [c[1] for c in con.execute("PRAGMA table_info(candidates)")]
    cat = ("category" if "category" in cols else
           "CASE ambiguous WHEN 'True' THEN 'ambiguous' ELSE 'non_ambiguous' END")
    rows = con.execute(
        "SELECT string, text, correct_id, ambiguous, wiki_candidates, "
        f"{cat} FROM candidates").fetchall()
    con.close()
    args = sys.argv[1:]
    if "amb" in args:
        rows = [r for r in rows if r[5] == "ambiguous"]
    elif "non" in args:
        rows = [r for r in rows if r[5] == "non_ambiguous"]
    limit = next((int(a) for a in args if a.isdigit()), None)
    rows = [r[:5] for r in rows]
    return rows[:limit] if limit else rows


def evaluate(rank):
    """run a rank function over the rows load_rows() selects, print top-1
    accuracy and MRR, write the json outputs next to the DB."""
    rows = load_rows()

    results = []  # (ambiguous, rank of gold or None)
    outputs = []
    for mention, context, gold, ambiguous, cands_json in rows:
        result = rank(mention, context, json.loads(cands_json))
        ranked = result["ranked"]
        outputs.append(to_json(mention, context, result))
        gold_rank = next((i + 1 for i, c in enumerate(ranked) if c["qid"] == gold), None)
        results.append((ambiguous, gold_rank))
        top = ranked[0] if ranked else None
        print(f"{mention!r}: gold={gold} rank={gold_rank} "
              f"top1={top['qid'] if top else None} ({top['label'] if top else '-'})")

    def report(name, subset):
        n = len(subset)
        acc = sum(r == 1 for r in subset) / n
        mrr = sum(1 / r for r in subset if r) / n
        print(f"  {name}: top-1 = {acc:.1%}, MRR = {mrr:.2f}  (n={n})")

    print("\nRanking results:")
    report("overall", [r for _, r in results])
    for amb in sorted({a for a, _ in results}):
        report(f"ambiguous={amb}", [r for a, r in results if a == amb])

    out = ROOT / f"linking_results_{Path(rank.__code__.co_filename).stem}.json"
    out.write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

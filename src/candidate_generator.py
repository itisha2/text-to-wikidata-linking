# candidate generator for entity linking, queries the wikidata API (cirrus search).
# cirrus search is an extension to the wikidata API that uses elasticssearch internally.
# mention expansion is done in src/text_preprocess.py, here we just take that
# output and query wikidata with it.

import sys
from pathlib import Path

import pandas as pd
import requests
import os
import time
import re

from multiprocessing import Pool, Manager

import hashlib
import Levenshtein
import warnings
warnings.filterwarnings("ignore")
from Crypto.Hash import MD4

# text processing (spaCy pipeline + mention expansion) is shared project-wide
sys.path.append(str(Path(__file__).resolve().parent))
from text_preprocess import (
    nlp_sm as nlp,
    text_preprocess,
    mention_expansion,
)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "entity-linking-assignment/0.1"}

session = requests.Session()
session.headers.update(HEADERS)


def api_get(params, timeout=60, max_retries=5):
    """GET with retries. backoff on timeouts/5xx, for 429 wait for Retry-After."""
    for attempt in range(max_retries):
        try:
            r = session.get(WIKIDATA_API, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
                print(f"rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = 2 ** attempt * 2
            print(f"{type(e).__name__}, retry {attempt + 1}/{max_retries} in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Wikidata API still failing after {max_retries} retries")

# these datatypes don't help ranking (ids, media files, coordinates etc)
SKIP_DATATYPES = {"external-id", "commonsMedia", "url", "globe-coordinate",
                  "math", "musical-notation", "tabular-data", "geo-shape",
                  "wikibase-lexeme", "wikibase-form", "wikibase-sense"}

# wikimedia housekeeping properties, mostly noise
SKIP_PROPERTIES = {
    "P373",   # Commons category
    "P487",   # Unicode character
    "P910",   # topic's main category
    "P1343",  # described by source
    "P1424",  # topic's main template
    "P5008",  # on focus list of Wikimedia project
    "P2959",  # permanent duplicated item
    "P1889",  # different from
    "P460",   # said to be the same as
}


def parse_triples(triples):
    if triples and triples[0][1] == "label":
        label = triples[0][2]
        temp = {"text": label}
        sent2 = label + " "
        aliases = []
        for triple in triples[1:]:
            if "alias" in triple[1]:
                aliases.append(triple[2])
            sent2 += triple[1] + " "+ triple[2] + " "
        if aliases:
            temp["aliases"] = aliases
        sent2 = re.sub(r"\s+", " ", sent2).strip()
        return [sent2, temp]
    else: return False


def fetch_entities(qids):
    """batched wbgetentities: label, description, aliases and claims in one go,
    so we don't hit the same endpoint twice. returns (details, claims_by_qid)."""
    details = {}
    claims_by_qid = {}
    # batches of 20 not 50, with full claims the responses get big and
    # sometimes time out on slow connections
    for i in range(0, len(qids), 20):
        batch = qids[i : i + 20]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|descriptions|aliases|claims",
            "languages": "en",
            "languagefallback": 1,   # resolves labels stored under "mul" too
            "format": "json",
        }
        r = api_get(params)
        for qid, ent in r.get("entities", {}).items():
            details[qid] = {
                "label": ent.get("labels", {}).get("en", {}).get("value"),
                "description": ent.get("descriptions", {}).get("en", {}).get("value"),
                "aliases": [a["value"] for a in ent.get("aliases", {}).get("en", [])],
            }
            claims_by_qid[qid] = ent.get("claims", {})
    return details, claims_by_qid


def get_labels(ids):
    """batched label lookup for entity/property ids (en, falls back to mul)."""
    labels = {}
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
            "languagefallback": 1,
            "format": "json",
        }
        r = api_get(params)
        for k, ent in r.get("entities", {}).items():
            labels[k] = ent.get("labels", {}).get("en", {}).get("value")
    return labels


def snak_value(snak):
    """readable value of a claim snak. entity values come back as ("entity", id)
    so the label can be looked up later."""
    if snak.get("datatype") in SKIP_DATATYPES:
        return None
    dv = snak.get("datavalue", {})
    t, v = dv.get("type"), dv.get("value")
    if t == "wikibase-entityid":
        return ("entity", v.get("id"))
    if t == "string":
        return ("literal", v)
    if t == "monolingualtext":
        return ("literal", v.get("text"))
    if t == "time":
        return ("literal", v.get("time", "").lstrip("+")[:10])
    if t == "quantity":
        return ("literal", v.get("amount", "").lstrip("+"))
    return None


def get_1_hop_triples(candidates_dedup, details, claims_by_qid):
    """1-hop triples per candidate, from the claims fetch_entities() already
    fetched. only the labels of referenced properties/entities need one more call."""
    # collect every property and entity id we still need a label for
    to_resolve = set()
    for claims in claims_by_qid.values():
        for pid, statements in claims.items():
            if pid in SKIP_PROPERTIES:
                continue
            to_resolve.add(pid)
            for st in statements:
                val = snak_value(st.get("mainsnak", {}))
                if val and val[0] == "entity" and val[1]:
                    to_resolve.add(val[1])
    labels = get_labels(sorted(to_resolve))

    out = {}
    for item in candidates_dedup:
        qid = item["QID"]
        d = details.get(qid, {})
        triples = [(qid, "label", d.get("label") or "")]
        if d.get("description"):
            triples.append((qid, "description", d["description"]))
        for n, alias in enumerate(d.get("aliases", []), start=1):
            triples.append((qid, f"alias_{n}", alias))
        # structured triples for the graph: (subject, predicate, object), each
        # part is (label, QID/PID, rdf term). the rdf term is the one wikidata's
        # own RDF export uses (wd:/wdt:, or rdfs:label / schema:description /
        # skos:altLabel for labels etc). literal objects get (value, None, None).
        subj = (d.get("label") or qid, qid, f"wd:{qid}")
        structured = []
        if d.get("label"):
            structured.append((subj, ("label", None, "rdfs:label"),
                               (d["label"], None, None)))
        if d.get("description"):
            structured.append((subj, ("description", None, "schema:description"),
                               (d["description"], None, None)))
        for alias in d.get("aliases", []):
            structured.append((subj, ("alias", None, "skos:altLabel"),
                               (alias, None, None)))

            
        for pid, statements in claims_by_qid.get(qid, {}).items():
            if pid in SKIP_PROPERTIES:
                continue
            p_label = labels.get(pid) or pid
            pred = (p_label, pid, f"wdt:{pid}")
            for st in statements:
                val = snak_value(st.get("mainsnak", {}))
                if not val:
                    continue
                # use the label for entity objects, QID only if there is none
                value = (labels.get(val[1]) or val[1]) if val[0] == "entity" else val[1]
                if value:
                    triples.append((qid, p_label, str(value)))
                    if val[0] == "entity":
                        obj = (str(value), val[1], f"wd:{val[1]}")
                    else:
                        obj = (str(value), None, None)
                    structured.append((subj, pred, obj))
        # keep P31 (instance of) as its own field, used for typing
        item["instance_of"] = [
            (labels.get(val[1]) or val[1], val[1])
            for st in claims_by_qid.get(qid, {}).get("P31", [])
            if (val := snak_value(st.get("mainsnak", {}))) and val[0] == "entity"]
        item["triples"] = structured
        parsed = parse_triples(triples)
        out[qid] = [parsed, item] if parsed else False
    return out


def query_wikidata(mention_form):
    candidates_per_form = []
    params = {
        "action": "wbsearchentities",
        #"action": "cirrusDumpResult",
        "wmgUseCirrus": True,
        "format": "json",
        "language": "en",
        "search": mention_form[0].strip(),
        "limit": mention_form[1][0]

    }
    r = api_get(params)
    rankWiki = 0


    if 'error' in r:
	    #raise Exception(r['error']['code'], r['error']['info'])
        print(Exception(r['error']['code'], r['error']['info']))
        pass
    elif "search" in r and r["search"]:
        for item in r["search"]:
            temp = {"mention_form": mention_form[0].strip(), "mention_type": mention_form[1][1]}
            rankWiki += 1
            temp["QID"] = item["id"]
            temp["label"] = item["label"]

            if item["match"]["type"] == "alias":
                temp["linked_by_alias"] = 1
                temp["linked_by_label"] = 0
            else:
                temp["linked_by_alias"] = 0
                temp["linked_by_label"] = 1

            temp["rank_by_wiki"] = rankWiki
            if "description" in item and item["description"] == "'Wikimedia disambiguation page":
                temp["is_disambiguation_page"] = 1
            else:
                temp["is_disambiguation_page"] = 0
            if "description" in item and len(item["description"].strip()) >= 4:
                temp["has_description"] = 1
            else:
                temp["has_description"] = 0

            print("temp = ", temp)
            candidates_per_form.append(temp)
            

    return candidates_per_form





def query_wikidata_ctx(mention_form, context):
    """same as query_wikidata but context aware. cirrus full-text search
    (action=query&list=search): the form has to be in the label, the content
    lemmas of the sentence get ORed in so items matching more of the context
    rank higher. labels are not known at this point (search only returns QIDs),
    candidate_generator fills and checks them afterwards."""
    form = mention_form[0].strip()
    terms = [t.lemma_.lower() for t in nlp(context)
             if t.is_alpha and not t.is_stop
             and t.lemma_.lower() not in form.lower()][:15]
    q = f'inlabel:"{form}"'
    if terms:
        q += " (" + " OR ".join(terms) + ")"
    params = {"action": "query", "list": "search", "srsearch": q,
              "srnamespace": 0, "srlimit": mention_form[1][0], "format": "json"}
    r = api_get(params)
    out = []
    for rank, item in enumerate(r.get("query", {}).get("search", []), 1):
        out.append({"mention_form": form, "mention_type": mention_form[1][1],
                    "QID": item["title"], "label": form,
                    "linked_by_alias": 0, "linked_by_label": 0,
                    "rank_by_wiki": rank, "is_disambiguation_page": 0,
                    "has_description": 0})
    return out


def add_CG_based_features(item, mention):

    item["levenstein_distance"] = Levenshtein.distance(mention.strip(), item["label"].strip())

    if item["label"] == mention.strip():
        item["mention_label_string_match"] = 1
    else:
        item["mention_label_string_match"] = 0


    return item


def process_mentionForm(mention_form, queue, context=None):
    # with context: Cirrus full-text search; without: plain wbsearchentities
    if context is not None:
        candidate_per_form = query_wikidata_ctx(mention_form, context)
    else:
        candidate_per_form = query_wikidata(mention_form)
    queue.put(mention_form)
    return candidate_per_form



def candidate_generator(mention, context, preprocessed=None, use_context=False):

    queue = Manager().Queue()

    # step 1: text preprocessing + mention expansion
    if preprocessed is None:
        preprocessed = text_preprocess(nlp(context), mention=mention)
    expanded_mention_dict = preprocessed["expanded_mention"]
    t1 = time.time()

    # step 2: one wikidata query per expanded form, in parallel. with context
    # we use cirrus full-text search, otherwise plain wbsearchentities
    ctx = context if use_context else None
    with Pool() as pool:
        candidates = pool.starmap(process_mentionForm, [((key, value), queue, ctx) for key, value in expanded_mention_dict.items()])

    # deduplication of candidates based on QID
    candidates_dedup = []
    qidlookup = []
    for sublist in candidates:
        for item in sublist:
            if item["QID"] not in qidlookup:
                qidlookup.append(item["QID"])
                item = add_CG_based_features(item, mention)
                print("dedup item = ", item)
                candidates_dedup.append(item)

    # fetch label/description/aliases + claims in one combined call
    details, claims_by_qid = fetch_entities(qidlookup)
    for item in candidates_dedup:
        d = details.get(item["QID"], {})
        item["label"] = d.get("label") or item.get("label")
        item["description"] = d.get("description")
        item["aliases"] = d.get("aliases", [])

    # the OR terms can leak past the inlabel: filter. now that real labels are
    # in, drop candidates whose label/aliases don't contain the form, and redo
    # the string features (they were computed against a placeholder label)
    if use_context:
        kept = []
        for item in candidates_dedup:
            f = item["mention_form"].lower()
            names = [item.get("label") or ""] + item.get("aliases", [])
            if not any(f in n.lower() for n in names):
                continue
            add_CG_based_features(item, mention)
            kept.append(item)
        candidates_dedup = kept

    print("No of candidates = ", len(candidates_dedup))


    t2 = time.time()

    # step 3: 1-hop triples for all candidates, from the claims fetched above
    candidate_triple_str_und_features = get_1_hop_triples(candidates_dedup, details, claims_by_qid)

    t3 = time.time()
    #print("t2 = ", t3-t2)
    return candidate_triple_str_und_features





def main():
    lookup = []
    p = 0
    n = 0
    k = 0
    target_data = pd.read_csv("/Users/ItishaYadav1/text_to_wikidata_linking/evaluation/dataset_3_sen_level/dataset_3_sen_level.csv")
    for ind, row in target_data.iterrows():
        mention = row["string"]
        if mention not in lookup and row["correct_id"]!=0:
            k += 1
            print(k)
            lookup.append(mention)
            print("mention = ", mention)
            print("context = ", row["text"])
            t1 = time.time()
            candidates = candidate_generator(mention, row["text"])
            for qid, bundle in candidates.items():
                if not bundle:
                    continue
                _, item = bundle
                # every candidate: qid, label, description, aliases
                print(f"qid = {qid} | {item['label']} | {item['description']} | {item['aliases']}")

                # related triples for every candidate QID: subj, pred, obj as (label, id)
                for subj, pred, obj in item["triples"]:
                    print(f"  {subj} --{pred}--> ({obj[0][:100]!r}, {obj[1]})")
                print("--------------------------------------------------")
            print("time = ", time.time()-t1)
            if row["correct_id"] in candidates:
                p += 1
            else:
                n += 1
            
        else:
            continue



        break

    print("pos:", p, "neg:", n, "total:", k)
    print("Recall = ", (p/k)*100)




if __name__ == "__main__":
    main()

# en_core_web_sm pre-processing per sentence: NER entities, POS tags, noun, phrases, verb phrases, lemmas, stems
# NER = spacy + a document level d1 pass: proper-noun / ALL-CAPS spans that
# spacy missed get typed by the d1 decision model against the test dataset's
# label set. catches things like city-named clubs or headline caps.

import re
from functools import lru_cache

import nltk
import requests
from nltk.stem import PorterStemmer
import spacy

nlp_sm = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

OLLAMA = "http://localhost:11434"
NER_MODEL = "d1-omni:600m"

# label types as in the test dataset (el_stratified_dataset.csv type_labels)
NER_TYPES = {
    "person": None,
    "organization": "companies, institutions, sports clubs and teams",
    "geographic entity": "countries, cities, regions, places",
    "position": "job titles, roles, offices",
    "planned event": "competitions, tournaments, holidays",
    "human population": "nationalities, ethnic or religious groups",
    "OTHER": "a named entity of another kind",
    "not_entity": "not a named entity",
}

try:
    from nltk.corpus import wordnet
    wordnet.synsets("test")           # trigger lazy load
except LookupError:
    nltk.download("wordnet")
    from nltk.corpus import wordnet


def covered(mention, terms):
    """true if the mention matches a term exactly or is contained in one."""
    return mention in terms or any(mention in t for t in terms)


def uncovered_spans(doc):
    """proper-noun / ALL-CAPS token runs that spacy's NER didn't already cover,
    so d1 only gets asked about new spans."""
    covered_idx = {i for e in doc.ents for i in range(e.start, e.end)}
    spans, cur = [], []
    for t in doc:
        if t.pos_ == "PROPN" or (t.is_alpha and t.is_upper and len(t.text) > 1):
            cur.append(t)
        else:
            if cur:
                spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    out, seen = [], {e.text.lower().strip() for e in doc.ents}
    for toks in spans:
        s = " ".join(t.text for t in toks)
        if (any(t.i in covered_idx for t in toks) or s.lower() in seen
                or not re.search(r"[A-Za-z]", s)):  # masked junk like '*'
            continue
        seen.add(s.lower())
        out.append(s)
    return out


@lru_cache(maxsize=None)
def d1_ner(text):
    """NER for the spans spacy missed. d1 types each span against NER_TYPES,
    one batched call per document, 'not_entity' spans get dropped.
    returns [(entity, label), ...]."""
    spans = uncovered_spans(nlp_sm(text))
    ents = []
    for i0 in range(0, len(spans), 64):  # systemone takes up to 64 questions
        chunk = spans[i0:i0 + 64]
        qs = {f"s{i}": {"type": "choice",
                        "instructions": f'What kind of named entity is "{s}" in this document?',
                        "criteria": NER_TYPES}
              for i, s in enumerate(chunk)}
        r = requests.post(f"{OLLAMA}/v1/systemone",
                          json={"model": NER_MODEL, "state": text, "questions": qs})
        r.raise_for_status()
        answers = r.json()["answers"]
        ents += [(s, answers[f"s{i}"]["choice"]) for i, s in enumerate(chunk)
                 if answers[f"s{i}"]["choice"] != "not_entity"]
    return ents


def verb_phrases(doc):
    """verb + its auxiliaries/negation/particle, in sentence order."""
    vps = []
    for tok in doc:
        if tok.pos_ == "VERB":
            parts = sorted(
                [tok] + [c for c in tok.children
                         if c.dep_ in ("aux", "auxpass", "neg", "prt")],
                key=lambda t: t.i)
            vps.append(" ".join(t.text for t in parts))
    return vps


def expand_with_synonyms(mention_forms_dict, mention):
    """add wordnet synonyms of the mention as extra query forms."""
    for syn in wordnet.synsets(mention):
        for lm in syn.lemmas():
            lm_name = re.sub(r"_|-", " ", str(lm.name()))
            lm_name = re.sub(r"\s+", " ", lm_name)
            if lm_name not in mention_forms_dict:
                mention_forms_dict[lm_name] = (1, "syn")  # adding into synonyms
    return mention_forms_dict


def mention_expansion(mention, synonyms=True):
    """
    input: single mention string
    output: expanded dict of query forms for the mention
            {form: (search_limit, form_type)} with form_type in
            surface / lemma / stem / syn
    synonyms=False skips the wordnet expansion (used for NER-covered
    mentions, synonyms just add junk queries there).
    """
    # strip leading the/a/an, they are not part of the concept name
    # ("the Nobel Prize in Physics") and break the label search
    mention = re.sub(r"^(the|a|an)\s+", "", mention.strip(), flags=re.I)
    expanded_men = {}
    lemma = " ".join([str(token.lemma_) for token in nlp_sm(mention)
                      if re.search(r"[A-Za-z0-9]", str(token.lemma_))])
    if "-" in mention:
        stem = re.split(r" |-", mention)[0]
    else:
        stem = " ".join([stemmer.stem(token) for token in re.split(r" |-", mention)
                         if re.search(r"[A-Za-z0-9]", stemmer.stem(token))])

    mention = re.sub(r"[^A-Za-z0-9\s]", " ", mention).strip().lower()
    mention = re.sub(r"\s+", " ", mention)

    expanded_men[mention] = (7, "surface")
    if lemma not in expanded_men:
        expanded_men[lemma] = (7, "lemma")
    if stem not in expanded_men:
        expanded_men[stem] = (7, "stem")
    if synonyms:
        expanded_men = expand_with_synonyms(expanded_men, mention)
    # e.g. {'furnace': (7, 'surface'), 'furnac': (7, 'stem'), ...}
    return expanded_men


def find_sentence(doc, start, end):
    """the sentence containing the [start, end) character span."""
    return next((s for s in doc.sents
                 if s.start_char <= start < s.end_char), None)


def text_preprocess(doc, mention=None, span=None, use_llm_ner=True):
    # spacy entities + the d1 pass (d1 only sees what spacy missed)
    sm_ents = [(e.text, e.label_) for e in doc.ents]
    llm_ents = d1_ner(doc.text) if use_llm_ner else []

    # combining both spacy ner and d1_ner
    ents = sm_ents + llm_ents
    ent_texts = {t.lower().strip() for t, _ in ents}

    # mention specific fields: which entity span covers the mention, its NER
    # type, and the expanded query forms for candidate retrieval
    ner_extracted = ner_mention = ner_label = None
    expanded_mention = {}
    if mention is not None:
        m = mention.lower().strip()
        ner_extracted = covered(m, ent_texts)
        ner_mention = next((t for t, _ in ents
                            if m == t.lower().strip() or m in t.lower()), None)
        ner_label = next((l for t, l in ents if t == ner_mention), None)
        expanded_mention = mention_expansion(mention, synonyms=not ner_extracted)

    # sentence split + the sentence the mention is in (by char span when
    # given, otherwise just search for the mention string)
    parent = None
    if span is not None:
        parent = find_sentence(doc, *span)
    elif mention is not None:
        parent = next((s for s in doc.sents if m in s.text.lower()), None)

    return {
        "text":             doc.text,
        "sentences":        [s.text.strip() for s in doc.sents],
        "parent_sentence":  parent.text.strip() if parent is not None else None,
        "expanded_mention": expanded_mention,               # {form: (limit, type)} for the candidate generator
        "ner_extracted":    ner_extracted,                      # covered by sm or LLM NER
        "ner_mention":      ner_mention,
        "ner_label":        ner_label,                          # PERSON, GPE, ORG, ...
        "ner_entities":     ents,                               # sm + LLM entities, merged
        "sm_ner_entities":  sm_ents,                            # spaCy sm only
        "llm_ner_entities": llm_ents,                           # LLM pass only
        "noun_phrases":     [c.text for c in doc.noun_chunks],
        "verb_phrases":     verb_phrases(doc),
        "pos_tags":         [(t.text, t.pos_) for t in doc if not t.is_space],
        "lemmas":           [t.lemma_ for t in doc if not t.is_space],   # all tokens
        "stems":            [stemmer.stem(t.text.lower()) for t in doc if t.is_alpha],
        "context_lemmas":   [t.lemma_.lower() for t in doc
                             if t.is_alpha and not t.is_stop],  # for later embedding/overlap scoring
    }


if __name__ == "__main__":
    from pprint import pprint
    doc = nlp_sm("Venus is a planet in our solar system.")
    pprint(text_preprocess(doc, mention="venus"))


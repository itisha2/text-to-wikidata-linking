# en_core_web_sm pre-processing per sentence: NER entities, POS tags, noun, phrases, verb phrases, lemmas, stems 

from nltk.stem import PorterStemmer
import spacy

nlp_sm = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()


def covered(mention, terms):
    """True if the mention matches a term exactly or is contained in one."""
    return mention in terms or any(mention in t for t in terms)


def verb_phrases(doc):
    """Verb + its auxiliaries/negation/particle, in sentence order."""
    vps = []
    for tok in doc:
        if tok.pos_ == "VERB":
            parts = sorted(
                [tok] + [c for c in tok.children
                         if c.dep_ in ("aux", "auxpass", "neg", "prt")],
                key=lambda t: t.i)
            vps.append(" ".join(t.text for t in parts))
    return vps


def text_preprocess(doc, mention=None):
    ents = [(e.text, e.label_) for e in doc.ents]
    ent_texts = {e.text.lower().strip() for e in doc.ents}

    # the entity span that covers the gold mention, and its NER type
    ner_extracted = ner_mention = ner_label = None
    if mention is not None:
        m = mention.lower().strip()
        ner_extracted = covered(m, ent_texts)
        ner_mention = next((t for t, _ in ents
                            if m == t.lower().strip() or m in t.lower()), None)
        ner_label = next((l for t, l in ents if t == ner_mention), None)

    return {
        "text":             doc.text,
        "sm_ner_extracted": ner_extracted,
        "ner_mention":      ner_mention,
        "ner_label":        ner_label,                          # PERSON, GPE, ORG, ...
        "ner_entities":     ents,                               # all entities in the sentence
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

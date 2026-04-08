import json
import os
import re

import pandas as pd
import spacy
import tqdm
from spacy.symbols import VERB, nsubj

nlp = spacy.load("en_core_web_sm")


def get_subject_verb_pairs(t):
    doc = nlp(t)
    subject_verb_pairs = []
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            subject_verb_pairs.append({possible_subject: possible_subject.head.lemma_})
    if len(subject_verb_pairs) > 0:
        return subject_verb_pairs
    else:
        return None


not_verbs = []

COLOR_WORDS = frozenset({
    "red", "gray", "grey", "blue", "green", "yellow", "white", "black",
    "brown", "orange", "purple", "pink", "silver", "gold",
})


def get_aux_verbs(t, main_verb_lemma):
    doc = nlp(t)
    aux = []
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        if tok.lemma_ == main_verb_lemma:
            continue
        # Skip past participles used as adjectival modifiers
        # (e.g. "angled", "perforated")
        if tok.tag_ in ("VBN", "VBD") and tok.dep_ == "amod":
            continue
        # Skip VBD/VBN with dep=conj — in enumerated lists spaCy
        # often mis-parses adjectival past participles (e.g. "angled")
        # as conj of another verb when they're really noun modifiers.
        if tok.tag_ in ("VBN", "VBD") and tok.dep_ == "conj":
            continue
        # Allow participial clauses modifying nouns (acl, relcl)
        # e.g. "booklet containing instructions", "bar placed on table"
        # but skip other verbs whose head is a noun
        if tok.head.pos_ in ("NOUN", "PROPN") and tok.dep_ not in ("acl", "relcl"):
            continue
        # "prep" covers participial prepositions (e.g. "following instructions")
        if tok.dep_ in ("xcomp", "advcl", "conj", "acl", "relcl", "prep"):
            aux.append(tok.lemma_)

    seen = set()
    return [v for v in aux if not (v in seen or seen.add(v))]


def get_preposition_object_pairs(t):
    doc = nlp(t)
    preposition_object_pairs = []
    for possible_object in doc:
        if possible_object.dep_ == "pobj" and possible_object.head.dep_ == "prep":
            pobj_str = noun_phrase(possible_object)
            prep_str = possible_object.head.lemma_
            preposition_object_pairs.append({pobj_str: prep_str})
    if len(preposition_object_pairs) > 0:
        return preposition_object_pairs
    else:
        return None


def check_verb(token):
    if token.pos_ == "VERB":
        indirect_object = False
        direct_object = False
        for item in token.children:
            if item.dep_ == "iobj" or item.dep_ == "pobj":
                indirect_object = True
            if item.dep_ == "dobj" or item.dep_ == "dative":
                direct_object = True
        if indirect_object and direct_object:
            return "DITRANVERB"
        elif direct_object and not indirect_object:
            return "TRANVERB"
        elif not direct_object and not indirect_object:
            return "INTRANVERB"
        else:
            return "VERB"


def check_dobj(token):
    if token.dep_ == "dobj":
        return token
    return None


def check_if_obj(token):
    if token.pos_ in ("NOUN", "PROPN"):
        # direct/indirect objects
        if token.dep_ in ("dobj", "obj", "iobj", "dative"):
            return str(token)

        # prepositional object
        if token.dep_ == "pobj" and token.head.dep_ == "prep":
            return str(token)

        # appositional or fallback deps (common in long enumerated lists
        # where spaCy mis-parses the dependency structure)
        if token.dep_ in ("appos", "dep"):
            return str(token)

        # coordinated noun — accept any NOUN/PROPN conjunct.
        # spaCy can assign the conj head to a noun *or* a verb
        # depending on sentence complexity (e.g. "there are scattered
        # red and gray plastic pieces, a monitor, ...").
        if token.dep_ == "conj":
            return str(token)

    return None


def extract_noun_compounds(token):
    compounds = []

    modifiers = []
    for child in token.children:
        if child.dep_ in ("compound", "amod", "nmod") and child.pos_ in (
            "NOUN",
            "ADJ",
            "PROPN",
        ):
            modifiers.append(child)
        # Past participles (e.g. "angled", "perforated") tagged as VERB
        elif child.dep_ == "amod" and child.pos_ == "VERB":
            modifiers.append(child)
        # Numeric modifiers (e.g. "4" in "4 perforated junction bar")
        elif child.dep_ == "nummod" and child.pos_ == "NUM":
            modifiers.append(child)

    modifiers = sorted(modifiers, key=lambda x: x.i)

    if modifiers:
        compound_phrase = " ".join([str(mod) for mod in modifiers]) + " " + str(token)
        compounds.append(compound_phrase)

    return compounds


def standardize_narration(t):
    if t:
        t = t.replace("The camera wearer", "The_camera_wearer")
        t = (
            t.replace("#Unsure", "")
            .replace("#unsure", "")
            .replace("#Sammary", "")
            .replace("#sammary", "")
            .replace("#Summary", "")
        )
        t = t.strip()
        t = re.sub(" +", " ", t)
        t = re.sub("#\w", "", t)
        t = t.strip()
        if t[0].islower():
            t = t[0].upper() + t[1:]
        if t.endswith("."):
            t = t[:-1]
        return t.strip()
    else:
        return ""


def noun_phrase(token):
    compounds = extract_noun_compounds(token)
    if compounds:
        return compounds[0]
    return str(token)


def norm_obj(s: str) -> str:
    return " ".join(s.lower().split())


def get_aux_verb_object_map(t, aux_verb_lemma, all_objects):
    doc = nlp(t)
    all_norm = {norm_obj(o): o for o in all_objects}

    found = []
    for v in doc:
        if v.pos_ != "VERB":
            continue
        if v.lemma_ != aux_verb_lemma:
            continue

        candidate_phrases = []

        for ch in v.children:
            if ch.dep_ in ("dobj", "obj"):
                # Use object_record for consistency with the left-scan
                # approach (captures coordinated colours, etc.)
                recs = object_record(ch)
                for rec in recs:
                    candidate_phrases.append(rec["raw"])

            if ch.dep_ == "prep":
                # Skip prep-pobj when a comma sits between the verb and
                # the preposition — that comma marks a clause boundary
                # (e.g. "following instructions …, with a monitor")
                has_comma = any(
                    doc[k].text == ","
                    for k in range(v.i + 1, ch.i)
                )
                if not has_comma:
                    for pobj in ch.children:
                        if pobj.dep_ == "pobj" and pobj.pos_ in ("NOUN", "PROPN"):
                            recs = object_record(pobj)
                            for rec in recs:
                                candidate_phrases.append(rec["raw"])

        for cand in candidate_phrases:
            key = norm_obj(cand)
            if key in all_norm:
                found.append(all_norm[key])

        seen = set()
        found = [x for x in found if not (x in seen or seen.add(x))]
        return found

    return []


def parse_annotate_action(action):
    standardized = standardize_narration(action)
    preposition_object_pairs = get_preposition_object_pairs(standardized)
    subject_verb_pairs = get_subject_verb_pairs(standardized)
    if subject_verb_pairs is None or len(subject_verb_pairs) == 0:
        return None, None, None, None, None, None, None, None, None, None

    subj_verb_dict = subject_verb_pairs[0]
    subject, verb = list(subj_verb_dict.keys())[0], list(subj_verb_dict.values())[0]

    aux_verbs = get_aux_verbs(standardized, verb)

    direct_object = None
    all_objects_map = {}
    base_objects = []
    attributes = []
    pos_mask = None
    tag_mask = None
    dep_mask = None
    verb_type = "VERB"
    phrasal_verb = None

    doc = nlp(standardized)
    verb_type = "VERB"
    prev_is_verb = False
    main_verb_token = None
    pos_toks, tag_toks, dep_toks = [], [], []
    for token in doc:
        pos_toks.append(token.pos_)
        tag_toks.append(token.tag_)
        dep_toks.append(token.dep_)

        if token.lemma_ == verb:
            prev_is_verb = True
            verb_type = check_verb(token)
            main_verb_token = token

        if token.dep_ == "prt" and prev_is_verb:
            phrasal_verb = str(verb) + "-" + str(token)

        res_dobj = check_dobj(token)
        if res_dobj is not None:
            recs_dobj = object_record(res_dobj)
            rec_dobj = recs_dobj[0]  # use primary record for direct object
            if direct_object is None:
                direct_object = rec_dobj["raw"]
            # Prefer the dobj directly attached to the main verb
            if main_verb_token is not None and res_dobj.head == main_verb_token:
                direct_object = rec_dobj["raw"]

        is_object = check_if_obj(token)
        if is_object is not None:
            recs = object_record(token)
            for rec in recs:
                all_objects_map[rec["raw"]] = {
                    "base_object": rec["base"],
                    "attributes": rec["attrs"],
                }

                base_objects.append(rec["base"])
                attributes.extend(rec["attrs"])

    pos_mask = str(pos_toks)
    tag_mask = str(tag_toks)
    dep_mask = str(dep_toks)

    if direct_object is None and all_objects_map:
        direct_object = next(iter(all_objects_map.keys()))

    object_aux_verb = {}
    for av in aux_verbs:
        object_aux_verb[av] = get_aux_verb_object_map(
            doc, av, list(all_objects_map.keys())
        )

    object_aux_verb = str(object_aux_verb)

    return (
        subject,
        verb,
        verb_type,
        direct_object,
        all_objects_map,
        base_objects,
        attributes,
        phrasal_verb,
        preposition_object_pairs,
        pos_mask,
        tag_mask,
        dep_mask,
        aux_verbs,
        object_aux_verb,
    )


def object_record(token):
    """Build object record(s) for head-noun *token*.

    Uses a left-scan approach instead of relying on spaCy's dep-tree
    children, because spaCy frequently mis-parses long enumerated lists.
    Returns a **list** of records.  Usually one, but can be multiple when
    coordinated colours are found (e.g. "red and gray perforated bars"
    → two entries).
    """
    doc = token.doc
    base_mods = []
    attr_mods = []

    def _is_color(t):
        return t.text.lower() in COLOR_WORDS

    # --- left-scan: walk backwards from the head noun ---
    idx = token.i - 1
    while idx >= 0:
        t = doc[idx]
        # stop at punctuation / clause boundaries
        if t.text in (",", ".", ";", ":") or t.pos_ == "PUNCT":
            break
        # Color words can be mis-tagged as VERB/NOUN/PROPN by spaCy;
        # always treat them as colour attributes regardless of POS.
        if _is_color(t):
            attr_mods.insert(0, t)
            idx -= 1
            continue
        # stop at real verbs (not past-participle modifiers)
        if t.pos_ == "VERB" and t.tag_ not in ("VBN", "VBD"):
            break
        if t.pos_ in ("ADP", "SCONJ"):
            break
        # stop at conjunctions — they separate noun phrases in enumerations
        if t.pos_ == "CCONJ":
            break
        # skip determiners
        if t.pos_ == "DET":
            idx -= 1
            continue

        # --- classify the modifier ---
        if t.tag_ in ("VBN", "VBD"):
            # past-participle descriptors → base  (angled, perforated)
            base_mods.insert(0, t)
        elif t.pos_ == "NUM":
            base_mods.insert(0, t)
        elif t.pos_ in ("NOUN", "PROPN"):
            base_mods.insert(0, t)
        elif t.pos_ == "ADJ":
            if t.text.lower().endswith("ed"):
                # adjective that looks like a past participle
                # (spaCy sometimes tags "perforated" as JJ)
                base_mods.insert(0, t)
            else:
                attr_mods.insert(0, t)
        else:
            break  # unknown token type → end of NP

        idx -= 1

    # --- detect coordinated colours before CCONJ ---
    # e.g. "red and gray perforated bars": at this point we have
    # attr_mods=[gray], base_mods=[perforated], head=bars, and idx
    # points to "and".  Scan further left to collect extra colours.
    extra_colors = []
    if idx >= 0 and doc[idx].pos_ == "CCONJ":
        j = idx - 1
        while j >= 0:
            ct = doc[j]
            if ct.pos_ == "PUNCT" or ct.text in (",", ".", ";", ":"):
                break
            # Color words may be tagged as ADJ, NOUN, PROPN, or even VERB
            if _is_color(ct):
                extra_colors.insert(0, ct)
                j -= 1
                continue
            if ct.pos_ == "CCONJ" or ct.pos_ == "DET":
                j -= 1
                continue
            break

    base_tokens = base_mods + [token]
    non_color_attr_mods = [t for t in attr_mods if t.text.lower() not in COLOR_WORDS]

    def base_token_str(t):
        if t.pos_ in ("NOUN", "PROPN"):
            return t.lemma_.lower()
        return t.text.lower()

    base_phrase = " ".join([base_token_str(t) for t in base_tokens])

    raw_tokens = sorted(set(attr_mods + base_mods + [token]), key=lambda x: x.i)
    raw_phrase = " ".join([t.text for t in raw_tokens]).lower()

    attrs = [t.text.lower() for t in attr_mods]

    records = [{"raw": raw_phrase, "base": base_phrase, "attrs": attrs}]

    # Create additional entries for coordinated colours
    for color_tok in extra_colors:
        extra_all = sorted(
            set([color_tok] + non_color_attr_mods + base_mods + [token]),
            key=lambda x: x.i,
        )
        extra_raw = " ".join([t.text for t in extra_all]).lower()
        extra_attrs = [color_tok.text.lower()] + [
            t.text.lower() for t in non_color_attr_mods
        ]
        records.append({"raw": extra_raw, "base": base_phrase, "attrs": extra_attrs})

    return records


def parse_annotate_folder(input_path):
    clips = [
        clip
        for clip in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, clip))
    ]
    objects = {}
    relationships = {}
    relationships["direct_object"] = 0
    verbs = {}
    attributes_dict = {}

    for clip in tqdm.tqdm(clips, desc=f"Parsing annotations from {input_path}"):
        rows = []
        with open(os.path.join(input_path, clip, "actions.txt"), "r") as f:
            actions_list = [line.strip() for line in f.readlines()]

        for i, action in tqdm.tqdm(enumerate(actions_list), total=len(actions_list), desc=f"Parsing actions in clip {clip}"):
            result = parse_annotate_action(action)
            if result[0] is not None:
                (
                    subject,
                    verb,
                    verb_type,
                    direct_object,
                    all_objects_map,
                    base_objects,
                    attributes,
                    phrasal_verb,
                    preposition_object_pairs,
                    pos_mask,
                    tag_mask,
                    dep_mask,
                    aux_verbs,
                    object_aux_verb,
                ) = result

                if base_objects:
                    for obj in base_objects:
                        objects[obj] = objects.get(obj, 0) + 1
                if attributes:
                    for attr in attributes:
                        attributes_dict[attr] = attributes_dict.get(attr, 0) + 1
                verbs[verb] = verbs.get(verb, 0) + 1
                for _verb in aux_verbs:
                    verbs[_verb] = verbs.get(_verb, 0) + 1

                if preposition_object_pairs:
                    for pdict in preposition_object_pairs:
                        for _, v in pdict.items():
                            relationships[v] = relationships.get(v, 0) + 1

                rows.append(
                    {
                        "frame_id": i,
                        "subject": str(subject),
                        "verb": verb,
                        "verb_type": verb_type,
                        "direct_object": direct_object,
                        "all_objects": json.dumps(all_objects_map, ensure_ascii=False),
                        "phrasal_verb": phrasal_verb,
                        "preposition_object_pairs": str(preposition_object_pairs),
                        "pos_mask": pos_mask,
                        "tag_mask": tag_mask,
                        "dep_mask": dep_mask,
                        "aux_verbs": aux_verbs,
                        "object_aux_verb": object_aux_verb,
                    }
                )

        if rows:
            clip_dataframe = pd.DataFrame(rows)
            clip_dataframe.to_csv(
                os.path.join(input_path, clip, "parse_annotation.csv"), index=False
            )

    if objects:
        objects_enum = {obj: idx for idx, obj in enumerate(sorted(objects.keys()))}
        with open(os.path.join(input_path, "objects.json"), "w") as f:
            json.dump(objects_enum, f, indent=2)

        with open(os.path.join(input_path, "objects_occurrences.json"), "w") as f:
            json.dump(objects, f, indent=2)

    if relationships:
        relationships_enum = {
            rel: idx for idx, rel in enumerate(sorted(relationships.keys()))
        }
        with open(os.path.join(input_path, "relationships.json"), "w") as f:
            json.dump(relationships_enum, f, indent=2)

        with open(os.path.join(input_path, "relationships_occurrences.json"), "w") as f:
            json.dump(relationships, f, indent=2)

    if verbs:
        verbs_enum = {verb: idx for idx, verb in enumerate(sorted(verbs.keys()))}
        with open(os.path.join(input_path, "verbs.json"), "w") as f:
            json.dump(verbs_enum, f, indent=2)

        with open(os.path.join(input_path, "verbs_occurrences.json"), "w") as f:
            json.dump(verbs, f, indent=2)


    if attributes_dict:
        attributes_enum = {
            attribute: idx
            for idx, attribute in enumerate(sorted(attributes_dict.keys()))
        }
        with open(os.path.join(input_path, "attributes.json"), "w") as f:
            json.dump(attributes_enum, f, indent=2)

        with open(os.path.join(input_path, "attributes_occurrences.json"), "w") as f:
            json.dump(attributes_dict, f, indent=2)

    statistics = {
        "total_counts": {
            "unique_objects": len(objects),
            "unique_relationships": len(relationships),
            "unique_verbs": len(verbs),
            "unique_attributes": len(attributes_dict),
            "total_object_occurrences": sum(objects.values()) if objects else 0,
            "total_relationship_occurrences": (
                sum(relationships.values()) if relationships else 0
            ),
            "total_verb_occurrences": sum(verbs.values()) if verbs else 0,
            "total_attributes_occurences": (
                sum(attributes_dict.values()) if attributes_dict else 0
            ),
        }
    }

    with open(os.path.join(input_path, "dataset_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=2)

def main():
    input_dataset_folder = "/home/s3758869/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
    parse_annotate_folder(os.path.join(input_dataset_folder, "Val"))
    parse_annotate_folder(os.path.join(input_dataset_folder, "Test"))
    parse_annotate_folder(os.path.join(input_dataset_folder, "Train"))

if __name__ == "__main__":
    main()

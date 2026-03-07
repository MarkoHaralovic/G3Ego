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


not_verbs = [
    "cheese",
    "counter",
    "fork",
    "orange",
    "oven",
    "pattern",
    "shin",
    "spoon",
    "stove",
    "utensil",
]


def get_aux_verbs(t, main_verb_lemma):
    doc = nlp(t)
    aux = []
    for tok in doc:
        if (
            (tok.pos_ != "VERB")
            or (tok.lemma_ == main_verb_lemma)
            or (tok.head.pos_ in ("NOUN", "PROPN"))
            or (tok.tag_ == "VBN" and tok.dep_ in ("acl", "amod"))
        ):
            continue
        if tok.dep_ in ("xcomp", "advcl", "conj"):
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

        # coordinated noun: "couch and picture"
        if token.dep_ == "conj" and token.head.pos_ in ("NOUN", "PROPN"):
            # keep it if the head noun is something we would have kept
            head = token.head
            if head.dep_ in ("dobj", "obj", "iobj", "dative"):
                return str(token)
            if head.dep_ == "pobj" and head.head.dep_ == "prep":
                return str(token)
            if head.dep_ in ("pobj", "conj"):
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
                candidate_phrases.append(noun_phrase(ch))

            if ch.dep_ == "prep":
                for pobj in ch.children:
                    if pobj.dep_ == "pobj" and pobj.pos_ in ("NOUN", "PROPN"):
                        candidate_phrases.append(noun_phrase(pobj))

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
    pos_toks, tag_toks, dep_toks = [], [], []
    for token in doc:
        pos_toks.append(token.pos_)
        tag_toks.append(token.tag_)
        dep_toks.append(token.dep_)

        if token.lemma_ == verb:
            prev_is_verb = True
            verb_type = check_verb(token)

        if token.dep_ == "prt" and prev_is_verb:
            phrasal_verb = str(verb) + "-" + str(token)

        res_dobj = check_dobj(token)
        if res_dobj is not None:
            direct_object = object_record(res_dobj)["raw"]

        is_object = check_if_obj(token)
        if is_object is not None:
            # compounds = extract_noun_compounds(token)
            # if compounds:
            #    all_objects.extend(compounds)
            #    base_objects.append(is_object)
            #    attributes = [atr for atr in compounds[0].split(is_object)[0].strip().split(" ")]
            #    attributes.extend(attributes)
            # else:
            #    all_objects.append(is_object)
            rec = object_record(token)

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
    base_mods = []
    attr_mods = []

    for ch in token.children:
        if ch.dep_ in ("compound",) and ch.pos_ in ("NOUN", "PROPN"):
            base_mods.append(ch)

        if ch.dep_ in ("amod",) and ch.pos_ == "ADJ":
            attr_mods.append(ch)

        if ch.dep_ == "amod" and ch.pos_ in ("DET",):
            attr_mods.append(ch)

    base_mods = sorted(base_mods, key=lambda x: x.i)
    attr_mods = sorted(attr_mods, key=lambda x: x.i)

    base_tokens = base_mods + [token]

    def base_token_str(t):
        if t.pos_ in ("NOUN", "PROPN"):
            return t.lemma_.lower()
        return t.text.lower()

    base_phrase = " ".join([base_token_str(t) for t in base_tokens])

    raw_tokens = sorted(set(attr_mods + base_mods + [token]), key=lambda x: x.i)
    raw_phrase = " ".join([t.text for t in raw_tokens]).lower()

    attrs = [t.text.lower() for t in attr_mods]

    return {"raw": raw_phrase, "base": base_phrase, "attrs": attrs}


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
    activities = {}
    attributes_dict = {}

    for clip in tqdm.tqdm(clips, desc=f"Parsing annotations from {input_path}"):
        rows = []
        with open(os.path.join(input_path, clip, "actions.txt"), "r") as f:
            actions_list = [line.strip() for line in f.readlines()]

        with open(os.path.join(input_path, clip, "activities.txt"), "r") as f:
            activities_list = [line.strip() for line in f.readlines()]

        for act in activities_list:
            activities[act] = activities.get(act, 0) + 1

        for i, action in enumerate(actions_list):
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

    if activities:
        activities_enum = {
            act: idx for idx, act in enumerate(sorted(activities.keys()))
        }
        with open(os.path.join(input_path, "activities.json"), "w") as f:
            json.dump(activities_enum, f, indent=2)

        with open(os.path.join(input_path, "activities_occurrences.json"), "w") as f:
            json.dump(activities, f, indent=2)

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
            "unique_activities": len(activities),
            "unique_attributes": len(attributes_dict),
            "total_object_occurrences": sum(objects.values()) if objects else 0,
            "total_relationship_occurrences": (
                sum(relationships.values()) if relationships else 0
            ),
            "total_verb_occurrences": sum(verbs.values()) if verbs else 0,
            "total_activity_occurrences": sum(activities.values()) if activities else 0,
            "total_attributes_occurences": (
                sum(attributes_dict.values()) if attributes_dict else 0
            ),
        }
    }

    with open(os.path.join(input_path, "dataset_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=2)


def main():
    INPUT_DATASET_FOLDER = (
        "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
    )
    parse_annotate_folder(INPUT_DATASET_FOLDER)


if __name__ == "__main__":
    main()

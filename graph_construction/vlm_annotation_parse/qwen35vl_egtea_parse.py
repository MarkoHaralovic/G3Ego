#!/usr/bin/env python3
"""Parse EGTEA Qwen VLM annotation CSVs into graph-friendly parse CSVs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter
from ast import literal_eval
from pathlib import Path
from typing import Any

try:
    import tqdm
except ModuleNotFoundError:
    tqdm = None


DEFAULT_INPUT_ROOT = Path(
    "/path/to/ego_graphs/vlm_datasets/egtea_gaze/"
    "vlm_ann_Qwen3-VL-32B-Instruct"
)

QUANTITY_PREFIXES = frozenset(
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "single",
        "double",
        "triple",
        "multiple",
        "several",
    }
)

BRAND_PREFIXES = frozenset(
    {
        "barilla",
        "dixie",
        "heinz",
        "honey",
        "jif",
        "morton",
        "nescafe",
        "nescafé",
        "philadelphia",
        "progresso",
        "sargento",
        "vego",
        "vigo",
        "vlasic",
        "yago",
    }
)

DESCRIPTOR_PREFIXES = frozenset(
    {
        "100",
        "%",
        "adjacent",
        "bella",
        "beta",
        "brand",
        "cartoon",
        "colored",
        "crispy",
        "covered",
        "delta",
        "extra",
        "folded",
        "framed",
        "lit",
        "metal",
        "metallic",
        "paper",
        "peta",
        "plastic",
        "pure",
        "steel",
        "striped",
        "stuffed",
        "wrapped",
        "stacked",
        "textured",
        "themed",
        "thin",
        "ultra",
        "wooden",
    }
)

OBJECT_CLASS_PREFIXES = QUANTITY_PREFIXES | BRAND_PREFIXES | DESCRIPTOR_PREFIXES

IRREGULAR_SINGULARS = {
    "leaves": "leaf",
}

UNCOUNTABLE_OBJECTS = frozenset(
    {
        "asparagus",
        "cheese",
        "clothes",
        "glass",
        "lettuce",
        "mayonnaise",
        "pasta",
        "rice",
        "scissors",
    }
)

PROCESS_STATE_TOKENS = frozenset(
    {
        "boiling",
        "cooking",
        "frying",
        "heating",
        "melting",
        "toasting",
    }
)

PROCESS_STATE_OBJECT_EXCEPTIONS = frozenset(
    {
        "frying pan",
        "heating element",
    }
)

GENERIC_OBJECT_BASES_TO_DROP = frozenset(
    {
        "area",
        "content",
        "item",
        "material",
        "object",
        "piece",
        "product",
        "section",
        "surface",
        "thing",
    }
)

STRICT_VERB_ALLOWLIST = frozenset(
    {
        "add",
        "adjust",
        "apply",
        "arrange",
        "assemble",
        "attach",
        "boil",
        "break",
        "chop",
        "clean",
        "close",
        "coat",
        "connect",
        "cook",
        "cover",
        "crack",
        "crush",
        "cut",
        "decorate",
        "dip",
        "discard",
        "dispense",
        "dispose",
        "drain",
        "draw",
        "drink",
        "drop",
        "dry",
        "eat",
        "empty",
        "feed",
        "fill",
        "fix",
        "flip",
        "fold",
        "fry",
        "gather",
        "grate",
        "grind",
        "hang",
        "heat",
        "hold",
        "insert",
        "inspect",
        "knead",
        "lift",
        "light",
        "load",
        "look",
        "melt",
        "mix",
        "move",
        "open",
        "operate",
        "paint",
        "peel",
        "pick",
        "place",
        "plug",
        "point",
        "pour",
        "prepare",
        "press",
        "pull",
        "put",
        "reach",
        "read",
        "remove",
        "rinse",
        "roll",
        "scoop",
        "scrub",
        "seal",
        "season",
        "separate",
        "serve",
        "shape",
        "sharpen",
        "shred",
        "sift",
        "sit",
        "slice",
        "sort",
        "spray",
        "spread",
        "sprinkle",
        "squeeze",
        "stack",
        "stand",
        "step",
        "stir",
        "sweep",
        "take",
        "tap",
        "tear",
        "throw",
        "tie",
        "tighten",
        "toast",
        "touch",
        "transfer",
        "turn",
        "unfold",
        "unpack",
        "unroll",
        "unscrew",
        "unwrap",
        "use",
        "walk",
        "wash",
        "whisk",
        "wipe",
        "wrap",
        "write",
    }
)


def get_parse_annotate_action():
    try:
        from graph_construction.vlm_annotation_parse.qwen35vl_meccano_parse import parse_annotate_action
    except ModuleNotFoundError as exc:
        if exc.name != "graph_construction":
            raise
        from qwen35vl_meccano_parse import parse_annotate_action
    return parse_annotate_action


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm.tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--pattern", default="annotations_qwen3vl_32b_instruct.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-name",
        default="parse_annotation.csv",
        help="Filename written next to each annotation CSV.",
    )
    parser.add_argument(
        "--vocab-filter-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON with invalid_objects, invalid_verbs, and/or "
            "invalid_attributes lists. These are applied in addition to "
            "issues.py and patched_vocab/errors.py."
        ),
    )
    parser.add_argument(
        "--no-default-vocab-filters",
        action="store_true",
        help="Do not load invalid terms from issues.py and patched_vocab/errors.py.",
    )
    return parser.parse_args()


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default
    value = str(value).strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        try:
            return literal_eval(value)
        except Exception:
            return default


def load_python_globals(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


def as_term_set(value: Any) -> set[str]:
    if not isinstance(value, (set, frozenset, list, tuple)):
        return set()
    return {str(item).strip().lower() for item in value if str(item).strip()}


def load_vocab_filters(
    input_root: Path,
    filter_json: Path | None,
    use_default_filters: bool,
) -> dict[str, set[str]]:
    filters = {
        "objects": set(),
        "verbs": set(),
        "attributes": set(),
        "force_aux_objects": set(),
    }
    if use_default_filters:
        issues = load_python_globals(input_root / "issues.py")
        errors = load_python_globals(input_root / "patched_vocab" / "errors.py")

        for key in (
            "SCENE_OR_REGION_TERMS",
            "ACTION_PROCESS_AS_OBJECT",
            "OCR_OR_PARSE_NOISE",
            "INVALID_OBJECT_TERMS",
        ):
            filters["objects"].update(as_term_set(issues.get(key)))
            filters["objects"].update(as_term_set(errors.get(key)))
        for key in ("NON_VERB_LEAKS", "SUSPICIOUS_VERB_LEAKS"):
            issue_terms = as_term_set(issues.get(key))
            error_terms = as_term_set(errors.get(key))
            filters["verbs"].update(issue_terms)
            filters["verbs"].update(error_terms)
            filters["force_aux_objects"].update(issue_terms)
            filters["force_aux_objects"].update(error_terms)
        for key in ("INVALID_ATTRIBUTE_TERMS",):
            filters["attributes"].update(as_term_set(issues.get(key)))
            filters["attributes"].update(as_term_set(errors.get(key)))

    if filter_json is not None:
        raw = json.loads(filter_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{filter_json} must contain a JSON object")
        aliases = {
            "objects": ("objects", "invalid_objects", "object_noise"),
            "verbs": ("verbs", "invalid_verbs", "verb_noise"),
            "attributes": ("attributes", "invalid_attributes", "attribute_noise"),
            "force_aux_objects": (
                "force_aux_objects",
                "rejected_aux_objects",
                "aux_object_terms",
            ),
        }
        for canonical, keys in aliases.items():
            for key in keys:
                filters[canonical].update(as_term_set(raw.get(key)))

    return filters


def is_invalid(term: Any, invalid_terms: set[str]) -> bool:
    return str(term).strip().lower() in invalid_terms


def is_allowed_verb(term: Any, invalid_terms: set[str]) -> bool:
    term = str(term).strip().lower()
    return bool(term) and term in STRICT_VERB_ALLOWLIST and term not in invalid_terms


def singularize_object_phrase(base: str) -> str:
    tokens = [tok for tok in str(base).strip().lower().split() if tok]
    if not tokens:
        return ""
    head = tokens[-1]
    if head in UNCOUNTABLE_OBJECTS:
        return " ".join(tokens)
    if head in IRREGULAR_SINGULARS:
        tokens[-1] = IRREGULAR_SINGULARS[head]
    elif head.endswith("ies") and len(head) > 3:
        tokens[-1] = head[:-3] + "y"
    elif head.endswith("ves") and len(head) > 3:
        tokens[-1] = head[:-3] + "f"
    elif head.endswith("oes") and len(head) > 3:
        tokens[-1] = head[:-2]
    elif head.endswith("s") and not head.endswith(("ss", "us", "is")) and len(head) > 3:
        tokens[-1] = head[:-1]
    return " ".join(tokens)


def normalize_object_base_and_attrs(
    base: str,
    attrs: list[str],
) -> tuple[str, list[str]]:
    tokens = [tok for tok in str(base).strip().lower().split() if tok]
    attrs_out = [str(attr).strip().lower() for attr in attrs if str(attr).strip()]
    original_base = " ".join(tokens)
    if original_base in PROCESS_STATE_OBJECT_EXCEPTIONS:
        return singularize_object_phrase(original_base), attrs_out

    moved_prefixes = []
    while tokens:
        first = tokens[0]
        if first.isdigit() or first in OBJECT_CLASS_PREFIXES:
            moved_prefixes.append(tokens.pop(0))
            continue
        if len(tokens) > 1 and first.endswith("ed"):
            moved_prefixes.append(tokens.pop(0))
            continue
        break

    moved_process_states = [
        token for token in tokens if token in PROCESS_STATE_TOKENS
    ]
    if moved_process_states:
        tokens = [token for token in tokens if token not in PROCESS_STATE_TOKENS]

    if not tokens:
        return "", attrs_out

    seen = set(attrs_out)
    for prefix in moved_prefixes + moved_process_states:
        if prefix not in seen:
            attrs_out.append(prefix)
            seen.add(prefix)

    base_out = singularize_object_phrase(" ".join(tokens))
    if base_out in GENERIC_OBJECT_BASES_TO_DROP:
        return "", attrs_out
    return base_out, attrs_out


def sanitize_all_objects(
    all_objects: Any,
    invalid_objects: set[str],
    invalid_attributes: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(all_objects, dict):
        return {}

    cleaned: dict[str, dict[str, Any]] = {}
    for raw_phrase, info in all_objects.items():
        if not isinstance(info, dict):
            continue
        raw = str(raw_phrase).strip()
        base = str(info.get("base_object", "")).strip()
        if not raw or not base:
            continue
        if is_invalid(raw, invalid_objects) or is_invalid(base, invalid_objects):
            continue
        attrs = [
            str(attr).strip()
            for attr in info.get("attributes", [])
            if str(attr).strip() and not is_invalid(attr, invalid_attributes)
        ]
        base, attrs = normalize_object_base_and_attrs(base, attrs)
        if not base or is_invalid(base, invalid_objects):
            continue
        cleaned[raw] = {"base_object": base, "attributes": attrs}
    return cleaned


def force_object_from_rejected_aux(
    all_objects_map: dict[str, dict[str, Any]],
    aux_term: Any,
    filters: dict[str, set[str]],
) -> bool:
    raw = str(aux_term).strip().lower()
    if not raw:
        return False
    if raw not in filters.get("force_aux_objects", set()):
        return False
    if is_invalid(raw, filters["objects"]) or is_invalid(raw, filters["attributes"]):
        return False
    base, attrs = normalize_object_base_and_attrs(raw, [])
    if not base or is_invalid(base, filters["objects"]):
        return False
    all_objects_map.setdefault(raw, {"base_object": base, "attributes": attrs})
    return True


def sanitize_parse_result(
    result: tuple[Any, ...],
    filters: dict[str, set[str]],
) -> tuple[Any, ...]:
    (
        subject,
        verb,
        verb_type,
        direct_object,
        all_objects_map,
        _base_objects,
        _attributes,
        phrasal_verb,
        preposition_object_pairs,
        pos_mask,
        tag_mask,
        dep_mask,
        aux_verbs,
        object_aux_verb,
    ) = result

    if not is_allowed_verb(verb, filters["verbs"]):
        return (None,) * 15

    all_objects_map = sanitize_all_objects(
        all_objects_map,
        filters["objects"],
        filters["attributes"],
    )
    base_objects = [info["base_object"] for info in all_objects_map.values()]
    attributes = [
        attr
        for info in all_objects_map.values()
        for attr in info.get("attributes", [])
    ]

    if direct_object not in all_objects_map:
        direct_object = next(iter(all_objects_map.keys()), "")

    kept_aux_verbs = []
    recovered_aux_objects = []
    for aux in aux_verbs or []:
        aux = str(aux).strip()
        if not aux:
            continue
        if is_allowed_verb(aux, filters["verbs"]):
            kept_aux_verbs.append(aux)
        elif force_object_from_rejected_aux(all_objects_map, aux, filters):
            recovered_aux_objects.append(aux)
    aux_verbs = kept_aux_verbs

    if recovered_aux_objects:
        base_objects = [info["base_object"] for info in all_objects_map.values()]
        attributes = [
            attr
            for info in all_objects_map.values()
            for attr in info.get("attributes", [])
        ]
        if not direct_object:
            direct_object = next(iter(all_objects_map.keys()), "")

    object_aux_verb_map = parse_jsonish(object_aux_verb, {})
    if not isinstance(object_aux_verb_map, dict):
        object_aux_verb_map = {}
    object_aux_verb_map = {
        str(aux): [str(obj) for obj in objects if str(obj) in all_objects_map]
        for aux, objects in object_aux_verb_map.items()
        if str(aux) in aux_verbs and isinstance(objects, list)
    }

    prep_pairs = parse_jsonish(preposition_object_pairs, [])
    if isinstance(preposition_object_pairs, list):
        prep_pairs = preposition_object_pairs
    if not isinstance(prep_pairs, list):
        prep_pairs = []
    preposition_object_pairs = [
        {str(obj): str(rel) for obj, rel in pair.items() if str(obj) in all_objects_map}
        for pair in prep_pairs
        if isinstance(pair, dict)
    ]
    preposition_object_pairs = [pair for pair in preposition_object_pairs if pair]
    if not preposition_object_pairs:
        preposition_object_pairs = None

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
        str(object_aux_verb_map),
        recovered_aux_objects,
    )


def empty_counters() -> dict[str, Counter[str]]:
    return {
        "verbs": Counter(),
        "objects": Counter(),
        "attributes": Counter(),
        "relationships": Counter(
            {"direct_object": 0, "aux_direct_object": 0, "aux_verb": 0}
        ),
    }


def count_vocab_from_rows(
    rows: list[dict[str, object]],
    filters: dict[str, set[str]] | None = None,
) -> dict[str, Counter[str]]:
    if filters is None:
        filters = {
            "objects": set(),
            "verbs": set(),
            "attributes": set(),
            "force_aux_objects": set(),
        }
    counters = empty_counters()
    for row in rows:
        verb = str(row.get("verb", "")).strip()
        if is_allowed_verb(verb, filters["verbs"]):
            counters["verbs"][verb] += 1

        all_objects = parse_jsonish(row.get("all_objects"), {})
        if not isinstance(all_objects, dict):
            all_objects = {}
        all_objects = sanitize_all_objects(
            all_objects,
            filters["objects"],
            filters["attributes"],
        )
        aux_verbs = parse_jsonish(row.get("aux_verbs"), [])
        if not isinstance(aux_verbs, list):
            aux_verbs = []
        for aux in aux_verbs:
            aux = str(aux).strip()
            if is_allowed_verb(aux, filters["verbs"]):
                counters["relationships"]["aux_verb"] += 1
            else:
                force_object_from_rejected_aux(all_objects, aux, filters)

        valid_raw_objects = set()
        for obj_info in all_objects.values():
            if not isinstance(obj_info, dict):
                continue
            base_object = obj_info.get("base_object")
            attrs = obj_info.get("attributes", [])
            if not isinstance(attrs, list):
                attrs = []
            base_object, attrs = normalize_object_base_and_attrs(str(base_object), attrs)
            if base_object and not is_invalid(base_object, filters["objects"]):
                counters["objects"][base_object] += 1
                raw = str(row.get("direct_object", "")).strip()
                valid_raw_objects.add(raw)
            for attr in attrs:
                attr = str(attr).strip()
                if attr and not is_invalid(attr, filters["attributes"]):
                    counters["attributes"][attr] += 1

        valid_raw_objects = set()
        for raw, info in all_objects.items():
            if not isinstance(info, dict) or is_invalid(raw, filters["objects"]):
                continue
            normalized_base, _ = normalize_object_base_and_attrs(
                str(info.get("base_object", "")),
                [],
            )
            if normalized_base and not is_invalid(normalized_base, filters["objects"]):
                valid_raw_objects.add(str(raw))

        object_aux_verb = parse_jsonish(row.get("object_aux_verb"), {})
        if not isinstance(object_aux_verb, dict):
            object_aux_verb = {}
        if object_aux_verb:
            counters["relationships"]["aux_direct_object"] += sum(
                len([obj for obj in objects_for_aux if str(obj) in valid_raw_objects])
                for objects_for_aux in object_aux_verb.values()
                if isinstance(objects_for_aux, list)
            )

        prep_pairs = parse_jsonish(row.get("preposition_object_pairs"), [])
        if not isinstance(prep_pairs, list):
            prep_pairs = []
        for prep_pair in prep_pairs:
            if not isinstance(prep_pair, dict):
                continue
            for obj, relation in prep_pair.items():
                if str(obj) not in valid_raw_objects:
                    continue
                relation = str(relation).strip()
                if relation:
                    counters["relationships"][relation] += 1

    return counters


def collect_maps_from_parse(
    path: Path,
    filters: dict[str, set[str]],
) -> dict[str, Counter[str]]:
    if not path.exists():
        return empty_counters()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return count_vocab_from_rows(rows, filters)


def parse_file(
    path: Path,
    output_name: str,
    overwrite: bool,
    filters: dict[str, set[str]],
) -> tuple[int, int, dict[str, Counter[str]]]:
    output_path = path.with_name(output_name)
    if output_path.exists() and not overwrite:
        return 0, 0, collect_maps_from_parse(output_path, filters)

    rows = []
    parsed = 0
    failed = 0
    parse_annotate_action = get_parse_annotate_action()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            action = (row.get("action") or "").strip()
            result = parse_annotate_action(action)
            if result[0] is None:
                failed += 1
                continue
            result = sanitize_parse_result(result, filters)
            if result[0] is None:
                failed += 1
                continue
            (
                subject,
                verb,
                verb_type,
                direct_object,
                all_objects_map,
                _base_objects,
                _attributes,
                phrasal_verb,
                preposition_object_pairs,
                pos_mask,
                tag_mask,
                dep_mask,
                aux_verbs,
                object_aux_verb,
                recovered_aux_objects,
            ) = result
            if object_aux_verb:
                literal_eval(object_aux_verb)
            rows.append(
                {
                    "sample_index": row.get("sample_index", ""),
                    "frame_id": row.get("frame_index", row.get("sample_index", "")),
                    "frame_file": row.get("frame_file", ""),
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
                    "aux_verbs": json.dumps(aux_verbs, ensure_ascii=False),
                    "object_aux_verb": object_aux_verb,
                    "recovered_aux_objects": json.dumps(
                        recovered_aux_objects,
                        ensure_ascii=False,
                    ),
                    "source_action": row.get("source_action", ""),
                    "action_id": row.get("action_id", ""),
                    "verb_id": row.get("verb_id", ""),
                    "noun_ids": row.get("noun_ids", ""),
                    "gaze_x": row.get("gaze_x", ""),
                    "gaze_y": row.get("gaze_y", ""),
                }
            )
            parsed += 1

    if rows:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return parsed, failed, count_vocab_from_rows(rows, filters)


def write_global_maps(
    input_root: Path,
    counters: dict[str, Counter[str]],
) -> None:
    for name in ("verbs", "objects", "attributes", "relationships"):
        counts = Counter(
            {key: value for key, value in counters[name].items() if value > 0}
        )
        enum = {key: idx for idx, key in enumerate(sorted(counts.keys()))}
        with (input_root / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(enum, f, indent=2, ensure_ascii=False)
        with (input_root / f"{name}_occurrences.json").open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(counts.items())), f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    paths = sorted(args.input_root.rglob(args.pattern))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No annotation CSVs found under {args.input_root}")

    total_parsed = 0
    total_failed = 0
    written_files = 0
    filters = load_vocab_filters(
        args.input_root,
        args.vocab_filter_json,
        not args.no_default_vocab_filters,
    )
    counters = empty_counters()
    for path in progress(paths, desc="Parsing EGTEA VLM annotations"):
        parsed, failed, file_counters = parse_file(
            path, args.output_name, args.overwrite, filters
        )
        total_parsed += parsed
        total_failed += failed
        for name, counter in file_counters.items():
            counters[name].update(counter)
        if parsed:
            written_files += 1

    write_global_maps(args.input_root, counters)

    print(
        f"parsed_files={written_files} parsed_rows={total_parsed} "
        f"failed_rows={total_failed} input_files={len(paths)} "
        f"global_verbs={len(counters['verbs'])} "
        f"global_objects={len(counters['objects'])} "
        f"global_attributes={len(counters['attributes'])} "
        f"global_relationships={len(counters['relationships'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

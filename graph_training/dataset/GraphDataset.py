import hashlib
import json
import os
import pickle
from ast import literal_eval
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset.meccano_aux import (
    get_split_name,
    load_clip_text_embeddings
)
from graph_construction.graphs.full_graph import FullActionGraph
from graph_construction.graphs.pruned_graph import PrunedActionGraph


def parse_jsonish(value, default):
    if isinstance(value, (dict, list)):
        return value
    if value is None or pd.isna(value):
        return default
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return literal_eval(value)


def normalize_optional_term(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    return value


def normalize_direct_object(value, objects_atr_map):
    value = normalize_optional_term(value)
    if value in objects_atr_map:
        return value
    if value is not None and objects_atr_map:
        singular_value = value.lower()
        for obj_name, obj_info in objects_atr_map.items():
            base_object = normalize_optional_term(obj_info.get("base_object", ""))
            if obj_name.lower() == singular_value or (
                base_object is not None and base_object.lower() == singular_value
            ):
                return obj_name
    if objects_atr_map:
        return next(iter(objects_atr_map.keys()))
    return None


def new_action_graph(graph_type, verbs, objs, rels, attrs):
    graph_cls = FullActionGraph if graph_type == "full" else PrunedActionGraph
    return graph_cls(verbs, objs, rels, attrs)


def build_action_graph(
    graph_type,
    row,
    vocab,
    clip_feat,
    obj_feats,
    clip_embeddings,
    normalize_object=True,
):
    objects_atr_map = parse_jsonish(row.get("all_objects"), {})
    rels_dict = parse_jsonish(row.get("preposition_object_pairs"), [])
    direct_object = row.get("direct_object")
    if normalize_object:
        direct_object = normalize_direct_object(direct_object, objects_atr_map)

    graph = new_action_graph(graph_type, *vocab)
    kwargs = {
        "verb": row["verb"],
        "direct_object": direct_object,
        "objects_atr_map": objects_atr_map,
        "clip_feat": clip_feat,
        "obj_feats": obj_feats,
        "rels_dict": rels_dict,
        "clip_embeddings": clip_embeddings,
    }
    if graph_type == "full":
        kwargs["aux_verbs"] = parse_jsonish(row.get("aux_verbs"), []) or None
        kwargs["aux_direct_objects_map"] = (
            parse_jsonish(row.get("object_aux_verb"), {}) or None
        )
    else:
        kwargs["gazed_at_object"] = normalize_optional_term(row.get("gazed_at_object"))
    return graph.create_graph(**kwargs)


def base_output(clip_name, block_idx, label_str, activity_to_idx):
    return {
        "clip_name": clip_name,
        "block_idx": block_idx,
        "activity_label": torch.tensor(activity_to_idx[label_str], dtype=torch.long),
        "activity_name": label_str,
    }


class GraphDatasetAria(Dataset):
    def __init__(self, input_path, samples, activity_to_idx, graph_type):
        self.graph_type = graph_type

        with open(os.path.join(input_path, "verbs.json"), "r") as f:
            self.verbs = json.load(f)

        with open(os.path.join(input_path, "objects.json"), "r") as f:
            self.objs = json.load(f)

        with open(os.path.join(input_path, "relationships.json"), "r") as f:
            self.rels = json.load(f)

        with open(os.path.join(input_path, "attributes.json"), "r") as f:
            self.attrs = json.load(f)
            
        clip_text_path = os.path.join(input_path, "clip_text_features.pkl")
        self.clip_textual_embeddings = load_clip_text_embeddings(clip_text_path)
        self.input_path = input_path

        self.samples = samples
        self.h5_paths = sorted(list({s[1] for s in samples}))
        self.h5_to_file_idx = {p: i for i, p in enumerate(self.h5_paths)}
        self.clip_names = [None] * len(self.h5_paths)
        for clip_name, h5_path, _, _, _, _, _, _ in samples:
            self.clip_names[self.h5_to_file_idx[h5_path]] = clip_name

        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
        self.sample_index = [
            (self.h5_to_file_idx[s[1]], s[2], s[3], s[4], s[5], s[6], s[7])
            for s in samples
        ]

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        """Each sample  is:
        h5  file index, block_idx, activity_label, per frame annotationsn, per frame paarsed annotoations, per frame clip feats, object features.
        They are then transformemd in 10 FullActionSceneGraphs
        Output is then:
           clip_name
           activity label
           activity name
           block_idx
           list of 10 FullActionGraphs
        """
        try:
            (
                file_idx,
                block_idx,
                label_str,
                frame_anns,
                frame_parsed_anns,
                frame_feats,
                obj_feats,
            ) = self.sample_index[idx]
            output = base_output(
                self.clip_names[file_idx], block_idx, label_str, self.activity_to_idx
            )
            vocab = (self.verbs, self.objs, self.rels, self.attrs)
            for i, frame_id in enumerate(frame_anns["frame_index"].tolist()):
                row = frame_parsed_anns[frame_parsed_anns["frame_id"] == frame_id].iloc[0]
                output.setdefault("full_action_graphs", {})[i] = build_action_graph(
                    self.graph_type,
                    row,
                    vocab,
                    frame_feats[i],
                    obj_feats[i],
                    self.clip_textual_embeddings,
                    normalize_object=False,
                )
            return output
        except Exception as e:
            print(e)
            print(f"ERROR loading sample {idx}: {type(e).__name__}: {e}")
            print(f"  clip: {self.clip_names[file_idx] if 'file_idx' in locals() else 'unknown'}")
            print(f"  label: {label_str if 'label_str' in locals() else 'unknown'}")
            raise e


class GraphDatasetEgtea(Dataset):
    def __init__(
        self,
        samples,
        activity_to_idx,
        graph_type,
        vocab,
        clip_text_path=None,
        easg_cache_path=None,
        rgb_feature_filename="frame_features_model_dinov3_vitl16.h5",
        additional_feature_mode=None,
        additional_feature_dim=0,
        ohd_feature_filename="hand_grounding_results_gdino_base.pkl",
        ohd_max_hands=2,
        ohd_hand_feature_dim=256,
        default_gdino_feat_dim=256,
    ):
        self.samples = samples
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in activity_to_idx.items()}
        self.graph_type = graph_type
        self.verbs = vocab["verbs"]
        self.objs = vocab["objects"]
        self.rels = vocab["relationships"]
        self.attrs = vocab["attributes"]
        self.clip_textual_embeddings = load_clip_text_embeddings(clip_text_path or "")
        self.easg_cache_path = easg_cache_path
        self.rgb_feature_filename = rgb_feature_filename
        self.additional_feature_mode = (
            str(additional_feature_mode).lower() if additional_feature_mode else None
        )
        self.additional_feature_dim = int(additional_feature_dim)
        self.ohd_feature_filename = ohd_feature_filename
        self.ohd_max_hands = int(ohd_max_hands)
        self.ohd_hand_feature_dim = int(ohd_hand_feature_dim)
        self.ohd_per_hand_dim = self.ohd_hand_feature_dim + 4
        if self.additional_feature_mode == "ohd" and self.additional_feature_dim <= 0:
            self.additional_feature_dim = self.ohd_max_hands * self.ohd_per_hand_dim
        self.cache = self._build_cache(vocab)
        self.default_gdino_feat_dim = default_gdino_feat_dim
        self._clip_cache = {}
        self._ohd_cache = {}
        self.sample_index = [
            (
                sample["clip_name"],
                sample["sample_id"],
                sample["label"],
                sample["clip_dir"],
                sample["feature_dir"],
                sample["frame_numbers"],
            )
            for sample in samples
        ]
        if self.easg_cache_path is not None:
            os.makedirs(self.easg_cache_path, exist_ok=True)

    def __len__(self):
        return len(self.sample_index)

    def _build_cache(self, vocab):
        cache = {
            "graph_type": self.graph_type,
            "preprocessing": "normalize_missing_direct_object_v1",
            "additional_feature_mode": self.additional_feature_mode,
            "additional_feature_dim": self.additional_feature_dim,
            "ohd_feature_filename": (
                self.ohd_feature_filename
                if self.additional_feature_mode == "ohd"
                else None
            ),
            "ohd_max_hands": self.ohd_max_hands,
            "ohd_hand_feature_dim": self.ohd_hand_feature_dim,
            "verbs": vocab["verbs"],
            "objects": vocab["objects"],
            "relationships": vocab["relationships"],
            "attributes": vocab["attributes"],
        }
        encoded = json.dumps(cache, sort_keys=True).encode("utf-8")
        return hashlib.md5(encoded).hexdigest()[:12]

    def _resolve_cache_path(self, clip_name, sample_id, frame_numbers):
        if self.easg_cache_path is None:
            return None
        frame_key = ",".join(str(frame_number) for frame_number in frame_numbers)
        frame_hash = hashlib.md5(frame_key.encode("utf-8")).hexdigest()[:8]
        cache_dir = os.path.join(
            self.easg_cache_path,
            self.graph_type,
            f"vocab_{self.cache}",
            clip_name,
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir, f"sample_{sample_id:06d}_g{len(frame_numbers)}_{frame_hash}.pt"
        )

    def _tensorize_cached_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            value = np.asarray(value).copy()
        return torch.as_tensor(value).cpu()

    def _store_cached_sample(self, cache_path, output, action_scene_graphs):
        cached_output = {
            "clip_name": output["clip_name"],
            "block_idx": output["block_idx"],
            "activity_label": output["activity_label"].detach().cpu(),
            "activity_name": output["activity_name"],
            "cache_signature": self.cache,
            "full_action_graphs": {
                graph_idx: {
                    key: self._tensorize_cached_value(value)
                    for key, value in graph.to_easg_tensors().items()
                }
                for graph_idx, graph in action_scene_graphs.items()
            },
        }
        tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(cached_output, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
        return cached_output

    def _resolve_grounding_path(self, clip_dir):
        matches = sorted(
            os.path.join(clip_dir, name)
            for name in os.listdir(clip_dir)
            if name.startswith("grounding_results_") and name.endswith(".pkl")
        )
        if not matches:
            return None
        preferred = os.path.join(clip_dir, "grounding_results_gdino_base.pkl")
        return preferred if preferred in matches else matches[0]

    def _infer_gdino_feat_dim(self, grounding_results):
        for payload in grounding_results.values():
            for entry in payload.get("objects", {}).values():
                feats = entry.get("feats")
                if feats is not None:
                    return int(feats.shape[0])
            gaze_feats = payload.get("object_gazed_at", {}).get("feats")
            if gaze_feats is not None:
                return int(gaze_feats.shape[0])
        return self.default_gdino_feat_dim

    def _zero_gdino_feature(self, gdino_feat_dim):
        return torch.zeros(gdino_feat_dim, dtype=torch.float32)

    def _sanitize_grounding_payload(self, payload, gdino_feat_dim):
        objects = {}
        for obj_idx, entry in payload.get("objects", {}).items():
            feats = entry.get("feats")
            if feats is None:
                feats = self._zero_gdino_feature(gdino_feat_dim)
            objects[int(obj_idx)] = {
                "feats": feats,
                "phrase": entry.get("phrase"),
                "confidence": entry.get("confidence", 0.0),
            }

        gaze_entry = payload.get("object_gazed_at", {})
        gaze_feats = gaze_entry.get("feats")
        if gaze_feats is None:
            gaze_feats = self._zero_gdino_feature(gdino_feat_dim)
        gaze_idx = gaze_entry.get("idx")
        return {
            "objects": objects,
            "object_gazed_at": {
                "feats": gaze_feats,
                "phrase": gaze_entry.get("phrase"),
                "idx": int(gaze_idx) if gaze_idx is not None else None,
            },
        }

    def _empty_grounding_payload(self, gdino_feat_dim):
        return {
            "objects": {},
            "object_gazed_at": {
                "feats": self._zero_gdino_feature(gdino_feat_dim),
                "phrase": None,
                "idx": None,
            },
        }

    def _load_clip_resources(self, clip_dir, feature_dir):
        cache_key = (clip_dir, feature_dir)
        cached = self._clip_cache.get(cache_key)
        if cached is not None:
            return cached

        parse_annotations = pd.read_csv(os.path.join(clip_dir, "parse_annotation.csv"))
        parse_annotations["frame_number"] = (
            parse_annotations["frame_file"]
            .astype(str)
            .map(lambda value: int(Path(value).stem))
        )

        h5_file = h5py.File(os.path.join(feature_dir, self.rgb_feature_filename), "r")
        frame_num_to_h5_idx = {}
        for idx, frame_name in enumerate(h5_file["frame_names"]):
            if isinstance(frame_name, bytes):
                frame_name = frame_name.decode("utf-8")
            frame_num_to_h5_idx[int(Path(frame_name).stem)] = idx

        grounding_results = None
        gdino_feat_dim = self.default_gdino_feat_dim
        grounding_path = self._resolve_grounding_path(clip_dir)
        if grounding_path is not None:
            with open(grounding_path, "rb") as f:
                grounding_results = pickle.load(f)
            gdino_feat_dim = self._infer_gdino_feat_dim(grounding_results)

        resources = {
            "parse_annotations": parse_annotations,
            "h5_file": h5_file,
            "frame_num_to_h5_idx": frame_num_to_h5_idx,
            "grounding_results": grounding_results,
            "gdino_feat_dim": gdino_feat_dim,
        }
        self._clip_cache[cache_key] = resources
        return resources

    def _load_ohd_payload(self, clip_dir):
        if self.additional_feature_mode != "ohd":
            return None
        cached = self._ohd_cache.get(clip_dir)
        if cached is not None:
            return cached

        ohd_path = os.path.join(clip_dir, self.ohd_feature_filename)
        if not os.path.exists(ohd_path):
            payload = {}
        else:
            with open(ohd_path, "rb") as f:
                payload = pickle.load(f)
        self._ohd_cache[clip_dir] = payload
        return payload

    def _zero_additional_feature(self):
        return np.zeros(self.additional_feature_dim, dtype=np.float32)

    def _to_numpy_feature(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _normalize_mask_bbox(self, hand):
        mask_bbox = hand.get("mask_bbox_xyxy_mask_space")
        mask_size = hand.get("mask_size")
        if mask_bbox is None or mask_size is None:
            mask_bbox = hand.get("mask_bbox_xyxy")
            mask_size = None
        if mask_bbox is None:
            return np.zeros(4, dtype=np.float32)

        coords = np.asarray(mask_bbox, dtype=np.float32).reshape(-1)[:4]
        if mask_size is not None and len(mask_size) >= 2:
            width = max(float(mask_size[0]), 1.0)
            height = max(float(mask_size[1]), 1.0)
            scale = np.asarray([width, height, width, height], dtype=np.float32)
            coords = coords / scale
        return np.clip(coords, 0.0, 1.0).astype(np.float32)

    def _build_ohd_feature(self, ohd_payload, frame_file):
        if self.additional_feature_mode != "ohd":
            return None
        if not ohd_payload:
            return self._zero_additional_feature()

        frame_payload = ohd_payload.get(frame_file)
        if frame_payload is None:
            frame_payload = ohd_payload.get(Path(frame_file).name)
        if frame_payload is None:
            return self._zero_additional_feature()

        compact = self._to_numpy_feature(frame_payload)
        if compact is not None and compact.size == self.additional_feature_dim:
            return compact.astype(np.float32)

        hands = frame_payload.get("hands", []) if isinstance(frame_payload, dict) else []
        hands = sorted(
            hands,
            key=lambda hand: (
                float(hand.get("confidence", 0.0)),
                float(hand.get("mask_area", 0.0)),
            ),
            reverse=True,
        )[: self.ohd_max_hands]

        parts = []
        for hand in hands:
            hand_feat = self._to_numpy_feature(hand.get("feats"))
            if hand_feat is None:
                hand_feat = np.zeros(self.ohd_hand_feature_dim, dtype=np.float32)
            if hand_feat.size < self.ohd_hand_feature_dim:
                hand_feat = np.pad(
                    hand_feat,
                    (0, self.ohd_hand_feature_dim - hand_feat.size),
                    mode="constant",
                )
            hand_feat = hand_feat[: self.ohd_hand_feature_dim].astype(np.float32)
            parts.append(np.concatenate([hand_feat, self._normalize_mask_bbox(hand)]))

        while len(parts) < self.ohd_max_hands:
            parts.append(np.zeros(self.ohd_per_hand_dim, dtype=np.float32))

        ohd_feature = np.concatenate(parts, axis=0).astype(np.float32)
        if ohd_feature.size < self.additional_feature_dim:
            ohd_feature = np.pad(
                ohd_feature,
                (0, self.additional_feature_dim - ohd_feature.size),
                mode="constant",
            )
        return ohd_feature[: self.additional_feature_dim]

    def _build_frame_feature(self, rgb_feature, additional_feature):
        rgb_feature = np.asarray(rgb_feature, dtype=np.float32)
        if self.additional_feature_mode is None:
            return rgb_feature
        if additional_feature is None:
            additional_feature = self._zero_additional_feature()
        return np.concatenate(
            [rgb_feature, np.asarray(additional_feature, dtype=np.float32)],
            axis=0,
        )

    def __getitem__(self, idx):
        try:
            (
                clip_name,
                sample_id,
                label_str,
                clip_dir,
                feature_dir,
                frame_numbers,
            ) = self.sample_index[idx]
            cache_path = self._resolve_cache_path(clip_name, sample_id, frame_numbers)
            if cache_path is not None and os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, map_location="cpu")
                except (EOFError, RuntimeError, OSError, pickle.UnpicklingError):
                    pass

            resources = self._load_clip_resources(clip_dir, feature_dir)
            parse_annotations = resources["parse_annotations"]
            h5_file = resources["h5_file"]
            frame_num_to_h5_idx = resources["frame_num_to_h5_idx"]
            grounding_results = resources["grounding_results"]
            gdino_feat_dim = resources["gdino_feat_dim"]
            ohd_payload = self._load_ohd_payload(clip_dir)

            frame_rows = []
            frame_feats = []
            obj_feats = []
            for frame_number in frame_numbers:
                frame_row = parse_annotations[
                    parse_annotations["frame_number"] == frame_number
                ]
                if frame_row.empty or frame_number not in frame_num_to_h5_idx:
                    raise KeyError(f"Missing frame {frame_number} for clip {clip_name}")
                frame_rows.append(frame_row.iloc[0].copy())
                frame_file = str(frame_row.iloc[0]["frame_file"])
                frame_feats.append(
                    self._build_frame_feature(
                        h5_file["visual_features"][frame_num_to_h5_idx[frame_number]],
                        self._build_ohd_feature(ohd_payload, frame_file),
                    )
                )
                if grounding_results is not None and frame_file in grounding_results:
                    obj_feats.append(
                        self._sanitize_grounding_payload(
                            grounding_results[frame_file], gdino_feat_dim
                        )
                    )
                else:
                    obj_feats.append(self._empty_grounding_payload(gdino_feat_dim))

            frame_parsed_anns = pd.DataFrame(frame_rows).reset_index(drop=True)
            output = base_output(clip_name, sample_id, label_str, self.activity_to_idx)
            action_scene_graphs = {}
            vocab = (self.verbs, self.objs, self.rels, self.attrs)
            for i, row in frame_parsed_anns.iterrows():
                action_scene_graphs[i] = build_action_graph(
                    self.graph_type,
                    row,
                    vocab,
                    frame_feats[i],
                    obj_feats[i],
                    self.clip_textual_embeddings,
                )

            output["full_action_graphs"] = action_scene_graphs
            if cache_path is not None:
                return self._store_cached_sample(cache_path, output, action_scene_graphs)
            return output
        except Exception as e:
            print(f"ERROR loading EGTEA sample {idx}: {type(e).__name__}: {e}")
            if "clip_name" in locals():
                print(f"  clip: {clip_name}")
            raise


class GraphDatasetMeccano(Dataset):
    def __init__(
        self,
        metadata_root,
        split_name,
        samples,
        activity_to_idx,
        graph_type,
        easg_cache_path=None,
        concat_depth_features=False,
        feature_mode=None,
        depth_feature_root=None,
        depth_feature_dim=1536,
        ohg_feature_root=None,
        ohg_feature_dim=20,
        rgb_feature_filename="frame_features_model_dinov3h16+.h5",
        default_gdino_feat_dim=256,
    ):
        self.samples = samples
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
        self.graph_type = graph_type
        self.split_name = get_split_name(split_name)
        self.metadata_root = metadata_root
        self.split_metadata_root = os.path.join(metadata_root, self.split_name)
        self.sample_index = [
            (
                sample["clip_name"],
                sample["sample_id"],
                sample["label"],
                sample["clip_dir"],
                sample["frame_numbers"],
            )
            for sample in samples
        ]
        self._clip_cache = {}
        self._depth_feature_cache = {}
        self._ohg_env_cache = {}
        self.default_gdino_feat_dim = default_gdino_feat_dim
        self.easg_cache_path = easg_cache_path
        self.rgb_feature_filename = rgb_feature_filename
        self.feature_mode = feature_mode or (
            "depth_concat" if concat_depth_features else "rgb"
        )
        self.feature_mode = str(self.feature_mode).lower()
        self.feature_mode = {
            "ogh": "ohg",
            "ogh_only": "ohg",
            "ohg_only": "ohg",
            "rgb_ogh": "rgb_ohg",
            "ogh_concat": "rgb_ohg",
            "ohg_concat": "rgb_ohg",
        }.get(self.feature_mode, self.feature_mode)
        self.concat_depth_features = self.feature_mode == "depth_concat"
        self.use_depth_features = self.feature_mode in {"depth_concat", "depth_only"}
        self.use_ohg_features = self.feature_mode in {"ohg", "rgb_ohg"}
        self.depth_feature_root = depth_feature_root
        self.depth_feature_dim = int(depth_feature_dim)
        self.ohg_feature_root = ohg_feature_root
        self.ohg_feature_dim = int(ohg_feature_dim)

        if self.easg_cache_path is not None:
            os.makedirs(self.easg_cache_path, exist_ok=True)

        self._load_metadata_vocabularies()
        self.clip_textual_embeddings = self._load_clip_text_embeddings()

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _load_clip_text_embeddings(self):
        clip_text_path = os.path.join(self.metadata_root, "clip_text_features.pkl")
        return load_clip_text_embeddings(clip_text_path)

    def _resolve_cache_path(self, clip_name, sample_id, frame_numbers):
        if self.easg_cache_path is None:
            return None
        num_graphs = len(frame_numbers)
        frame_key = ",".join(str(frame_number) for frame_number in frame_numbers)
        frame_hash = hashlib.md5(frame_key.encode("utf-8")).hexdigest()[:8]
        cache_dir = os.path.join(
            self.easg_cache_path,
            self.split_name,
            self.graph_type,
            self.feature_mode,
            clip_name,
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir, f"sample_{sample_id:06d}_g{num_graphs}_{frame_hash}.pt"
        )

    def _tensorize_cached_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            value = np.asarray(value).copy()
        return torch.as_tensor(value).cpu()

    def _build_cached_graphs(self, action_scene_graphs):
        return {
            graph_idx: {
                key: self._tensorize_cached_value(value)
                for key, value in graph.to_easg_tensors().items()
            }
            for graph_idx, graph in action_scene_graphs.items()
        }

    def _store_cached_sample(self, cache_path, output, action_scene_graphs):
        cached_output = {
            "clip_name": output["clip_name"],
            "block_idx": output["block_idx"],
            "activity_label": output["activity_label"].detach().cpu(),
            "activity_name": output["activity_name"],
            "full_action_graphs": self._build_cached_graphs(action_scene_graphs),
        }
        tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(cached_output, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
        return cached_output

    def _load_vocab_group(self, root, prefix=""):
        names = ("verbs", "objects", "relationships", "attributes")
        suffix = ".json"
        return {
            name: self._load_json(os.path.join(root, f"{prefix}{name}{suffix}"))
            for name in names
        }

    def _load_metadata_vocabularies(self):
        global_vocab = self._load_vocab_group(self.metadata_root, prefix="global_")
        self.verbs = global_vocab["verbs"]
        self.objs = global_vocab["objects"]
        self.rels = global_vocab["relationships"]
        self.attrs = global_vocab["attributes"]

        split_vocab = self._load_vocab_group(self.split_metadata_root)
        split_objs = split_vocab["objects"]
        self.split_obj_idx_to_name = {
            idx: name for name, idx in split_objs.items()
        }
        self.split_to_global_obj_idx = {
            split_idx: self.objs[name]
            for split_idx, name in self.split_obj_idx_to_name.items()
            if name in self.objs
        }

    def _zero_gdino_feature(self, gdino_feat_dim):
        return torch.zeros(gdino_feat_dim, dtype=torch.float32)

    def _empty_grounding_payload(self, gdino_feat_dim):
        return {
            "objects": {},
            "object_gazed_at": {
                "feats": self._zero_gdino_feature(gdino_feat_dim),
                "phrase": None,
                "idx": None,
            },
        }

    def _resolve_grounding_path(self, clip_dir):
        matches = sorted(
            os.path.join(clip_dir, file_name)
            for file_name in os.listdir(clip_dir)
            if file_name.startswith("grounding_results_")
            and file_name.endswith(".pkl")
        )
        if not matches:
            return None

        for preferred_name in (
            "grounding_results_gdino_base.pkl",
            "grounding_results_gdino_tiny.pkl",
        ):
            preferred_path = os.path.join(clip_dir, preferred_name)
            if preferred_path in matches:
                return preferred_path

        return matches[0]

    def _infer_gdino_feat_dim(self, grounding_results):
        for payload in grounding_results.values():
            for entry in payload.get("objects", {}).values():
                feats = entry.get("feats")
                if feats is not None:
                    return int(feats.shape[0])
            gaze_feats = payload.get("object_gazed_at", {}).get("feats")
            if gaze_feats is not None:
                return int(gaze_feats.shape[0])
        return self.default_gdino_feat_dim

    def _remap_grounding_payload(self, payload, gdino_feat_dim):
        remapped_objects = {}
        for split_obj_idx, entry in payload.get("objects", {}).items():
            global_obj_idx = self.split_to_global_obj_idx.get(split_obj_idx)
            if global_obj_idx is None:
                continue

            feats = entry.get("feats")
            if feats is None:
                feats = self._zero_gdino_feature(gdino_feat_dim)

            remapped_objects[global_obj_idx] = {
                "feats": feats,
                "phrase": entry.get("phrase"),
                "confidence": entry.get("confidence", 0.0),
            }

        gaze_entry = payload.get("object_gazed_at", {})
        gaze_feats = gaze_entry.get("feats")
        if gaze_feats is None:
            gaze_feats = self._zero_gdino_feature(gdino_feat_dim)
        gaze_idx = gaze_entry.get("idx")

        return {
            "objects": remapped_objects,
            "object_gazed_at": {
                "feats": gaze_feats,
                "phrase": gaze_entry.get("phrase"),
                "idx": self.split_to_global_obj_idx.get(gaze_idx),
            },
        }

    def _derive_parse_annotations(self, clip_dir):
        parse_annotations = pd.read_csv(os.path.join(clip_dir, "parse_annotation.csv"))
        annotations_path = os.path.join(clip_dir, "annotations_qwen3vl_32b_instruct.csv")
        annotations = pd.read_csv(
            annotations_path, usecols=["frame_index", "frame_file"]
        )
        annotations["frame_number"] = (
            annotations["frame_file"]
            .astype(str)
            .str.replace(".jpg", "", regex=False)
            .astype(int)
        )
        parse_annotations = parse_annotations.merge(
            annotations,
            left_on="frame_id",
            right_on="frame_index",
            how="inner",
        )

        return parse_annotations

    def _load_clip_resources(self, clip_dir):
        cached = self._clip_cache.get(clip_dir)
        if cached is not None:
            return cached

        parse_annotations = self._derive_parse_annotations(clip_dir)
        parse_annotations["frame_number"] = parse_annotations["frame_number"].astype(int)
        h5_path = os.path.join(clip_dir, self.rgb_feature_filename)
        h5_file = h5py.File(h5_path, "r")
        frame_num_to_h5_idx = {}

        for idx, frame_name in enumerate(h5_file["frame_names"]):
            if isinstance(frame_name, bytes):
                frame_name = frame_name.decode("utf-8")
            frame_num_to_h5_idx[int(os.path.splitext(frame_name)[0])] = idx

        grounding_results = None
        gdino_feat_dim = self.default_gdino_feat_dim
        grounding_path = self._resolve_grounding_path(clip_dir)
        if grounding_path is not None:
            with open(grounding_path, "rb") as f:
                grounding_results = pickle.load(f)
            gdino_feat_dim = self._infer_gdino_feat_dim(grounding_results)

        resources = {
            "parse_annotations": parse_annotations,
            "h5_file": h5_file,
            "frame_num_to_h5_idx": frame_num_to_h5_idx,
            "grounding_results": grounding_results,
            "gdino_feat_dim": gdino_feat_dim,
        }
        self._clip_cache[clip_dir] = resources
        return resources

    def _load_depth_features(self, clip_name):
        if not self.use_depth_features:
            return None
        cached = self._depth_feature_cache.get(clip_name)
        if cached is not None:
            return cached

        clip_id = os.path.basename(str(clip_name))
        depth_path = os.path.join(
            self.depth_feature_root,
            self.split_name,
            f"{clip_id}.npy",
        )
        depth_features = np.load(depth_path, mmap_mode="r")
        self._depth_feature_cache[clip_name] = depth_features
        return depth_features

    def _ohg_split_dir(self):
        if self.split_name == "Test":
            return "RULSTM_MECCANO_test_features"
        return "RULSTM_MECCANO_trainval_features"

    def _load_ohg_envs(self):
        if not self.use_ohg_features:
            return None
        if self.ohg_feature_root is None:
            raise ValueError("ohg_feature_root is required for OHG feature modes.")

        cache_key = (self.ohg_feature_root, self.split_name)
        cached = self._ohg_env_cache.get(cache_key)
        if cached is not None:
            return cached

        import lmdb

        base_dir = os.path.join(self.ohg_feature_root, self._ohg_split_dir())
        envs = []
        for modality in ("obj", "gaze", "hands"):
            lmdb_path = os.path.join(base_dir, modality)
            if not os.path.isdir(lmdb_path):
                raise FileNotFoundError(f"Missing OHG LMDB: {lmdb_path}")
            envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False))

        self._ohg_env_cache[cache_key] = envs
        return envs

    def _ohg_frame_keys(self, clip_name, frame_number):
        clip_str = str(clip_name)
        clip_candidates = [clip_str]
        try:
            clip_int = int(clip_str)
            clip_candidates.extend(
                [
                    f"{clip_int:02d}",
                    f"{clip_int:04d}",
                    str(clip_int),
                ]
            )
        except ValueError:
            pass

        frame_int = int(frame_number)
        frame_candidates = [
            f"{frame_int:05d}.jpg",
            f"{frame_int}.jpg",
        ]
        keys = []
        for clip_id in dict.fromkeys(clip_candidates):
            for frame_name in frame_candidates:
                keys.append(f"{clip_id}_{frame_name}".encode("utf-8"))
        return keys

    def _load_ohg_feature(self, clip_name, frame_number):
        envs = self._load_ohg_envs()
        if envs is None:
            return None

        parts = []
        keys = self._ohg_frame_keys(clip_name, frame_number)
        for env in envs:
            feature = None
            with env.begin() as txn:
                for key in keys:
                    value = txn.get(key)
                    if value is not None:
                        feature = np.frombuffer(value, dtype=np.float32)
                        break
            if feature is None:
                feature = np.zeros(self.ohg_feature_dim, dtype=np.float32)
            parts.append(np.asarray(feature, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def _build_frame_feature(self, rgb_feature, depth_features, ohg_feature, frame_number):
        if self.feature_mode == "rgb":
            return rgb_feature
        if self.feature_mode == "ohg":
            return ohg_feature
        if self.feature_mode == "rgb_ohg":
            return np.concatenate(
                [
                    np.asarray(rgb_feature, dtype=np.float32),
                    np.asarray(ohg_feature, dtype=np.float32),
                ],
                axis=0,
            )
        depth_feature = depth_features[int(frame_number) - 1]
        if self.feature_mode == "depth_only":
            return np.asarray(depth_feature, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(rgb_feature, dtype=np.float32),
                np.asarray(depth_feature, dtype=np.float32),
            ],
            axis=0,
        )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        try:
            clip_name, sample_id, label_str, clip_dir, frame_numbers = self.sample_index[idx]
            cache_path = self._resolve_cache_path(clip_name, sample_id, frame_numbers)
            if cache_path is not None and os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, map_location="cpu")
                except (EOFError, RuntimeError, OSError, pickle.UnpicklingError) as e:
                    print(
                        f"Ignoring unreadable EASG cache {cache_path}: "
                        f"{type(e).__name__}: {e}"
                    )

            resources = self._load_clip_resources(clip_dir)
            depth_features = self._load_depth_features(clip_name)
            ohg_envs = self._load_ohg_envs()
            parse_annotations = resources["parse_annotations"]
            h5_file = resources["h5_file"]
            frame_num_to_h5_idx = resources["frame_num_to_h5_idx"]
            grounding_results = resources["grounding_results"]
            gdino_feat_dim = resources["gdino_feat_dim"]

            frame_rows = []
            frame_feats = []
            obj_feats = []
            for frame_number in frame_numbers:
                frame_row = parse_annotations[
                    parse_annotations["frame_number"] == frame_number
                ]
                if frame_row.empty or frame_number not in frame_num_to_h5_idx:
                    raise KeyError(
                        f"Missing frame {frame_number} for clip {clip_name} in {clip_dir}"
                    )

                frame_rows.append(frame_row.iloc[0].copy())
                ohg_feature = (
                    self._load_ohg_feature(clip_name, frame_number)
                    if ohg_envs is not None
                    else None
                )
                frame_feats.append(
                    self._build_frame_feature(
                        h5_file["visual_features"][frame_num_to_h5_idx[frame_number]],
                        depth_features,
                        ohg_feature,
                        frame_number,
                    )
                )
                frame_file = str(frame_row.iloc[0]["frame_file"])
                if grounding_results is not None and frame_file in grounding_results:
                    obj_feats.append(
                        self._remap_grounding_payload(
                            grounding_results[frame_file], gdino_feat_dim
                        )
                    )
                else:
                    obj_feats.append(self._empty_grounding_payload(gdino_feat_dim))

            frame_parsed_anns = pd.DataFrame(frame_rows).reset_index(drop=True)
            frame_anns = pd.DataFrame(
                {"frame_index": frame_parsed_anns["frame_id"].astype(int).tolist()}
            )

            output = base_output(clip_name, sample_id, label_str, self.activity_to_idx)
            action_scene_graphs = {}
            vocab = (self.verbs, self.objs, self.rels, self.attrs)

            for i, frame_id in enumerate(frame_anns["frame_index"].tolist()):
                frame_parsed_ann = frame_parsed_anns[
                    frame_parsed_anns["frame_id"] == frame_id
                ]
                action_scene_graphs[i] = build_action_graph(
                    self.graph_type,
                    frame_parsed_ann.iloc[0],
                    vocab,
                    frame_feats[i],
                    obj_feats[i],
                    self.clip_textual_embeddings,
                )
                output["full_action_graphs"] = action_scene_graphs

            if cache_path is not None:
                return self._store_cached_sample(cache_path, output, action_scene_graphs)

            return output
        except Exception as e:
            print(e)
            print(f"ERROR loading sample {idx}: {type(e).__name__}: {e}")
            print(f"  clip: {clip_name if 'clip_name' in locals() else 'unknown'}")
            print(f"  label: {label_str if 'label_str' in locals() else 'unknown'}")
            raise e

def feature_collate_fn(batch):
    output = {}

    output["activity_label"] = torch.stack([item["activity_label"] for item in batch])
    output["activity_name"] = [item["activity_name"] for item in batch]
    output["clip_name"] = [item["clip_name"] for item in batch]
    output["block_idx"] = [item["block_idx"] for item in batch]
    output["full_action_graphs"] = [item["full_action_graphs"] for item in batch]

    return output

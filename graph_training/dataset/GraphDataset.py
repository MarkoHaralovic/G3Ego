import hashlib
import json
import os
import pickle
from ast import literal_eval

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset.meccano_aux import (
    get_split_name,
    resolve_meccano_global_root,
    resolve_meccano_split_root,
    load_clip_text_embeddings
)
from graph_construction.graphs.full_graph import FullActionGraph
from graph_construction.graphs.pruned_graph import PrunedActionGraph

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
            output = {"clip_name": self.clip_names[file_idx], "block_idx": block_idx}
            output["activity_label"] = torch.tensor(
                self.activity_to_idx[label_str], dtype=torch.long
            )
            output["activity_name"] = label_str

            action_scene_graphs = {}

            for i, frame_id in enumerate(frame_anns["frame_index"].tolist()):
                frame_parsed_ann = frame_parsed_anns[
                    frame_parsed_anns["frame_id"] == frame_id
                ]
                if self.graph_type == "full":
                    graph = FullActionGraph(
                        self.verbs, self.objs, self.rels, self.attrs
                    )
                elif self.graph_type == "pruned":
                    graph = PrunedActionGraph(
                        self.verbs, self.objs, self.rels, self.attrs
                    )

                verb = frame_parsed_ann["verb"].iloc[0]
                direct_object = frame_parsed_ann["direct_object"].iloc[0]
                gazed_at_object = (
                    frame_parsed_ann["gazed_at_object"].iloc[0]
                    if "gazed_at_object" in frame_parsed_ann.columns
                    else None
                )
                objects_atr_val = frame_parsed_ann["all_objects"].iloc[0]
                objects_atr_map = (
                    literal_eval(objects_atr_val)
                    if not pd.isna(objects_atr_val)
                    else {}
                )
                rels_val = frame_parsed_ann["preposition_object_pairs"].iloc[0]
                rels_dict = literal_eval(rels_val) if not pd.isna(rels_val) else []
                aux_verbs_str = frame_parsed_ann["aux_verbs"].iloc[0]
                aux_verbs = (
                    literal_eval(aux_verbs_str)
                    if aux_verbs_str
                    and aux_verbs_str != "[]"
                    and not pd.isna(aux_verbs_str)
                    else None
                )
                aux_obj_str = frame_parsed_ann["object_aux_verb"].iloc[0]
                aux_direct_objects_map = (
                    literal_eval(aux_obj_str)
                    if aux_obj_str and aux_obj_str != "{}" and not pd.isna(aux_obj_str)
                    else None
                )

                if self.graph_type == "full":
                    graph = graph.create_graph(
                        verb=verb,
                        direct_object=direct_object,
                        objects_atr_map=objects_atr_map,
                        clip_feat=frame_feats[i],
                        obj_feats=obj_feats[i],
                        rels_dict=rels_dict,
                        aux_verbs=aux_verbs,
                        aux_direct_objects_map=aux_direct_objects_map,
                        clip_embeddings=self.clip_textual_embeddings
                    )
                elif self.graph_type == "pruned":
                    graph = graph.create_graph(
                        verb=verb,
                        gazed_at_object=gazed_at_object,
                        direct_object=direct_object,
                        objects_atr_map=objects_atr_map,
                        clip_feat=frame_feats[i],
                        obj_feats=obj_feats[i],
                        rels_dict=rels_dict,
                        clip_embeddings=self.clip_textual_embeddings
                    )
                action_scene_graphs[i] = graph
                output["full_action_graphs"] = action_scene_graphs
            return output
        except Exception as e:
            print(e)
            print(f"ERROR loading sample {idx}: {type(e).__name__}: {e}")
            print(f"  clip: {self.clip_names[file_idx] if 'file_idx' in locals() else 'unknown'}")
            print(f"  label: {label_str if 'label_str' in locals() else 'unknown'}")
            raise e


class GraphDatasetMeccano(Dataset):
    def __init__(
        self,
        metadata_root,
        split_name,
        samples,
        activity_to_idx,
        graph_type,
        easg_cache_path=None,
    ):
        self.samples = samples
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
        self.graph_type = graph_type
        self.split_name = get_split_name(split_name)
        self.split_metadata_root = resolve_meccano_split_root(metadata_root, self.split_name)
        self.global_metadata_root = resolve_meccano_global_root(metadata_root)
        self.metadata_root = self.global_metadata_root or self.split_metadata_root
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
        self.default_gdino_feat_dim = 256
        self.easg_cache_path = easg_cache_path

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
            clip_name,
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir, f"sample_{sample_id:06d}_g{num_graphs}_{frame_hash}.pt"
        )

    def _tensorize_cached_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
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
        global_vocab = (
            self._load_vocab_group(self.global_metadata_root, prefix="global_")
            if self.global_metadata_root is not None
            else self._load_vocab_group(self.split_metadata_root)
        )
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

        if os.path.exists(annotations_path):
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
                how="left",
            )
        else:
            parse_annotations["frame_number"] = (
                parse_annotations["frame_id"].astype(int) * 3 + 1
            )
            parse_annotations["frame_file"] = parse_annotations["frame_number"].map(
                lambda x: f"{x:05d}.jpg"
            )

        return parse_annotations

    def _load_clip_resources(self, clip_dir):
        cached = self._clip_cache.get(clip_dir)
        if cached is not None:
            return cached

        parse_annotations = self._derive_parse_annotations(clip_dir)
        h5_path = os.path.join(clip_dir, "frame_features_model_dinov3h16+.h5")
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
                frame_feats.append(
                    h5_file["visual_features"][frame_num_to_h5_idx[frame_number]]
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

            output = {"clip_name": clip_name, "block_idx": sample_id}
            output["activity_label"] = torch.tensor(
                self.activity_to_idx[label_str], dtype=torch.long
            )
            output["activity_name"] = label_str

            action_scene_graphs = {}

            for i, frame_id in enumerate(frame_anns["frame_index"].tolist()):
                frame_parsed_ann = frame_parsed_anns[
                    frame_parsed_anns["frame_id"] == frame_id
                ]
                if self.graph_type == "full":
                    graph = FullActionGraph(
                        self.verbs, self.objs, self.rels, self.attrs
                    )
                elif self.graph_type == "pruned":
                    graph = PrunedActionGraph(
                        self.verbs, self.objs, self.rels, self.attrs
                    )
                else:
                    raise ValueError(f"Unsupported graph_type: {self.graph_type}")

                verb = frame_parsed_ann["verb"].iloc[0]
                direct_object = frame_parsed_ann["direct_object"].iloc[0]
                gazed_at_object = (
                    frame_parsed_ann["gazed_at_object"].iloc[0]
                    if "gazed_at_object" in frame_parsed_ann.columns
                    else None
                )
                objects_atr_val = frame_parsed_ann["all_objects"].iloc[0]
                objects_atr_map = (
                    literal_eval(objects_atr_val)
                    if not pd.isna(objects_atr_val)
                    else {}
                )
                rels_val = frame_parsed_ann["preposition_object_pairs"].iloc[0]
                rels_dict = literal_eval(rels_val) if not pd.isna(rels_val) else []
                aux_verbs_str = frame_parsed_ann["aux_verbs"].iloc[0]
                aux_verbs = (
                    literal_eval(aux_verbs_str)
                    if aux_verbs_str
                    and aux_verbs_str != "[]"
                    and not pd.isna(aux_verbs_str)
                    else None
                )
                aux_obj_str = frame_parsed_ann["object_aux_verb"].iloc[0]
                aux_direct_objects_map = (
                    literal_eval(aux_obj_str)
                    if aux_obj_str and aux_obj_str != "{}" and not pd.isna(aux_obj_str)
                    else None
                )

                if self.graph_type == "full":
                    graph = graph.create_graph(
                        verb=verb,
                        direct_object=direct_object,
                        objects_atr_map=objects_atr_map,
                        clip_feat=frame_feats[i],
                        obj_feats=obj_feats[i],
                        rels_dict=rels_dict,
                        aux_verbs=aux_verbs,
                        aux_direct_objects_map=aux_direct_objects_map,
                        clip_embeddings=self.clip_textual_embeddings,
                    )
                else:
                    graph = graph.create_graph(
                        verb=verb,
                        gazed_at_object=gazed_at_object,
                        direct_object=direct_object,
                        objects_atr_map=objects_atr_map,
                        clip_feat=frame_feats[i],
                        obj_feats=obj_feats[i],
                        rels_dict=rels_dict,
                        clip_embeddings=self.clip_textual_embeddings,
                    )

                action_scene_graphs[i] = graph
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

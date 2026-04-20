import json
import os
import pickle
from ast import literal_eval

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from global_feature_training.data_loading.dataset_split import (
        decode_label,
        map_or_skip_label,
        stratified_split,
    )
except ImportError:
    from global_feature_training.data_loading.dataset_split_aria import (
        decode_label,
        map_or_skip_label,
        stratified_split,
    )
from graph_construction.graphs.full_graph import FullActionGraph
from graph_construction.graphs.pruned_graph import PrunedActionGraph


class ZeroTextEmbeddings(dict):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.emb_dim = emb_dim

    def __missing__(self, key):
        return torch.zeros(self.emb_dim, dtype=torch.float32)

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
        if os.path.exists(clip_text_path):
            self.clip_textual_embeddings = pickle.load(open(clip_text_path, "rb"))
        else:
            self.clip_textual_embeddings = ZeroTextEmbeddings()
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
    def __init__(self, input_path, samples, activity_to_idx, graph_type):
        self.input_path = input_path
        self.samples = samples
        self.activity_to_idx = activity_to_idx
        self.idx_to_activity = {v: k for k, v in self.activity_to_idx.items()}
        self.graph_type = graph_type
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

        self.metadata_root = self._resolve_metadata_root(input_path)
        with open(os.path.join(self.metadata_root, "verbs.json"), "r") as f:
            self.verbs = json.load(f)
        with open(os.path.join(self.metadata_root, "objects.json"), "r") as f:
            self.objs = json.load(f)
        with open(os.path.join(self.metadata_root, "relationships.json"), "r") as f:
            self.rels = json.load(f)
        with open(os.path.join(self.metadata_root, "attributes.json"), "r") as f:
            self.attrs = json.load(f)

        clip_text_path = os.path.join(self.metadata_root, "clip_text_features.pkl")
        if os.path.exists(clip_text_path):
            self.clip_textual_embeddings = pickle.load(open(clip_text_path, "rb"))
        else:
            self.clip_textual_embeddings = ZeroTextEmbeddings()

    def _resolve_metadata_root(self, input_path):
        if os.path.exists(os.path.join(input_path, "verbs.json")):
            return input_path

        for split_name in ("Train", "Val", "Test"):
            split_root = os.path.join(input_path, split_name)
            if os.path.exists(os.path.join(split_root, "verbs.json")):
                return split_root

        raise FileNotFoundError(
            f"Could not find verbs.json/objects.json metadata under {input_path}"
        )

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

        resources = {
            "parse_annotations": parse_annotations,
            "h5_file": h5_file,
            "frame_num_to_h5_idx": frame_num_to_h5_idx,
        }
        self._clip_cache[clip_dir] = resources
        return resources

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        try:
            clip_name, sample_id, label_str, clip_dir, frame_numbers = self.sample_index[idx]
            resources = self._load_clip_resources(clip_dir)
            parse_annotations = resources["parse_annotations"]
            h5_file = resources["h5_file"]
            frame_num_to_h5_idx = resources["frame_num_to_h5_idx"]

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
                obj_feats.append({"objects": {}, "object_gazed_at": {}})

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

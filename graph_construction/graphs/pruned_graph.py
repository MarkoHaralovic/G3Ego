from typing import Dict, List

import spacy
import torch

from .base_graph import Edge, Node
from .full_graph import FullActionGraph

nlp = spacy.load("en_core_web_sm")


def to_singular(word: str) -> str:
    if len(word.split(" ")) > 1:
        return word
    doc = nlp(word)
    return_class = []
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
            return_class.append(tok.lemma_.lower())
    if return_class:
        return " ".join(c for c in return_class)
    return word.lower()


class PrunedActionGraph(FullActionGraph):
    def __init__(self, verbs, objs, rels, attrs):
        node_types = ["camera_wearer", "verb", "object"]
        rels = rels.copy()
        if "aux_direct_object" in rels:
            del rels["aux_direct_object"]
        if "aux_verb" in rels:
            del rels["aux_verb"]
        if "gazed_at" not in rels:
            rels["gazed_at"] = max(rels.values(), default=-1) + 1
        super(FullActionGraph, self).__init__(node_types, verbs, objs, rels, attrs)
        self.num_rels = max(self.rels.values(), default=-1) + 1
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []

    def create_graph(
        self,
        verb,
        gazed_at_object,
        direct_object,
        objects_atr_map,
        clip_feat,
        obj_feats,
        rels_dict,
        clip_embeddings,
        object_dim=256,
    ):
        self.object_dim = object_dim
        if gazed_at_object is None:
            gazed_at_object = direct_object

        obj_data = []
        for obj_name, obj_info in objects_atr_map.items():
            if obj_name != gazed_at_object and obj_name != direct_object:
                continue
            if to_singular(obj_info["base_object"]) in self.objs.keys():
                obj_idx = self.objs[to_singular(obj_info["base_object"])]
                obj_data.append((obj_idx, obj_name, obj_info["attributes"]))

        rels_vecs = torch.zeros(len(self.objs), self.num_rels)
        for _item in rels_dict:
            obj_name, rel = list(_item.items())[0]
            if obj_name != gazed_at_object and obj_name != direct_object:
                continue
            if (
                rel not in self.rels
                or to_singular(objects_atr_map.get(obj_name, {}).get("base_object", ""))
                not in self.objs
            ):
                continue
            rels_vecs[
                self.objs[to_singular(objects_atr_map[obj_name]["base_object"])],
                self.rels[rel],
            ] = 1.0
        node_id_counter = 0

        # CW node
        cw_id = node_id_counter
        node_id_counter += 1
        cw_node = self.new_camera_wearer_node(
            node_id=node_id_counter, 
            text_embedding=clip_embeddings["camera_wearer"]
        )
        self.nodes[cw_id] = cw_node

        # Verb node
        verb_embedding = clip_embeddings[verb]
        verb_id = node_id_counter
        node_id_counter += 1
        verb_node = self.new_main_verb_node(
            node_id=verb_id, 
            verb=verb, 
            clip_feat=clip_feat, 
            text_embedding=verb_embedding
        )
        self.nodes[verb_id] = verb_node

        # CW -> verb edge
        self.edges.append(self.agent_edge(cw_id, verb_id))

        # object nodes and edges
        obj_id_map = {}
        attr_vecs = torch.zeros((len(self.objs), len(self.attrs)))

        for obj_idx, orig_name, attributes in obj_data:
            if obj_idx in obj_feats.get("objects", {}):
                obj_feat = obj_feats["objects"][obj_idx]["feats"]
            else:
                if obj_feats.get("objects") and len(obj_feats["objects"]) > 0:
                    first_key = next(iter(obj_feats["objects"]))
                    feat_shape = obj_feats["objects"][first_key]["feats"].shape
                else:
                    feat_shape = torch.Size([self.object_dim])

                obj_feat = torch.zeros(feat_shape)

            node_id = node_id_counter
            node_id_counter += 1

            for attr in attributes:
                attr_vecs[obj_idx, self.attrs[attr]] = 1.0

            text = " ".join(attributes + [objects_atr_map[orig_name]["base_object"]])
            obj_text_embedding = clip_embeddings[text]
            obj_node = self.new_object_node(
                node_id=node_id,
                obj_idx=obj_idx,
                obj_feat=obj_feat,
                attr=attributes,
                attr_feat=attr_vecs[obj_idx],
                text_embedding=obj_text_embedding,
                verb_idx=self.verbs[verb],
            )
            self.nodes[node_id] = obj_node
            obj_id_map[obj_idx] = node_id

            if orig_name == direct_object and "direct_object" in self.rels:
                rels_vecs[obj_idx, self.rels["direct_object"]] = 1.0
                self.edges.append(
                    self.rel_edge(verb_id, node_id, self.rels["direct_object"])
                )

            if orig_name == gazed_at_object and "gazed_at" in self.rels:
                rels_vecs[obj_idx, self.rels["gazed_at"]] = 1.0
                self.edges.append(
                    self.rel_edge(verb_id, node_id, self.rels["gazed_at"])
                )

            rels_vec = rels_vecs[obj_idx]
            for rel_idx in torch.where(rels_vec > 0)[0].tolist():
                if rel_idx == self.rels.get(
                    "direct_object"
                ) or rel_idx == self.rels.get("gazed_at"):
                    continue
                self.edges.append(self.rel_edge(verb_id, node_id, rel_idx))

        self.rels_vecs = rels_vecs
        self.attr_vecs = attr_vecs

        return self

    def to_easg_tensors(self) -> Dict[str, torch.Tensor]:

        verb_node = [n for n in self.nodes.values() if n.node_type == "verb"]
        assert len(verb_node) == 1
        verb_node = verb_node[0]
        verb_idx_tensor = torch.tensor([verb_node.idx], dtype=torch.long)
        clip_feat_tensor = verb_node.feat

        obj_nodes = [n for n in self.nodes.values() if n.node_type == "object"]

        if len(obj_nodes) == 0:
            obj_indices = torch.zeros(0, dtype=torch.long)
            obj_feats = torch.zeros(0, self.object_dim)
            obj_text_embs = torch.zeros(0, 1) 
        else:
            obj_indices = torch.tensor([n.idx for n in obj_nodes], dtype=torch.long)
            obj_feats = torch.stack([n.feat for n in obj_nodes], dim=0)
            obj_text_embs = torch.stack([n.text_embedding for n in obj_nodes], dim=0)

        all_nodes_sorted = [self.nodes[k] for k in sorted(self.nodes.keys())]
        node_text_embs = torch.stack([
            n.text_embedding if n.text_embedding is not None else torch.zeros_like(obj_text_embs[0])
            for n in all_nodes_sorted
        ], dim=0) if len(obj_nodes) > 0 else torch.zeros(0, 1)
        
        num_rels = self.num_rels
        rels_vecs = torch.zeros((len(obj_nodes), num_rels), dtype=torch.float32)

        
        num_attrs = len(self.attrs)
        attr_vecs = torch.zeros((len(obj_nodes), num_attrs), dtype=torch.float32)

        for e in self.edges:
            if e.rel_idx is None:
                continue
            obj_node = self.nodes[e.dst]
            obj_row = obj_nodes.index(obj_node)
            rels_vecs[obj_row, e.rel_idx] = 1.0

        triplets = []

        for obj_node in obj_nodes:
            obj_row = obj_nodes.index(obj_node)
            if obj_node.attr:
                attrs_list = (
                    obj_node.attr
                    if isinstance(obj_node.attr, list)
                    else [obj_node.attr]
                )
                for attr in attrs_list:
                    if attr in self.attrs:
                        attr_vecs[obj_row, self.attrs[attr]] = 1.0
            for rel_idx in torch.where(rels_vecs[obj_row] > 0)[0]:
                verb_for_triplet = (
                    obj_node.related_to_verb
                    if obj_node.related_to_verb is not None
                    else verb_node.idx
                )
                triplets.append((verb_for_triplet, obj_node.idx, rel_idx.item()))
        triplets = (
            torch.tensor(triplets, dtype=torch.long)
            if len(triplets) > 0
            else torch.zeros((0, 3), dtype=torch.long)
        )

        return {
            "verb_idx": verb_idx_tensor,
            "aux_verb_idx": torch.zeros_like(verb_idx_tensor),
            "clip_feat": clip_feat_tensor,
            "obj_indices": obj_indices,
            "attr_vecs": attr_vecs,
            "obj_feats": obj_feats,
            "rels_vecs": rels_vecs,
            "triplets": triplets,
            "node_text_embs": node_text_embs,
        }

from typing import Dict, List
import spacy
import torch
from .base_graph import BaseGraph, Edge, Node
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


def normalize_object_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def resolve_object_name(name: str, objects_atr_map: Dict) -> str:
    if name in objects_atr_map:
        return name

    normalized_name = normalize_object_name(name)
    normalized_lookup = {
        normalize_object_name(candidate): candidate for candidate in objects_atr_map
    }
    return normalized_lookup.get(normalized_name)

class FullActionGraph(BaseGraph):
    def __init__(self, verbs: Dict, objs: Dict, rels: Dict, attrs: Dict):
        node_types = ["camera_wearer", "verb", "object", "aux_verb"]
        super(FullActionGraph, self).__init__(node_types, verbs, objs, rels, attrs)
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []

    def create_graph(
        self,
        verb,
        direct_object,
        objects_atr_map,
        clip_feat,
        obj_feats,
        rels_dict,
        clip_embeddings,
        aux_verbs=None,
        aux_direct_objects_map=None,
        object_dim=256,
        
    ):
        self.object_dim = object_dim
        original_direct_object = direct_object
        direct_object = (
            resolve_object_name(direct_object, objects_atr_map)
            if direct_object
            else None
        )
        if original_direct_object and direct_object is None:
            print(
                f"direct_object '{original_direct_object}' not found in "
                f"objects_atr_map. Available: {list(objects_atr_map.keys())}"
            )
            direct_object = None

        aux_verb_idxs = (
            [self.verbs[aux_verb] for aux_verb in aux_verbs]
            if aux_verbs is not None
            else None
        )

        obj_data = []
        for obj_name, obj_info in objects_atr_map.items():
            if to_singular(obj_info["base_object"]) in self.objs.keys():
                obj_idx = self.objs[to_singular(obj_info["base_object"])]
                obj_data.append((obj_idx, obj_name, obj_info["attributes"]))

        obj_row_map = {
            obj_name: row_idx for row_idx, (_, obj_name, _) in enumerate(obj_data)
        }

        rels_vecs = torch.zeros(len(obj_data), len(self.rels))
        for _item in rels_dict:
            obj_name, rel = list(_item.items())[0]
            resolved_obj_name = resolve_object_name(obj_name, objects_atr_map)
            if (
                rel not in self.rels
                or resolved_obj_name not in obj_row_map
            ):
                continue
            rels_vecs[obj_row_map[resolved_obj_name], self.rels[rel]] = 1.0
        node_id_counter = 0

        aux_obj_to_verbs = {}
        if aux_direct_objects_map:
            for aux_verb, aux_objects in aux_direct_objects_map.items():
                if aux_verb not in self.verbs:
                    continue
                for aux_object in aux_objects:
                    resolved_aux_object = resolve_object_name(
                        aux_object, objects_atr_map
                    )
                    if resolved_aux_object not in obj_row_map:
                        continue
                    aux_obj_to_verbs.setdefault(resolved_aux_object, []).append(aux_verb)
                    if "aux_direct_object" in self.rels:
                        rels_vecs[
                            obj_row_map[resolved_aux_object], self.rels["aux_direct_object"]
                        ] = 1.0

        # CW node
        cw_id = node_id_counter
        node_id_counter += 1
        cw_node = self.new_camera_wearer_node(
            node_id=node_id_counter, 
            text_embedding=clip_embeddings["camera_wearer"])
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

        aux_verb_node_map = {}
        if aux_verb_idxs:
            for aux_verb in aux_verbs:
                aux_verb_id = node_id_counter
                aux_verb_embedding = clip_embeddings[aux_verb]
                node_id_counter += 1
                aux_verb_node = self.new_aux_verb_node(
                    node_id=aux_verb_id, 
                    verb=aux_verb, 
                    clip_feat=None, 
                    text_embedding=aux_verb_embedding
                )
                self.nodes[aux_verb_id] = aux_verb_node
                aux_verb_node_map[aux_verb] = aux_verb_id

        # object nodes + edges
        obj_name_to_node_id: Dict[str, int] = {}
        attr_vecs = torch.zeros((len(obj_data), len(self.attrs)))
        for obj_row, (obj_idx, orig_name, attributes) in enumerate(obj_data):
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
                if attr in self.attrs:
                    attr_vecs[obj_row, self.attrs[attr]] = 1.0

            if orig_name not in aux_obj_to_verbs:
                text = " ".join(attributes + [objects_atr_map[orig_name]["base_object"]])
                obj_text_embedding = clip_embeddings[text]
                obj_node = self.new_object_node(
                    node_id=node_id,
                    obj_idx=obj_idx,
                    obj_feat=obj_feat,
                    attr=attributes,
                    attr_feat=attr_vecs[obj_row],
                    text_embedding=obj_text_embedding,
                    verb_idx=self.verbs[verb],
                )
            else:
                verb_related_to = aux_obj_to_verbs[orig_name][0]
                text = " ".join(attributes + [objects_atr_map[orig_name]["base_object"]])
                obj_text_embedding = clip_embeddings[text]
                obj_node = self.new_object_node(
                    node_id=node_id,
                    obj_idx=obj_idx,
                    obj_feat=obj_feat,
                    attr=attributes,
                    attr_feat=attr_vecs[obj_row],
                    text_embedding=obj_text_embedding,
                    verb_idx=self.verbs[verb_related_to],
                )

            self.nodes[node_id] = obj_node
            obj_name_to_node_id[orig_name] = node_id

            if direct_object is not None and orig_name == direct_object:
                self.edges.append(
                    self.rel_edge(verb_id, node_id, rel_idx=self.rels["direct_object"])
                )

            rels_vec = rels_vecs[obj_row]
            for rel_idx in torch.where(rels_vec > 0)[0].tolist():
                if rel_idx == self.rels["direct_object"]:
                    continue
                self.edges.append(self.rel_edge(verb_id, node_id, rel_idx))

        if aux_verb_idxs and aux_direct_objects_map:
            for aux_verb, aux_verb_node_id in aux_verb_node_map.items():
                for aux_obj in aux_direct_objects_map.get(aux_verb, []):
                    resolved_aux_obj = resolve_object_name(aux_obj, objects_atr_map)
                    if resolved_aux_obj in obj_name_to_node_id:
                        self.edges.append(
                            self.rel_edge(
                                aux_verb_node_id,
                                obj_name_to_node_id[resolved_aux_obj],
                                rel_idx=self.rels["aux_direct_object"],
                            )
                        )

        self.rels_vecs = rels_vecs
        self.attr_vecs = attr_vecs

        return self

    def to_easg_tensors(self) -> Dict[str, torch.Tensor]:

        verb_node = [n for n in self.nodes.values() if n.node_type == "verb"]
        assert len(verb_node) == 1
        verb_node = verb_node[0]
        verb_idx_tensor = torch.tensor([verb_node.idx], dtype=torch.long)
        clip_feat_tensor = verb_node.feat

        aux_verb_nodes = [n for n in self.nodes.values() if n.node_type == "aux_verb"]
        aux_verb_tensors = [
            torch.tensor([aux_verb_node.idx], dtype=torch.long)
            for aux_verb_node in aux_verb_nodes
        ]
        aux_verb_tensors = torch.tensor(aux_verb_tensors)

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

        num_rels = len(self.rels)
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

        if "aux_verb" in self.rels:
            for aux_verb in aux_verb_nodes:
                triplets.append((verb_node.idx, aux_verb.idx, self.rels["aux_verb"]))

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
            "aux_verb_idx": aux_verb_tensors,
            "clip_feat": clip_feat_tensor,
            "obj_indices": obj_indices,
            "attr_vecs": attr_vecs,
            "obj_feats": obj_feats,
            "rels_vecs": rels_vecs,
            "triplets": triplets,
            "node_text_embs": node_text_embs,
        }

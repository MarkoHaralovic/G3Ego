from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import init


class AttentionPooling(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.score = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.Tanh(),
            nn.Linear(self.hidden_features, 1),
        )

        self.proj = nn.Linear(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor):
        if x is None or x.shape[0] == 0:
            return torch.zeros(self.out_features)
        scores = self.score(x)
        w = torch.softmax(scores, dim=0)
        pooled = (w * x).sum(dim=0)
        return self.proj(pooled)


class MultiQueryPooling(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128, k=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.k = k

        self.queries = nn.Parameter(torch.randn(k, self.in_features))
        self.proj = nn.Linear(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.zeros(
                self.k * self.out_features, device=x.device, dtype=x.dtype
            )
        if x.ndim == 1:
            x = x.unsqueeze(0)
        scores = (self.queries @ x.mT) / (self.in_features**0.5)
        attn = torch.softmax(scores, dim=-1)
        pooled = attn @ x
        projs = self.proj(pooled)
        return projs.reshape(-1)


class Embedder(nn.Module):
    def __init__(self, num_instances: int, emb_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(num_instances, emb_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long()
        if idx.ndim == 1 and idx.numel() == 1:
            idx = idx.squeeze(0)
        elif idx.ndim == 2 and idx.size(-1) == 1:
            idx = idx.squeeze(-1)
        return self.emb(idx)


class ActionGraphEmbedding(nn.Module):
    def __init__(
        self,
        num_verbs,
        num_objects,
        num_rels,
        num_attrs,
        clip_dim,
        clip_emb_dim,
        obj_feat_dim,
        clip_text_emb_out_feats=64,
        emb_dim=32,
        obj_out=128,
        aux_out=32,
        attr_out=32,
        rel_out=32,
        trip_out=32,
        k_obj=4,
        k_aux=2,
        k_trip=2,
        use_triplets=True,
        use_clip_text_emb=False,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.use_triplets = use_triplets

        self.verb_emb = Embedder(num_verbs, emb_dim)
        self.obj_emb = Embedder(num_objects, emb_dim)
        self.rel_emb = Embedder(num_rels, emb_dim)

        self.use_clip_text_emb = use_clip_text_emb
        if self.use_clip_text_emb:
            self.clip_text_embedding_pooling = AttentionPooling(in_features=512, out_features=clip_text_emb_out_feats, hidden_features=128)
        
        self.obj_pool = MultiQueryPooling(obj_feat_dim + emb_dim, obj_out, k=k_obj)
        if aux_out:
            self.aux_pool = MultiQueryPooling(emb_dim, aux_out, k=k_aux)
        else:
            self.aux_pool = None
        self.attr_proj = nn.Linear(num_attrs * 2, attr_out)
        self.rel_proj = nn.Linear(num_rels * 2, rel_out)

        if self.use_triplets:
            self.trip_mlp = nn.Sequential(
                nn.Linear(emb_dim * 3, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.trip_pool = MultiQueryPooling(emb_dim, trip_out, k=k_trip)

        self.clip_proj = nn.Linear(clip_dim, clip_emb_dim)

        self.out_dim = (
            clip_emb_dim
            + emb_dim
            + k_aux * aux_out
            + k_obj * obj_out
            + attr_out
            + rel_out
            + (k_trip * trip_out if self.use_triplets else 0)
            + (clip_text_emb_out_feats if self.use_clip_text_emb else 0)
        )

    def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        clip_feat = g["clip_feat"]
        clip = torch.as_tensor(clip_feat, device=self.device).float()

        v = self.verb_emb(g["verb_idx"].to(self.device))
        aux_idx = g.get("aux_verb_idx", None)
        if aux_idx is None or aux_idx.numel() == 0:
            aux_tokens = torch.zeros(
                (0, self.verb_emb.emb.embedding_dim), device=self.device
            )
        else:
            aux_tokens = self.verb_emb(aux_idx.to(self.device))
            if aux_tokens.ndim == 1:
                aux_tokens = aux_tokens.unsqueeze(0)
        aux_vec = self.aux_pool(aux_tokens) if self.aux_pool else None

        obj_feats = g["obj_feats"].to(self.device)
        if obj_feats.shape[0] == 0:
            obj_tokens = torch.zeros(
                (0, obj_feats.shape[1] + self.obj_emb.emb.embedding_dim),
                device=self.device,
                dtype=obj_feats.dtype,
            )
        else:
            obj_ids = self.obj_emb(g["obj_indices"].to(self.device))
            if obj_ids.ndim == 1:
                obj_ids = obj_ids.unsqueeze(0)
            obj_tokens = torch.cat([obj_feats, obj_ids.to(obj_feats.dtype)], dim=-1)
        obj_vec = self.obj_pool(obj_tokens)

        if self.use_clip_text_emb:
            text_embds = g["node_text_embs"].to(self.device)
            text_embedding_vec = self.clip_text_embedding_pooling(text_embds)
            
        attr_vecs = g["attr_vecs"].to(self.device)
        attr_sum = (
            attr_vecs.sum(dim=0)
            if attr_vecs.shape[0] > 0
            else torch.zeros(attr_vecs.shape[1], device=self.device)
        )
        attr_emb = self.attr_proj(torch.cat([attr_sum, torch.log1p(attr_sum)], dim=0))

        rels_vecs = g["rels_vecs"].to(self.device)
        rel_sum = (
            rels_vecs.sum(dim=0)
            if rels_vecs.shape[0] > 0
            else torch.zeros(rels_vecs.shape[1], device=self.device)
        )
        rel_emb = self.rel_proj(torch.cat([rel_sum, torch.log1p(rel_sum)], dim=0))

        if self.use_triplets:
            trip = g["triplets"].to(self.device)
            if trip.shape[0] == 0:
                trip_tokens = torch.zeros(
                    (0, self.verb_emb.emb.embedding_dim), device=self.device
                )
            else:
                t = torch.cat(
                    [
                        self.verb_emb(trip[:, 0]),
                        self.obj_emb(trip[:, 1]),
                        self.rel_emb(trip[:, 2]),
                    ],
                    dim=-1,
                )

                trip_tokens = self.trip_mlp(t)
            trip_vec = self.trip_pool(trip_tokens)

        parts = [
            self.clip_proj(clip),
            v.to(clip.dtype),
            obj_vec.to(clip.dtype),
            attr_emb.to(clip.dtype),
            rel_emb.to(clip.dtype),
        ]
        
        if self.use_clip_text_emb:
            parts.append(text_embedding_vec.to(clip.dtype))

        if aux_vec is not None:
            parts.append(aux_vec.to(clip.dtype))
        if self.use_triplets:
            parts.append(trip_vec.to(clip.dtype))

        return torch.cat(parts, dim=-1)


class GraphMLP(nn.Module):
    def __init__(
        self,
        num_graphs,
        num_verbs,
        num_objects,
        num_rels,
        num_attrs,
        n_classes,
        fc_layers_num,
        graph_emb_dim,
        final_graph_emb_dim,
        graph_pool_interim_feat,
        layer_norm,
        gelu,
        action_graph_kwargs,
        device="cuda",
        use_pool=True,
        use_proj=True,
        head_dropout=0.5,
        head_activation="gelu",
    ):
        super().__init__()
        self.device = device
        self.num_graphs = num_graphs

        self.action_graph_embedder = ActionGraphEmbedding(
            num_verbs=num_verbs,
            num_objects=num_objects,
            num_rels=num_rels,
            num_attrs=num_attrs,
            device=device,
            **action_graph_kwargs,
        )

        self.input_dim = self.action_graph_embedder.out_dim
        self.fc_layers_num = fc_layers_num
        self.n_classes = n_classes
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation).lower()

        self.pool = use_pool
        self.proj = use_proj

        self.graph_emb_dim = (
            graph_emb_dim if use_proj else self.action_graph_embedder.out_dim
        )

        if self.proj:
            self.graph_proj = [
                nn.Linear(self.input_dim, self.graph_emb_dim),
            ]
            if layer_norm:
                self.graph_proj.append(nn.LayerNorm(self.graph_emb_dim))
            if gelu:
                self.graph_proj.append(nn.GELU())
            else:
                self.graph_proj.append(nn.ReLU())

            self.graph_proj = nn.Sequential(*self.graph_proj)

        if self.pool:
            self.final_graph_emb_dim = final_graph_emb_dim
            self.graph_pool_interim_feat = graph_pool_interim_feat
            self.graph_pool = AttentionPooling(
                self.graph_emb_dim,
                self.final_graph_emb_dim,
                hidden_features=self.graph_pool_interim_feat,
            )
        else:
            self.final_graph_emb_dim = self.graph_emb_dim

        if self.fc_layers_num == 1:
            fc = nn.Linear(self.final_graph_emb_dim, self.n_classes)
            init.xavier_uniform_(fc.weight)
            init.zeros_(fc.bias)
        else:
            layers = []
            for _ in range(self.fc_layers_num - 1):
                layers.append(
                    nn.Linear(self.final_graph_emb_dim, self.final_graph_emb_dim)
                )
                if self.head_dropout > 0:
                    layers.append(nn.Dropout(self.head_dropout))
                if self.head_activation == "gelu":
                    layers.append(nn.GELU())
                elif self.head_activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(
                        f"Unsupported head activation: {self.head_activation}"
                    )
            layers.append(nn.Linear(self.final_graph_emb_dim, self.n_classes))
            fc = nn.Sequential(*layers)

            for layer in fc:
                if isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    init.zeros_(layer.bias)

        self.head = fc

    def _graph_to_tensors(self, graph_or_tensors):
        if isinstance(graph_or_tensors, dict):
            return graph_or_tensors
        return graph_or_tensors.to_easg_tensors()

    def forward(self, sequence_graphs: List):
        output = []
        for _sequence_graphs in sequence_graphs:
            graph_embs = [
                self.action_graph_embedder(self._graph_to_tensors(graph)).to(self.device)
                for graph in _sequence_graphs.values()
            ]
            if len(_sequence_graphs) < self.num_graphs:
                for _ in range(self.num_graphs - len(_sequence_graphs)):
                    graph_embs.append(graph_embs[-1])
            graph_embs = torch.stack(graph_embs, dim=0)

            if self.proj:
                graph_embs = self.graph_proj(graph_embs)

            if self.pool:
                graph_embs = self.graph_pool(graph_embs)
            else:
                graph_embs = graph_embs.mean(dim=0)

            out = self.head(graph_embs)

            output.append(out)
        return torch.stack(output, dim=0)

    def get_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f" Total network parameters : {total_params}, trainable parameters : {trainable_params}, pct trained {100*(trainable_params/total_params)}:.2f"
        )
        return trainable_params

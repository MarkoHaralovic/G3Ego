from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


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
        if x is None:
            return torch.zeros(self.out_features)
        if x.shape[0] == 0:
            return torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        scores = self.score(x)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights * x).sum(dim=0)
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
            self.clip_text_embedding_pooling = AttentionPooling(
                in_features=512,
                out_features=clip_text_emb_out_feats,
                hidden_features=128,
            )

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

        v = self.verb_emb(g["verb_idx"].to(self.device).long())
        aux_idx = g.get("aux_verb_idx", None)
        if aux_idx is None or aux_idx.numel() == 0:
            aux_tokens = torch.zeros(
                (0, self.verb_emb.emb.embedding_dim), device=self.device
            )
        else:
            aux_tokens = self.verb_emb(aux_idx.to(self.device).long())
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
            obj_ids = self.obj_emb(g["obj_indices"].to(self.device).long())
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
            trip = g["triplets"].long()
            if trip.shape[0] == 0:
                trip_tokens = torch.zeros(
                    (0, self.verb_emb.emb.embedding_dim), device=self.device
                )
            else:
                trip = trip.to(self.device)
                trip_tokens = self.trip_mlp(
                    torch.cat(
                        [
                            self.verb_emb(trip[:, 0]),
                            self.obj_emb(trip[:, 1]),
                            self.rel_emb(trip[:, 2]),
                        ],
                        dim=-1,
                    )
                )
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


class GraphLSTM(nn.Module):
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
        head_dropout=0.2,
        head_activation="gelu",
        hidden_size=256,
        num_layers=2,
        bias=True,
        bidirectional=False,
        recurrent_dropout=0.0,
        temporal_readout=None,
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

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.recurrent_dropout = (
            float(recurrent_dropout) if self.num_layers > 1 else 0.0
        )
        self.lstm = nn.LSTM(
            input_size=self.graph_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=bool(bias),
            dropout=self.recurrent_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.temporal_out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.temporal_readout = (
            str(temporal_readout).lower()
            if temporal_readout is not None
            else ("attention" if self.pool else "last_hidden")
        )

        if self.temporal_readout == "attention":
            self.final_graph_emb_dim = final_graph_emb_dim
            self.graph_pool_interim_feat = graph_pool_interim_feat
            self.temporal_pool = AttentionPooling(
                self.temporal_out_dim,
                self.final_graph_emb_dim,
                hidden_features=self.graph_pool_interim_feat,
            )
            self.temporal_readout_proj = nn.Identity()
        elif self.temporal_readout == "last_hidden":
            self.final_graph_emb_dim = final_graph_emb_dim
            if self.final_graph_emb_dim == self.temporal_out_dim:
                self.temporal_readout_proj = nn.Identity()
            else:
                self.temporal_readout_proj = nn.Linear(
                    self.temporal_out_dim, self.final_graph_emb_dim
                )
                init.xavier_uniform_(self.temporal_readout_proj.weight)
                init.zeros_(self.temporal_readout_proj.bias)
        else:
            raise ValueError(f"Unsupported temporal readout: {self.temporal_readout}")

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

    def _build_sequence_tensor(self, sequence_graphs: Dict[int, Dict[str, torch.Tensor]]):
        graph_embs = [
            self.action_graph_embedder(self._graph_to_tensors(graph)).to(self.device)
            for graph in sequence_graphs.values()
        ]
        if not graph_embs:
            raise ValueError("Received an empty graph sequence.")
        graph_embs = torch.stack(graph_embs, dim=0)
        if self.proj:
            graph_embs = self.graph_proj(graph_embs)
        return graph_embs

    def _run_lstm(self, sequence_graphs: List):
        batch_sequences = [
            self._build_sequence_tensor(sequence_graphs_sample)
            for sequence_graphs_sample in sequence_graphs
        ]
        lengths = torch.tensor(
            [seq.shape[0] for seq in batch_sequences], device=self.device
        )
        padded_sequences = pad_sequence(batch_sequences, batch_first=True)
        packed_sequences = pack_padded_sequence(
            padded_sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, (hidden_state, _) = self.lstm(packed_sequences)
        unpacked_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return unpacked_out, hidden_state, lengths, len(batch_sequences)

    def extract_temporal_features(self, sequence_graphs: List):
        """Return one LSTM hidden-state sequence per input clip."""
        unpacked_out, _hidden_state, lengths, _batch_size = self._run_lstm(
            sequence_graphs
        )
        return [
            sample_lstm_out[:length].detach()
            for sample_lstm_out, length in zip(unpacked_out, lengths.tolist())
        ]

    def extract_graph_embedding_features(self, sequence_graphs: List):
        """Return one pre-LSTM graph-embedding sequence per input clip."""
        return [
            self._build_sequence_tensor(sequence_graphs_sample).detach()
            for sequence_graphs_sample in sequence_graphs
        ]

    def forward(self, sequence_graphs: List):
        unpacked_out, hidden_state, lengths, batch_size = self._run_lstm(
            sequence_graphs
        )

        if self.temporal_readout == "last_hidden":
            if self.bidirectional:
                hidden_state = hidden_state.view(
                    self.num_layers, 2, batch_size, self.hidden_size
                )
                sequence_repr = torch.cat(
                    [hidden_state[-1, 0], hidden_state[-1, 1]], dim=-1
                )
            else:
                sequence_repr = hidden_state[-1]
            sequence_repr = self.temporal_readout_proj(sequence_repr)
        else:
            pooled_outputs = []
            for sample_lstm_out, length in zip(unpacked_out, lengths.tolist()):
                pooled_outputs.append(self.temporal_pool(sample_lstm_out[:length]))
            sequence_repr = torch.stack(pooled_outputs, dim=0)

        return self.head(sequence_repr)

    def get_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f" Total network parameters : {total_params}, trainable parameters : {trainable_params}, pct trained {100*(trainable_params/total_params)}:.2f"
        )
        return trainable_params


class GraphTemporalAggregator(nn.Module):
    """Graph embedder plus a non-recurrent temporal aggregation head."""

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
        use_proj=True,
        head_dropout=0.2,
        head_activation="gelu",
        temporal_layers=2,
        temporal_heads=8,
        temporal_ff_dim=1024,
        temporal_dropout=0.2,
        temporal_pool="attention",
    ):
        super().__init__()
        self.device = device
        self.num_graphs = int(num_graphs)
        self.n_classes = n_classes
        self.temporal_pool_mode = str(temporal_pool).lower()

        self.action_graph_embedder = ActionGraphEmbedding(
            num_verbs=num_verbs,
            num_objects=num_objects,
            num_rels=num_rels,
            num_attrs=num_attrs,
            device=device,
            **action_graph_kwargs,
        )
        self.input_dim = self.action_graph_embedder.out_dim
        self.proj = use_proj
        self.graph_emb_dim = graph_emb_dim if use_proj else self.input_dim

        if self.proj:
            layers = [nn.Linear(self.input_dim, self.graph_emb_dim)]
            if layer_norm:
                layers.append(nn.LayerNorm(self.graph_emb_dim))
            layers.append(nn.GELU() if gelu else nn.ReLU())
            self.graph_proj = nn.Sequential(*layers)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_graphs, self.graph_emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.graph_emb_dim,
            nhead=int(temporal_heads),
            dim_feedforward=int(temporal_ff_dim),
            dropout=float(temporal_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=int(temporal_layers)
        )

        if self.temporal_pool_mode == "attention":
            self.temporal_pool = AttentionPooling(
                self.graph_emb_dim,
                final_graph_emb_dim,
                hidden_features=graph_pool_interim_feat,
            )
            head_in = final_graph_emb_dim
        elif self.temporal_pool_mode == "mean_max":
            head_in = self.graph_emb_dim * 2
        elif self.temporal_pool_mode == "last":
            head_in = self.graph_emb_dim
        else:
            raise ValueError(f"Unsupported temporal_pool: {temporal_pool}")

        activation = str(head_activation).lower()
        if fc_layers_num == 1:
            self.head = nn.Linear(head_in, self.n_classes)
        else:
            head_layers = []
            hidden_dim = final_graph_emb_dim
            head_layers.append(nn.Linear(head_in, hidden_dim))
            if head_dropout > 0:
                head_layers.append(nn.Dropout(head_dropout))
            if activation == "gelu":
                head_layers.append(nn.GELU())
            elif activation == "relu":
                head_layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported head activation: {head_activation}")
            for _ in range(fc_layers_num - 2):
                head_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if head_dropout > 0:
                    head_layers.append(nn.Dropout(head_dropout))
                head_layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            head_layers.append(nn.Linear(hidden_dim, self.n_classes))
            self.head = nn.Sequential(*head_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def _graph_to_tensors(self, graph_or_tensors):
        if isinstance(graph_or_tensors, dict):
            return graph_or_tensors
        return graph_or_tensors.to_easg_tensors()

    def _build_sequence_tensor(self, sequence_graphs: Dict[int, Dict[str, torch.Tensor]]):
        graph_embs = [
            self.action_graph_embedder(self._graph_to_tensors(graph)).to(self.device)
            for graph in sequence_graphs.values()
        ]
        if not graph_embs:
            raise ValueError("Received an empty graph sequence.")
        graph_embs = torch.stack(graph_embs, dim=0)
        if self.proj:
            graph_embs = self.graph_proj(graph_embs)
        return graph_embs

    def forward(self, sequence_graphs: List):
        batch_sequences = [
            self._build_sequence_tensor(sequence_graphs_sample)
            for sequence_graphs_sample in sequence_graphs
        ]
        lengths = torch.tensor(
            [seq.shape[0] for seq in batch_sequences], device=self.device
        )
        padded_sequences = pad_sequence(batch_sequences, batch_first=True)
        max_len = padded_sequences.shape[1]
        if max_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {max_len} exceeds configured num_graphs {self.pos_emb.shape[1]}"
            )
        padded_sequences = padded_sequences + self.pos_emb[:, :max_len]

        positions = torch.arange(max_len, device=self.device).unsqueeze(0)
        pad_mask = positions >= lengths.unsqueeze(1)
        encoded = self.temporal_encoder(padded_sequences, src_key_padding_mask=pad_mask)

        pooled = []
        for sample_encoded, length in zip(encoded, lengths.tolist()):
            valid = sample_encoded[:length]
            if self.temporal_pool_mode == "attention":
                pooled.append(self.temporal_pool(valid))
            elif self.temporal_pool_mode == "mean_max":
                pooled.append(torch.cat([valid.mean(dim=0), valid.max(dim=0).values]))
            else:
                pooled.append(valid[-1])
        sequence_repr = torch.stack(pooled, dim=0)
        return self.head(sequence_repr)

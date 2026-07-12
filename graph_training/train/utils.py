import json
import os

import torch
from torch.utils.data import DataLoader

from dataset.GraphDataset import feature_collate_fn
from modeling.GraphMLP import GraphMLP
from train.evaluate import evaluate


def float_metrics(metrics):
    return {key: float(value) for key, value in metrics.items()}


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def resolve_device(device):
    return "cpu" if device == "cuda" and not torch.cuda.is_available() else device


def graph_relation_count(relationships, graph_type, drop_aux=False):
    if graph_type != "pruned":
        return len(relationships)
    rels = {
        name: idx
        for name, idx in relationships.items()
        if not drop_aux or name not in {"aux_direct_object", "aux_verb"}
    }
    if "gazed_at" not in rels:
        rels["gazed_at"] = max(rels.values(), default=-1) + 1
    return max(rels.values(), default=-1) + 1


def build_optimizer(model, training_cfg):
    name = training_cfg["optimizer"]
    kwargs = {
        "lr": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, training_cfg):
    factor = training_cfg["scheduler_factor"]
    if factor == 1.0:
        return None
    return torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _epoch: factor
    )


def loader_kwargs(worker_count, pin_memory=False, persistent_workers=None, prefetch_factor=2):
    kwargs = {"num_workers": worker_count, "pin_memory": pin_memory}
    if worker_count > 0:
        kwargs["persistent_workers"] = (
            worker_count > 0 if persistent_workers is None else persistent_workers
        )
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def make_loader(dataset, batch_size, shuffle=False, sampler=None, **kwargs):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=feature_collate_fn,
        **kwargs,
    )


def make_loaders(
    datasets,
    data_cfg,
    train_sampler=None,
    eval_num_workers=None,
):
    train_dataset, val_dataset, test_dataset = datasets
    num_workers = int(data_cfg["num_workers"])
    eval_workers = int(
        data_cfg.get("eval_num_workers", min(num_workers, 4))
        if eval_num_workers is None
        else eval_num_workers
    )
    pin_memory = bool(data_cfg.get("pin_memory", False))
    common = {
        "pin_memory": pin_memory,
        "persistent_workers": bool(data_cfg.get("persistent_workers", num_workers > 0)),
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)),
    }
    return (
        make_loader(
            train_dataset,
            data_cfg["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs(num_workers, **common),
        ),
        make_loader(
            val_dataset,
            data_cfg["batch_size"],
            **loader_kwargs(eval_workers, **common),
        ),
        make_loader(
            test_dataset,
            data_cfg["batch_size"],
            **loader_kwargs(eval_workers, **common),
        ),
    )


def build_graph_mlp(config, vocab, n_classes, num_rels, device):
    mlp_cfg = config["mlp"]
    projector_cfg = mlp_cfg["projector"]
    pool_cfg = mlp_cfg["attention_pooler"]
    head_cfg = mlp_cfg.get("head", {})
    return GraphMLP(
        num_graphs=mlp_cfg["num_graphs"],
        num_verbs=len(vocab["verbs"]),
        num_objects=len(vocab["objects"]),
        num_rels=num_rels,
        num_attrs=len(vocab["attributes"]),
        n_classes=n_classes,
        fc_layers_num=mlp_cfg["fc_layers_num"],
        graph_emb_dim=projector_cfg["graph_emb_dim"],
        final_graph_emb_dim=pool_cfg["final_graph_emb_dim"],
        graph_pool_interim_feat=pool_cfg["graph_pool_interim_feat"],
        layer_norm=projector_cfg.get("layer_norm", True),
        gelu=projector_cfg.get("gelu", True),
        head_dropout=head_cfg.get("dropout", 0.5),
        head_activation=head_cfg.get("activation", "gelu"),
        device=device,
        action_graph_kwargs=mlp_cfg["action_graph_embedder"],
        use_pool=mlp_cfg["use_pool"],
        use_proj=mlp_cfg["use_proj"],
    ).to(device)


def load_best_checkpoint_if_available(model, save_path, metric="top1", device="cpu"):
    prefix = f"best_model_{metric}_epoch_"
    candidates = [
        name for name in os.listdir(save_path) if name.startswith(prefix) and name.endswith(".pt")
    ]
    if not candidates:
        return None
    checkpoint_path = os.path.join(
        save_path,
        max(candidates, key=lambda name: int(name[len(prefix) : -3])),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint_path


def evaluate_checkpoint(model, save_path, metric, test_loader, test_dataset, device):
    checkpoint_path = load_best_checkpoint_if_available(model, save_path, metric, device)
    print(
        f"Loaded best {metric} checkpoint for test: {checkpoint_path}"
        if checkpoint_path
        else f"No best {metric} checkpoint found; using current model."
    )
    test_result, preds, targets = evaluate(
        model, test_loader, device, num_classes=len(test_dataset.activity_to_idx)
    )
    return {
        "checkpoint_metric": metric,
        "checkpoint_path": checkpoint_path,
        "metrics": float_metrics(test_result["eval_metrics"]),
        "predictions": [test_dataset.idx_to_activity[i] for i in preds],
        "targets": [test_dataset.idx_to_activity[i] for i in targets],
    }


def write_training_outputs(save_path, results, epoch_preds):
    save_json(
        os.path.join(save_path, "training_results.json"),
        {
            str(ep): {
                "train": float_metrics(res["train"]["eval_metrics"]),
                "val": float_metrics(res["val"]["eval_metrics"]),
            }
            for ep, res in results.items()
        },
    )
    save_json(
        os.path.join(save_path, "per_epoch_predictions.json"),
        {
            str(ep): {"predictions": res["predictions"], "targets": res["targets"]}
            for ep, res in epoch_preds.items()
        },
    )

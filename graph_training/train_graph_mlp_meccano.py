import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from datetime import datetime

import torch
from dataset.GraphDataset import (
    GraphDatasetMeccano,
    feature_collate_fn
)
from dataset.meccano_aux import return_meccano_train_val_samples

from modeling.GraphMLP import GraphMLP
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.evaluate import build_loss_fn, compute_class_weights, store_model
from train.train import do_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to the experiment config JSON file")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, "r") as f:
        config = json.load(f)

    mlp_cfg = config["mlp"]
    action_graph_cfg = mlp_cfg["action_graph_embedder"]
    projector_cfg = mlp_cfg["projector"]
    attention_pool_cfg = mlp_cfg["attention_pooler"]

    experiment_name = config["experiment_name"]
    print(f"Running experiment: {experiment_name}")

    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_epochs = config["training"]["num_epochs"]
    optimizer_name = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    scheduler_factor = config["training"]["scheduler_factor"]
    criterion_metrics = config["training"]["criterion_metrics"]

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    pin_memory = config["data"]["pin_memory"]

    fc_layers_num = mlp_cfg["fc_layers_num"]
    num_graphs = mlp_cfg["num_graphs"]

    use_pool = config["mlp"]["use_pool"]
    use_proj = config["mlp"]["use_proj"]

    graph_emb_dim = projector_cfg["graph_emb_dim"]
    layer_norm = projector_cfg.get("layer_norm", True)
    gelu = projector_cfg.get("gelu", True)

    graph_pool_interim_feat = attention_pool_cfg["graph_pool_interim_feat"]
    final_graph_emb_dim = attention_pool_cfg["final_graph_emb_dim"]

    data_path = config["data"]["input_path"]
    train_actions_csv = config["data"]["train_actions_csv"]
    val_actions_csv = config["data"]["val_actions_csv"]

    train_samples, val_samples, activity_to_idx, split_stats = return_meccano_train_val_samples(
        dataset_root=data_path,
        train_actions_csv=train_actions_csv,
        val_actions_csv=val_actions_csv,
        num_graphs=num_graphs,
    )

    metadata_root = data_path
    if not os.path.exists(os.path.join(metadata_root, "verbs.json")):
        metadata_root = os.path.join(data_path, "Train")

    with open(os.path.join(metadata_root, "verbs.json"), "r") as f:
        verbs = json.load(f)

    with open(os.path.join(metadata_root, "objects.json"), "r") as f:
        objs = json.load(f)

    with open(os.path.join(metadata_root, "relationships.json"), "r") as f:
        rels = json.load(f)

    with open(os.path.join(metadata_root, "attributes.json"), "r") as f:
        attrs = json.load(f)

    graph_type = config["data"].get("graph_type", "full")

    if graph_type == "pruned":
        num_rels = len(rels)
        if "aux_direct_object" in rels:
            num_rels -= 1
        if "aux_verb" in rels:
            num_rels -= 1
        num_rels += 1
    else:
        num_rels = len(rels)

    print(f"activity_to_idx : {activity_to_idx}")
    print(f"len(train_samples) : {len(train_samples)}")
    print(f"len(val_samples) : {len(val_samples)}")
    print(f"Graph type : {graph_type}")
    print(f"MECCANO split stats : {json.dumps(split_stats, indent=2)}")

    train_dataset = GraphDatasetMeccano(
        data_path, 
        train_samples, 
        activity_to_idx, 
        graph_type
    )

    validation_dataset = GraphDatasetMeccano(
        data_path, val_samples, activity_to_idx, graph_type
    )
    
    
    assert train_dataset.activity_to_idx == validation_dataset.activity_to_idx
    cls_mapping = train_dataset.activity_to_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_collate_fn,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=feature_collate_fn,
    )
    model = GraphMLP(
        num_graphs=num_graphs,
        num_verbs=len(verbs),
        num_objects=len(objs),
        num_rels=num_rels,
        num_attrs=len(attrs),
        n_classes=len(cls_mapping),
        fc_layers_num=fc_layers_num,
        graph_emb_dim=graph_emb_dim,
        final_graph_emb_dim=final_graph_emb_dim,
        graph_pool_interim_feat=graph_pool_interim_feat,
        layer_norm=layer_norm,
        gelu=gelu,
        device=device,
        action_graph_kwargs=action_graph_cfg,
        use_pool=use_pool,
        use_proj=use_proj,
    ).to(device)

    class_weights = None
    if config["training"]["loss"]["ifw"]:
        class_weights = compute_class_weights(train_dataset, activity_to_idx)
        print(class_weights)
    loss_func = build_loss_fn(config["training"]["loss"], class_weights)

    if optimizer_name == "adam":
        opt = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    scheduler = None
    if scheduler_factor != 1.0:
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            opt, lr_lambda=lambda epoch: scheduler_factor
        )

    save_path = os.path.join(
        config["output"]["base_path"],
        config["experiment_name"],
        f"dino_model_fc_layer_{fc_layers_num}_num_epoch_{num_epochs}_graph_emb_dim_{graph_emb_dim}_final_graph_emb_dim_{final_graph_emb_dim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(save_path, exist_ok=True)

    results = {}
    best_epoch_result = {"acc": -1, "f1": -1}
    global_step = 0

    experiment_config = {
        "experiment_name": experiment_name,
        "fc_layers_num": fc_layers_num,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "scheduler_factor": scheduler_factor,
        "num_classes": len(cls_mapping),
        "device": device,
        "mlp": config["mlp"],
        "data": config["data"],
        "split_stats": split_stats,
    }

    with open(os.path.join(save_path, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=2)

    with open(os.path.join(save_path, "class_mapping.json"), "w") as f:
        json.dump(cls_mapping, f, indent=2)

    model.train()
    trainable_params = model.get_trainable_params()

    epoch_preds = {k: {} for k in range(num_epochs)}

    for epoch in tqdm(
        range(num_epochs),
        desc=f"Training for {num_epochs} epochs",
        unit="epoch",
        total=num_epochs,
    ):
        print(f"Epoch {epoch+1}/{num_epochs}\n")

        epoch_result, global_step, preds, targets = do_epoch(
            device=device,
            net=model,
            opt=opt,
            train_loader=train_loader,
            validate_loader=val_loader,
            global_step=global_step,
            num_classes_train=len(cls_mapping),
            num_classes_val=len(validation_dataset.activity_to_idx),
            loss_func=loss_func,
        )

        epoch_preds[epoch]["predictions"] = [
            train_dataset.idx_to_activity[i] for i in preds
        ]
        epoch_preds[epoch]["targets"] = [
            train_dataset.idx_to_activity[i] for i in targets
        ]

        val_metrics = epoch_result["val"]["eval_metrics"]
        train_metrics = epoch_result["train"]["eval_metrics"]

        print(
            f"\nValidation - Accuracy: {val_metrics['acc']*100:.2f}%, F1: {val_metrics['f1']*100:.2f}%"
        )

        if val_metrics["acc"] > best_epoch_result["acc"]:
            best_epoch_result["acc"] = val_metrics["acc"]
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="acc"
            )
            print(f"New best accuracy model saved: {val_metrics['acc']*100:.2f}%")

        if val_metrics["f1"] > best_epoch_result["f1"]:
            best_epoch_result["f1"] = val_metrics["f1"]
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="f1"
            )
            print(f"New best F1 model saved: {val_metrics['f1']*100:.2f}%")

        results[epoch] = epoch_result

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1} completed")

    with open(os.path.join(save_path, "training_results.json"), "w") as f:
        json_results = {}
        for ep, res in results.items():
            json_results[str(ep)] = {
                "train": {
                    "acc": float(res["train"]["eval_metrics"]["acc"]),
                    "f1": float(res["train"]["eval_metrics"]["f1"]),
                },
                "val": {
                    "acc": float(res["val"]["eval_metrics"]["acc"]),
                    "f1": float(res["val"]["eval_metrics"]["f1"]),
                },
            }
        json.dump(json_results, f, indent=2)

    with open(os.path.join(save_path, "per_epoch_predictions.json"), "w") as f:
        json_results = {}
        for ep, res in epoch_preds.items():
            json_results[str(ep)] = {
                "predictions": res["predictions"],
                "targets": res["targets"],
            }
        json.dump(json_results, f, indent=2)

    print(f"Best validation accuracy: {best_epoch_result['acc']*100:.2f}%")
    print(f"Best validation F1: {best_epoch_result['f1']*100:.2f}%")

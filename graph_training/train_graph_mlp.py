import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from datetime import datetime

from dataset.GraphDataset import (
    GraphDatasetAria,
)
from dataset.aria_aux import return_train_val_samples
from tqdm import tqdm
from train.evaluate import build_loss_fn, compute_class_weights, store_model
from train.train import do_epoch
from train.utils import (
    build_graph_mlp,
    build_optimizer,
    build_scheduler,
    graph_relation_count,
    loader_kwargs,
    make_loader,
    resolve_device,
    save_json,
    write_training_outputs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Path to the experiment config JSON file")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, "r") as f:
        config = json.load(f)

    mlp_cfg = config["mlp"]
    projector_cfg = mlp_cfg["projector"]
    attention_pool_cfg = mlp_cfg["attention_pooler"]

    experiment_name = config["experiment_name"]
    print(f"Running experiment: {experiment_name}")

    device = resolve_device(config["device"])

    num_epochs = config["training"]["num_epochs"]
    optimizer_name = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    scheduler_factor = config["training"]["scheduler_factor"]

    batch_size = config["data"]["batch_size"]

    fc_layers_num = mlp_cfg["fc_layers_num"]
    graph_emb_dim = projector_cfg["graph_emb_dim"]
    final_graph_emb_dim = attention_pool_cfg["final_graph_emb_dim"]

    train_samples, val_samples, activity_to_idx = return_train_val_samples(
        pooling="concat"
    )

    data_path = config["data"]["input_path"]
    with open(os.path.join(data_path, "verbs.json"), "r") as f:
        verbs = json.load(f)

    with open(os.path.join(data_path, "objects.json"), "r") as f:
        objs = json.load(f)

    with open(os.path.join(data_path, "relationships.json"), "r") as f:
        rels = json.load(f)

    with open(os.path.join(data_path, "attributes.json"), "r") as f:
        attrs = json.load(f)

    graph_type = config["data"].get("graph_type", "full")

    num_rels = graph_relation_count(rels, graph_type, drop_aux=True)

    print(f"activity_to_idx : {activity_to_idx}")
    print(f"len(train_samples) : {len(train_samples)}")
    print(f"len(val_samples) : {len(val_samples)}")
    print(f"Graph type : {graph_type}")

    train_dataset = GraphDatasetAria(
        data_path, train_samples, activity_to_idx, graph_type
    )

    validation_dataset = GraphDatasetAria(
        data_path, val_samples, activity_to_idx, graph_type
    )
    
    
    assert train_dataset.activity_to_idx == validation_dataset.activity_to_idx
    cls_mapping = train_dataset.activity_to_idx

    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs(config["data"]["num_workers"], config["data"]["pin_memory"]),
    )
    val_loader = make_loader(
        validation_dataset,
        batch_size=batch_size,
    )
    vocab = {"verbs": verbs, "objects": objs, "relationships": rels, "attributes": attrs}
    model = build_graph_mlp(config, vocab, len(cls_mapping), num_rels, device)

    class_weights = None
    if config["training"]["loss"]["ifw"]:
        class_weights = compute_class_weights(train_dataset, activity_to_idx)
        print(class_weights)
    loss_func = build_loss_fn(config["training"]["loss"], class_weights)

    opt = build_optimizer(model, config["training"])

    scheduler = build_scheduler(opt, config["training"])

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
    }

    save_json(os.path.join(save_path, "experiment_config.json"), experiment_config)
    save_json(os.path.join(save_path, "class_mapping.json"), cls_mapping)

    model.train()
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
        print(
            f"\nValidation - Accuracy: {val_metrics['acc']*100:.2f}%, F1: {val_metrics['f1']*100:.2f}%"
        )

        checkpoint_acc_metric = val_metrics.get("top5", val_metrics["acc"])
        if checkpoint_acc_metric > best_epoch_result["acc"]:
            best_epoch_result["acc"] = checkpoint_acc_metric
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="acc"
            )
            print(f"New best Top-5 accuracy model saved: {checkpoint_acc_metric*100:.2f}%")

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

    write_training_outputs(save_path, results, epoch_preds)

    print(f"Best validation Top-5 accuracy: {best_epoch_result['acc']*100:.2f}%")
    print(f"Best validation F1: {best_epoch_result['f1']*100:.2f}%")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import torch
from dataset.GraphDataset import (
    GraphDatasetMeccano,
)
from dataset.meccano_aux import (
    resolve_meccano_global_root,
    resolve_meccano_split_root,
    return_meccano_train_val_test_samples,
)
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from train.evaluate import (
    build_loss_fn,
    compute_class_weights,
    store_model,
)
from train.train import do_epoch
from train.utils import (
    build_graph_mlp,
    build_optimizer,
    build_scheduler,
    evaluate_checkpoint,
    graph_relation_count,
    make_loaders,
    resolve_device,
    save_json,
    write_training_outputs,
)


def build_train_sampler(dataset, activity_to_idx, sampler_cfg):
    if not sampler_cfg or sampler_cfg.get("name") in {None, "none"}:
        return None
    if sampler_cfg.get("name") != "balanced":
        raise ValueError(f"Unsupported sampler: {sampler_cfg.get('name')}")

    labels = [activity_to_idx[label_str] for _, _, label_str, _, _ in dataset.sample_index]
    counts = Counter(labels)
    power = float(sampler_cfg.get("power", 1.0))
    weights = torch.tensor(
        [1.0 / (counts[label] ** power) for label in labels],
        dtype=torch.double,
    )
    num_samples = int(sampler_cfg.get("num_samples", len(weights)))
    replacement = bool(sampler_cfg.get("replacement", True))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)


def main(args, config):
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
    num_graphs = mlp_cfg["num_graphs"]

    graph_emb_dim = projector_cfg["graph_emb_dim"]
    final_graph_emb_dim = attention_pool_cfg["final_graph_emb_dim"]

    data_path = config["data"]["input_path"]
    metadata_root = config["data"]["metadata_root"]
    train_actions_csv = config["data"]["train_actions_csv"]
    val_actions_csv = config["data"]["val_actions_csv"]
    test_actions_csv = config["data"]["test_actions_csv"]
    easg_cache_path = config["data"].get("easg_cache_path")
    feature_mode = config["data"].get("feature_mode")
    concat_depth_features = config["data"].get("concat_depth_features", False)
    depth_feature_root = config["data"].get("depth_feature_root")
    depth_feature_dim = config["data"].get("depth_feature_dim", 1536)
    rgb_feature_filename = config["data"].get(
        "rgb_feature_filename", "frame_features_model_dinov3h16+.h5"
    )

    (
        train_samples,
        val_samples,
        test_samples,
        activity_to_idx,
        split_stats,
    ) = return_meccano_train_val_test_samples(
        dataset_root=data_path,
        train_actions_csv=train_actions_csv,
        val_actions_csv=val_actions_csv,
        test_actions_csv=test_actions_csv,
        num_graphs=num_graphs,
    )

    global_metadata_root = resolve_meccano_global_root(metadata_root)
    train_metadata_root = resolve_meccano_split_root(metadata_root, "Train")
    vocab_root = global_metadata_root or train_metadata_root

    if global_metadata_root is not None:
        verbs_path = os.path.join(vocab_root, "global_verbs.json")
        objs_path = os.path.join(vocab_root, "global_objects.json")
        rels_path = os.path.join(vocab_root, "global_relationships.json")
        attrs_path = os.path.join(vocab_root, "global_attributes.json")
    else:
        verbs_path = os.path.join(vocab_root, "verbs.json")
        objs_path = os.path.join(vocab_root, "objects.json")
        rels_path = os.path.join(vocab_root, "relationships.json")
        attrs_path = os.path.join(vocab_root, "attributes.json")

    with open(verbs_path, "r") as f:
        verbs = json.load(f)

    with open(objs_path, "r") as f:
        objs = json.load(f)

    with open(rels_path, "r") as f:
        rels = json.load(f)

    with open(attrs_path, "r") as f:
        attrs = json.load(f)

    graph_type = config["data"].get("graph_type", "full")
    num_rels = graph_relation_count(rels, graph_type, drop_aux=True)

    print(f"activity_to_idx : {activity_to_idx}")
    print(f"len(train_samples) : {len(train_samples)}")
    print(f"len(val_samples) : {len(val_samples)}")
    print(f"len(test_samples) : {len(test_samples)}")
    print(f"Graph type : {graph_type}")
    print(f"Vocabulary root : {vocab_root}")
    print(f"EASG cache path : {easg_cache_path}")
    print(f"RGB feature filename : {rgb_feature_filename}")
    print(f"MECCANO split stats : {json.dumps(split_stats, indent=2)}")

    train_dataset = GraphDatasetMeccano(
        metadata_root,
        "Train",
        train_samples,
        activity_to_idx,
        graph_type,
        easg_cache_path=easg_cache_path,
        concat_depth_features=concat_depth_features,
        feature_mode=feature_mode,
        depth_feature_root=depth_feature_root,
        depth_feature_dim=depth_feature_dim,
        rgb_feature_filename=rgb_feature_filename,
    )
    validation_dataset = GraphDatasetMeccano(
        metadata_root,
        "Val",
        val_samples,
        activity_to_idx,
        graph_type,
        easg_cache_path=easg_cache_path,
        concat_depth_features=concat_depth_features,
        feature_mode=feature_mode,
        depth_feature_root=depth_feature_root,
        depth_feature_dim=depth_feature_dim,
        rgb_feature_filename=rgb_feature_filename,
    )
    test_dataset = GraphDatasetMeccano(
        metadata_root,
        "Test",
        test_samples,
        activity_to_idx,
        graph_type,
        easg_cache_path=easg_cache_path,
        concat_depth_features=concat_depth_features,
        feature_mode=feature_mode,
        depth_feature_root=depth_feature_root,
        depth_feature_dim=depth_feature_dim,
        rgb_feature_filename=rgb_feature_filename,
    )

    assert train_dataset.activity_to_idx == validation_dataset.activity_to_idx
    assert train_dataset.activity_to_idx == test_dataset.activity_to_idx
    cls_mapping = train_dataset.activity_to_idx

    sampler_cfg = config["training"].get("sampler", {"name": "none"})
    train_sampler = build_train_sampler(train_dataset, activity_to_idx, sampler_cfg)
    train_loader, val_loader, test_loader = make_loaders(
        (train_dataset, validation_dataset, test_dataset),
        config["data"],
        train_sampler=train_sampler,
        eval_num_workers=0,
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
        (
            f"dino_model_fc_layer_{fc_layers_num}_num_epoch_{num_epochs}"
            f"_graph_emb_dim_{graph_emb_dim}_final_graph_emb_dim_{final_graph_emb_dim}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    results = {}
    best_epoch_result = {"top1": -1, "top5": -1, "f1": -1}
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
        "sampler": sampler_cfg,
        "loss": config["training"]["loss"],
        "num_classes": len(cls_mapping),
        "device": device,
        "mlp": config["mlp"],
        "data": config["data"],
        "split_stats": split_stats,
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
            "\nValidation metrics: "
            f"Top-1 {val_metrics['top1']*100:.2f}% | "
            f"Top-5 {val_metrics['top5']*100:.2f}% | "
            f"Avg. Prec. {val_metrics['avg_precision']*100:.2f}% | "
            f"Avg. Recall {val_metrics['avg_recall']*100:.2f}% | "
            f"Avg. F1 {val_metrics['avg_f1']*100:.2f}%"
        )

        if val_metrics["top1"] > best_epoch_result["top1"]:
            best_epoch_result["top1"] = val_metrics["top1"]
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="top1"
            )
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="acc"
            )
            print(f"New best Top-1 accuracy model saved: {val_metrics['top1']*100:.2f}%")

        if val_metrics["top5"] > best_epoch_result["top5"]:
            best_epoch_result["top5"] = val_metrics["top5"]
            store_model(
                net=model, opt=opt, epoch=epoch, save_path=save_path, metric="top5"
            )
            print(f"New best Top-5 accuracy model saved: {val_metrics['top5']*100:.2f}%")

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

    top1_test_summary = evaluate_checkpoint(
        model,
        save_path,
        "top1",
        test_loader,
        test_dataset,
        device,
    )
    top5_test_summary = evaluate_checkpoint(
        model,
        save_path,
        "top5",
        test_loader,
        test_dataset,
        device,
    )

    save_json(os.path.join(save_path, "final_test_results_top1.json"), top1_test_summary)
    save_json(os.path.join(save_path, "final_test_results_top5.json"), top5_test_summary)
    save_json(os.path.join(save_path, "final_test_results.json"), top1_test_summary)

    print(f"Best validation Top-1 accuracy: {best_epoch_result['top1']*100:.2f}%")
    print(f"Best validation Top-5 accuracy: {best_epoch_result['top5']*100:.2f}%")
    print(f"Best validation F1: {best_epoch_result['f1']*100:.2f}%")
    print(
        "Best Top-1 checkpoint test metrics: "
        f"Top-1 {top1_test_summary['metrics']['top1']*100:.2f}% | "
        f"Top-5 {top1_test_summary['metrics']['top5']*100:.2f}% | "
        f"Avg. Prec. {top1_test_summary['metrics']['avg_precision']*100:.2f}% | "
        f"Avg. Recall {top1_test_summary['metrics']['avg_recall']*100:.2f}% | "
        f"Avg. F1 {top1_test_summary['metrics']['avg_f1']*100:.2f}%"
    )
    print(
        "Best Top-5 checkpoint test metrics: "
        f"Top-1 {top5_test_summary['metrics']['top1']*100:.2f}% | "
        f"Top-5 {top5_test_summary['metrics']['top5']*100:.2f}% | "
        f"Avg. Prec. {top5_test_summary['metrics']['avg_precision']*100:.2f}% | "
        f"Avg. Recall {top5_test_summary['metrics']['avg_recall']*100:.2f}% | "
        f"Avg. F1 {top5_test_summary['metrics']['avg_f1']*100:.2f}%"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, help="Path to the experiment config JSON file"
    )
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, "r") as f:
        config = json.load(f)

    main(args, config)

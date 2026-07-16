from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

GRAPH_TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GRAPH_TRAINING_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GRAPH_TRAINING_ROOT))

from dataset.GraphDataset import GraphDatasetEgtea
from modeling.LSTM import GraphLSTM, GraphTemporalAggregator
from train.evaluate import build_loss_fn, compute_class_weights, store_model
from train.train import do_epoch
from train_graph_mlp_egtea import (
    build_activity_mapping,
    build_train_sampler,
    build_vocab,
    collect_split_samples,
    split_train_val,
)
from train.utils import (
    build_optimizer,
    build_scheduler,
    evaluate_checkpoint,
    float_metrics,
    graph_relation_count,
    make_loaders,
    resolve_device,
    save_json,
    write_training_outputs,
)


def _build_split_datasets(config, split_id):
    mlp_cfg = config["mlp"]
    data_cfg = config["data"]
    num_graphs = mlp_cfg["num_graphs"]
    input_root = data_cfg["input_path"]
    rgb_feature_filename = data_cfg.get(
        "rgb_feature_filename", "frame_features_model_dinov3_vitl16.h5"
    )

    train_split_root = data_cfg.get(
        "train_split_root", os.path.join(input_root, "train", f"train_split_{split_id}")
    )
    test_split_root = data_cfg.get(
        "test_split_root", os.path.join(input_root, "test", f"test_split_{split_id}")
    )
    annotations_root = data_cfg.get(
        "annotations_root",
        "/path/to/ego_graphs/vlm_datasets/egtea_gaze/annotations",
    )
    train_actions_path = os.path.join(annotations_root, f"train_split{split_id}.txt")
    test_actions_path = os.path.join(annotations_root, f"test_split{split_id}.txt")
    action_idx_path = os.path.join(annotations_root, "action_idx.txt")

    train_all_samples, train_stats = collect_split_samples(
        train_split_root,
        data_cfg["feature_root"],
        num_graphs,
        rgb_feature_filename,
        split_actions_path=train_actions_path,
        action_idx_path=action_idx_path,
    )
    test_samples, test_stats = collect_split_samples(
        test_split_root,
        data_cfg["feature_root"],
        num_graphs,
        rgb_feature_filename,
        split_actions_path=test_actions_path,
        action_idx_path=action_idx_path,
    )
    val_distribution_source = str(
        data_cfg.get("val_distribution_source", "test")
    ).lower()
    val_reference_samples = test_samples if val_distribution_source == "test" else None
    train_samples, val_samples = split_train_val(
        train_all_samples,
        float(data_cfg.get("val_fraction", 0.1)),
        val_num_samples=data_cfg.get("val_num_samples"),
        reference_samples=val_reference_samples,
    )
    all_samples = train_samples + val_samples + test_samples
    activity_to_idx = build_activity_mapping(all_samples)
    vocab = build_vocab(all_samples, input_root)
    graph_type = data_cfg.get("graph_type", "full")

    dataset_kwargs = {
        "activity_to_idx": activity_to_idx,
        "graph_type": graph_type,
        "vocab": vocab,
        "clip_text_path": data_cfg.get("clip_text_path"),
        "easg_cache_path": data_cfg.get("easg_cache_path"),
        "rgb_feature_filename": rgb_feature_filename,
        "additional_feature_mode": data_cfg.get("additional_feature_mode"),
        "additional_feature_dim": data_cfg.get("additional_feature_dim", 0),
        "ohd_feature_filename": data_cfg.get(
            "ohd_feature_filename", "hand_grounding_results_gdino_base.pkl"
        ),
        "ohd_max_hands": data_cfg.get("ohd_max_hands", 2),
        "ohd_hand_feature_dim": data_cfg.get("ohd_hand_feature_dim", 256),
    }
    train_dataset = GraphDatasetEgtea(train_samples, **dataset_kwargs)
    validation_dataset = GraphDatasetEgtea(val_samples, **dataset_kwargs)
    test_dataset = GraphDatasetEgtea(test_samples, **dataset_kwargs)

    split_stats = {
        "train_all": train_stats,
        "test": test_stats,
        "train_after_val_split": len(train_samples),
        "val_after_val_split": len(val_samples),
        "val_distribution_source": val_distribution_source,
    }
    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        activity_to_idx,
        vocab,
        split_stats,
    )


def _build_loaders(
    config, train_dataset, validation_dataset, test_dataset, activity_to_idx
):
    sampler_cfg = config["training"].get("sampler", {"name": "none"})
    train_sampler = build_train_sampler(train_dataset, activity_to_idx, sampler_cfg)
    return make_loaders(
        (train_dataset, validation_dataset, test_dataset),
        config["data"],
        train_sampler=train_sampler,
    )


def _build_model(config, vocab, activity_to_idx, num_rels, device):
    mlp_cfg = config["mlp"]
    lstm_cfg = config.get("lstm", {})
    temporal_cfg = config.get("temporal_aggregator", {})
    action_graph_cfg = mlp_cfg["action_graph_embedder"]
    projector_cfg = mlp_cfg["projector"]
    attention_pool_cfg = mlp_cfg["attention_pooler"]
    head_cfg = mlp_cfg.get("head", {})
    model_type = str(config.get("model_type", "lstm")).lower()

    if model_type in {"temporal_aggregator", "graph_temporal_aggregator", "tempagg"}:
        return GraphTemporalAggregator(
            num_graphs=mlp_cfg["num_graphs"],
            num_verbs=len(vocab["verbs"]),
            num_objects=len(vocab["objects"]),
            num_rels=num_rels,
            num_attrs=len(vocab["attributes"]),
            n_classes=len(activity_to_idx),
            fc_layers_num=mlp_cfg["fc_layers_num"],
            graph_emb_dim=projector_cfg["graph_emb_dim"],
            final_graph_emb_dim=attention_pool_cfg["final_graph_emb_dim"],
            graph_pool_interim_feat=attention_pool_cfg["graph_pool_interim_feat"],
            layer_norm=projector_cfg.get("layer_norm", True),
            gelu=projector_cfg.get("gelu", True),
            head_dropout=head_cfg.get("dropout", 0.5),
            head_activation=head_cfg.get("activation", "gelu"),
            device=device,
            action_graph_kwargs=action_graph_cfg,
            use_proj=mlp_cfg["use_proj"],
            temporal_layers=temporal_cfg.get("layers", 2),
            temporal_heads=temporal_cfg.get("heads", 8),
            temporal_ff_dim=temporal_cfg.get("ff_dim", 1024),
            temporal_dropout=temporal_cfg.get("dropout", 0.2),
            temporal_pool=temporal_cfg.get("pool", "attention"),
        ).to(device)

    return GraphLSTM(
        num_graphs=mlp_cfg["num_graphs"],
        num_verbs=len(vocab["verbs"]),
        num_objects=len(vocab["objects"]),
        num_rels=num_rels,
        num_attrs=len(vocab["attributes"]),
        n_classes=len(activity_to_idx),
        fc_layers_num=mlp_cfg["fc_layers_num"],
        graph_emb_dim=projector_cfg["graph_emb_dim"],
        final_graph_emb_dim=attention_pool_cfg["final_graph_emb_dim"],
        graph_pool_interim_feat=attention_pool_cfg["graph_pool_interim_feat"],
        layer_norm=projector_cfg.get("layer_norm", True),
        gelu=projector_cfg.get("gelu", True),
        head_dropout=head_cfg.get("dropout", 0.5),
        head_activation=head_cfg.get("activation", "gelu"),
        device=device,
        action_graph_kwargs=action_graph_cfg,
        use_pool=mlp_cfg["use_pool"],
        use_proj=mlp_cfg["use_proj"],
        hidden_size=lstm_cfg.get(
            "hidden_size", attention_pool_cfg["final_graph_emb_dim"]
        ),
        num_layers=lstm_cfg.get("num_layers", 2),
        bias=lstm_cfg.get("bias", True),
        bidirectional=lstm_cfg.get("bidirectional", False),
        recurrent_dropout=lstm_cfg.get("recurrent_dropout", 0.0),
        temporal_readout=lstm_cfg.get("readout"),
    ).to(device)


def _print_split_metrics(split_id, summary):
    metrics = summary["metrics"]
    print(
        f"Split {split_id} test metrics "
        f"({summary['checkpoint_metric']} checkpoint): "
        f"Mean Acc. {metrics['mean_accuracy'] * 100:.2f}% | "
        f"Top-1 {metrics['top1'] * 100:.2f}% | "
        f"Top-5 {metrics['top5'] * 100:.2f}% | "
        f"F1 {metrics['f1'] * 100:.2f}%"
    )


def train_one_split(config, split_id, run_root):
    split_config = copy.deepcopy(config)
    split_config["data"]["split_id"] = int(split_id)
    mlp_cfg = split_config["mlp"]
    lstm_cfg = split_config.get("lstm", {})
    projector_cfg = mlp_cfg["projector"]
    attention_pool_cfg = mlp_cfg["attention_pooler"]

    device = resolve_device(split_config["device"])

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        activity_to_idx,
        vocab,
        split_stats,
    ) = _build_split_datasets(split_config, split_id)
    graph_type = split_config["data"].get("graph_type", "full")
    num_rels = graph_relation_count(vocab["relationships"], graph_type)

    print(f"\nRunning experiment: {split_config['experiment_name']}")
    print(f"EGTEA split id: {split_id}")
    print(f"len(train_samples): {len(train_dataset)}")
    print(f"len(val_samples): {len(validation_dataset)}")
    print(f"len(test_samples): {len(test_dataset)}")
    print(f"num_classes: {len(activity_to_idx)}")
    print(
        "vocab sizes: "
        f"verbs={len(vocab['verbs'])} objects={len(vocab['objects'])} "
        f"rels={num_rels} attrs={len(vocab['attributes'])}"
    )

    train_loader, val_loader, test_loader = _build_loaders(
        split_config, train_dataset, validation_dataset, test_dataset, activity_to_idx
    )
    model = _build_model(split_config, vocab, activity_to_idx, num_rels, device)

    class_weights = None
    if split_config["training"]["loss"]["ifw"]:
        class_weight_alpha = float(
            split_config["training"]["loss"].get("class_weight_alpha", 0.5)
        )
        class_weights = compute_class_weights(
            train_dataset, activity_to_idx, alpha=class_weight_alpha
        )
        print(class_weights)

    opt = build_optimizer(model, split_config["training"])
    scheduler = build_scheduler(opt, split_config["training"])

    num_epochs = split_config["training"]["num_epochs"]
    hidden_size = lstm_cfg.get("hidden_size", attention_pool_cfg["final_graph_emb_dim"])
    model_tag = str(split_config.get("model_type", "lstm")).lower()
    save_path = os.path.join(
        run_root,
        f"split_{split_id}",
        (
            f"{model_tag}_dino_model_fc_layer_{mlp_cfg['fc_layers_num']}_num_epoch_{num_epochs}"
            f"_graph_emb_dim_{projector_cfg['graph_emb_dim']}"
            f"_lstm_hidden_{hidden_size}"
            f"_final_graph_emb_dim_{attention_pool_cfg['final_graph_emb_dim']}"
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    save_json(os.path.join(save_path, "experiment_config.json"), {**split_config, "split_stats": split_stats})
    save_json(os.path.join(save_path, "class_mapping.json"), activity_to_idx)

    results = {}
    epoch_preds = {k: {} for k in range(num_epochs)}
    best_epoch_result = {"mean_accuracy": -1, "top1": -1, "top5": -1, "f1": -1}
    global_step = 0

    for epoch in tqdm(
        range(num_epochs),
        desc=f"Training split {split_id} for {num_epochs} epochs",
        unit="epoch",
    ):
        print(f"Epoch {epoch + 1}/{num_epochs}\n")
        loss_func = build_loss_fn(
            split_config["training"]["loss"],
            class_weights,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        epoch_result, global_step, preds, targets = do_epoch(
            device=device,
            net=model,
            opt=opt,
            train_loader=train_loader,
            validate_loader=val_loader,
            global_step=global_step,
            num_classes_train=len(activity_to_idx),
            num_classes_val=len(activity_to_idx),
            loss_func=loss_func,
            train_progress_desc=(
                f"Split {split_id} epoch {epoch + 1}/{num_epochs} train"
            ),
            val_progress_desc=f"Split {split_id} epoch {epoch + 1}/{num_epochs} val",
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
            f"Mean Acc. {val_metrics['mean_accuracy'] * 100:.2f}% | "
            f"Top-1 {val_metrics['top1'] * 100:.2f}% | "
            f"Top-5 {val_metrics['top5'] * 100:.2f}% | "
            f"Avg. F1 {val_metrics['avg_f1'] * 100:.2f}%"
        )

        if val_metrics["mean_accuracy"] > best_epoch_result["mean_accuracy"]:
            best_epoch_result["mean_accuracy"] = val_metrics["mean_accuracy"]
            store_model(model, opt, epoch, save_path, metric="mean_accuracy")
            print(
                "New best Mean Accuracy model saved: "
                f"{val_metrics['mean_accuracy'] * 100:.2f}%"
            )
        if val_metrics["top1"] > best_epoch_result["top1"]:
            best_epoch_result["top1"] = val_metrics["top1"]
            store_model(model, opt, epoch, save_path, metric="top1")
        if val_metrics["top5"] > best_epoch_result["top5"]:
            best_epoch_result["top5"] = val_metrics["top5"]
            store_model(model, opt, epoch, save_path, metric="top5")
            print(f"New best Top-5 model saved: {val_metrics['top5'] * 100:.2f}%")
        if val_metrics["f1"] > best_epoch_result["f1"]:
            best_epoch_result["f1"] = val_metrics["f1"]
            store_model(model, opt, epoch, save_path, metric="f1")

        results[epoch] = epoch_result
        if scheduler is not None:
            scheduler.step()

    write_training_outputs(save_path, results, epoch_preds)

    checkpoint_summaries = {}
    for metric in ("mean_accuracy", "top5", "f1"):
        checkpoint_summaries[metric] = evaluate_checkpoint(
            model, save_path, metric, test_loader, test_dataset, device
        )
        save_json(
            os.path.join(save_path, f"final_test_results_{metric}.json"),
            checkpoint_summaries[metric],
        )

    final_checkpoint_metric = split_config["training"].get(
        "final_checkpoint_metric", "mean_accuracy"
    )
    if final_checkpoint_metric not in checkpoint_summaries:
        raise ValueError(
            "training.final_checkpoint_metric must be one of: mean_accuracy, top5, f1"
        )
    final_summary = checkpoint_summaries[final_checkpoint_metric]
    save_json(os.path.join(save_path, "final_test_results.json"), final_summary)

    print(f"Best validation Mean Accuracy: {best_epoch_result['mean_accuracy'] * 100:.2f}%")
    print(f"Best validation Top-1 accuracy: {best_epoch_result['top1'] * 100:.2f}%")
    print(f"Best validation Top-5 accuracy: {best_epoch_result['top5'] * 100:.2f}%")
    print(f"Best validation F1: {best_epoch_result['f1'] * 100:.2f}%")
    _print_split_metrics(split_id, final_summary)

    return {
        "split_id": int(split_id),
        "save_path": save_path,
        "best_validation": float_metrics(best_epoch_result),
        "final_checkpoint_metric": final_checkpoint_metric,
        "test": final_summary,
    }


def resolve_split_ids(args, config):
    if args.split_ids:
        return [
            int(split_id)
            for split_id in args.split_ids.split(",")
            if split_id.strip()
        ]
    data_cfg = config["data"]
    if "split_ids" in data_cfg:
        return [int(split_id) for split_id in data_cfg["split_ids"]]
    return [int(data_cfg.get("split_id", 1))]


def write_cross_split_summary(run_root, split_results):
    metrics = ("mean_accuracy", "top1", "top5", "f1")
    per_split = []
    for result in split_results:
        test_metrics = result["test"]["metrics"]
        per_split.append(
            {
                "split_id": result["split_id"],
                "checkpoint_metric": result["final_checkpoint_metric"],
                "mean_accuracy": test_metrics["mean_accuracy"],
                "top1": test_metrics["top1"],
                "top5": test_metrics["top5"],
                "f1": test_metrics["f1"],
                "save_path": result["save_path"],
            }
        )

    averages = {
        metric: sum(split[metric] for split in per_split) / len(per_split)
        for metric in metrics
    }
    summary = {"per_split": per_split, "average": averages}
    save_json(os.path.join(run_root, "cross_split_summary.json"), summary)

    print("\nEGTEA cross-split summary")
    for split in per_split:
        print(
            f"Split {split['split_id']}: "
            f"Mean Acc. {split['mean_accuracy'] * 100:.2f}% | "
            f"Top-1 {split['top1'] * 100:.2f}% | "
            f"Top-5 {split['top5'] * 100:.2f}% | "
            f"F1 {split['f1'] * 100:.2f}%"
        )
    print(
        "Average: "
        f"Mean Acc. {averages['mean_accuracy'] * 100:.2f}% | "
        f"Top-1 {averages['top1'] * 100:.2f}% | "
        f"Top-5 {averages['top5'] * 100:.2f}% | "
        f"F1 {averages['f1'] * 100:.2f}%"
    )
    return summary


def main(args, config):
    apply_cli_overrides(args, config)
    split_ids = resolve_split_ids(args, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(
        config["output"]["base_path"],
        config["experiment_name"],
        f"egtea_lstm_cross_split_{timestamp}",
    )
    os.makedirs(run_root, exist_ok=True)
    save_json(os.path.join(run_root, "base_config.json"), config)

    print(f"Resolved EGTEA split ids: {split_ids}")
    print(f"Run root: {run_root}")

    split_results = [
        train_one_split(config, split_id=split_id, run_root=run_root)
        for split_id in split_ids
    ]
    write_cross_split_summary(run_root, split_results)


def apply_cli_overrides(args, config):
    if args.num_graphs is not None:
        config["mlp"]["num_graphs"] = int(args.num_graphs)
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = int(args.num_epochs)
    if args.graph_type is not None:
        config["data"]["graph_type"] = args.graph_type
    if args.batch_size is not None:
        config["data"]["batch_size"] = int(args.batch_size)
    if args.val_num_samples is not None:
        config["data"]["val_num_samples"] = int(args.val_num_samples)
    if getattr(args, "val_distribution_source", None) is not None:
        config["data"]["val_distribution_source"] = args.val_distribution_source
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.output_base_path is not None:
        config["output"]["base_path"] = args.output_base_path
    if args.easg_cache_path is not None:
        config["data"]["easg_cache_path"] = args.easg_cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the experiment config JSON file",
    )
    parser.add_argument(
        "--split-ids",
        type=str,
        default=None,
        help="Comma-separated EGTEA split ids. Overrides data.split_ids in the config.",
    )
    parser.add_argument("--num-graphs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--graph-type", choices=("full", "pruned"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-num-samples", type=int, default=None)
    parser.add_argument(
        "--val-distribution-source",
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-base-path", type=str, default=None)
    parser.add_argument("--easg-cache-path", type=str, default=None)
    parsed_args = parser.parse_args()
    with open(parsed_args.config_path, "r") as f:
        loaded_config = json.load(f)
    main(parsed_args, loaded_config)

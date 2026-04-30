import copy
import json
import os

LOSS_CONFIGS = {
    "cross_entropy": {"name": "cross_entropy", "ifw": False, "focal_gamma": 2.0},
    "annealed_focal": {
        "name": "focal_loss_annealed",
        "ifw": False,
        "focal_gamma": 2.0,
        "focal_gamma_start": 2.0,
        "focal_gamma_end": 0.1,
    },
    "regular_focal": {"name": "focal_loss", "ifw": False, "focal_gamma": 2.0},
    "ifw_cross_entropy_reduced": {
        "name": "cross_entropy",
        "ifw": True,
        "focal_gamma": 2.0,
        "class_weight_alpha": 0.25,
    },
}

NEXT_ACTION_LOSSES = ("cross_entropy", "annealed_focal")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def selected_run_config(ablation_config, graph_count=None, loss_name=None):
    selected = load_json(ablation_config["selected_config_path"])
    mlp_config = copy.deepcopy(selected["mlp"])
    if graph_count is not None:
        mlp_config["num_graphs"] = graph_count

    data_config = copy.deepcopy(selected["data"])
    data_config["batch_size"] = selected.get("batch_size", data_config["batch_size"])

    training_config = {
        "num_epochs": selected["num_epochs"],
        "optimizer": selected["optimizer"],
        "learning_rate": selected["learning_rate"],
        "weight_decay": selected["weight_decay"],
        "scheduler_factor": selected["scheduler_factor"],
        "criterion_metrics": selected.get("criterion_metrics", ["f1", "acc"]),
        "loss": copy.deepcopy(
            LOSS_CONFIGS[loss_name] if loss_name else selected["loss"]
        ),
        "sampler": copy.deepcopy(selected.get("sampler", {"name": "none"})),
        "final_checkpoint_metric": ablation_config.get("objective_metric", "top5"),
    }

    config = {
        "experiment_name": ablation_name(
            ablation_config, selected, graph_count, loss_name
        ),
        "mlp": mlp_config,
        "data": data_config,
        "training": training_config,
        "output": {"base_path": ablation_config["output_base_path"]},
        "device": selected.get("device", "cuda"),
    }
    if "lstm" in selected:
        config["lstm"] = copy.deepcopy(selected["lstm"])
    return config


def source_run_config(ablation_config, graph_count=None, loss_name=None, num_epochs=None):
    source = load_json(ablation_config["source_config_path"])
    config = copy.deepcopy(source)

    if graph_count is not None:
        config["mlp"]["num_graphs"] = graph_count
    if num_epochs is not None:
        config["training"]["num_epochs"] = num_epochs
    if loss_name is not None:
        config["training"]["loss"] = copy.deepcopy(LOSS_CONFIGS[loss_name])

    config["training"]["final_checkpoint_metric"] = ablation_config.get(
        "objective_metric", "top5"
    )
    config["output"]["base_path"] = ablation_config["output_base_path"]
    config["experiment_name"] = ablation_name(
        ablation_config, source, graph_count, loss_name
    )
    return config


def ablation_name(ablation_config, source_config, graph_count=None, loss_name=None):
    name = ablation_config.get(
        "experiment_name_prefix", source_config["experiment_name"]
    )
    if graph_count is not None:
        name = f"{name}_graphs_{graph_count}"
    if loss_name is not None:
        name = f"{name}_{loss_name}"
    return name


def run_ablation(config, ablation_config):
    os.makedirs(config["output"]["base_path"], exist_ok=True)
    if "notes" in ablation_config:
        print(f"Config notes: {ablation_config['notes']}")

    train_one_run = trainer_for(ablation_config["model_type"])
    return train_one_run(
        config,
        trial=None,
        run_test=ablation_config.get("run_test", True),
        objective_metric=ablation_config.get("objective_metric", "top5"),
    )


def trainer_for(model_type):
    if model_type == "lstm":
        from train_graph_lstm_meccano_optuna import train_one_run as train_lstm_one_run

        return train_lstm_one_run
    if model_type == "mlp":
        from train_graph_mlp_meccano_optuna import train_one_run as train_mlp_one_run

        return train_mlp_one_run

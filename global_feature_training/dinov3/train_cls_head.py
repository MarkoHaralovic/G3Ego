import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from global_feature_training.base_model.sequence_model import ImageSequenceClassificator
from egocentric_video_graph_framework_ar.global_feature_training.data_loading.dataset_split_meccano import return_train_val_samples
from global_feature_training.data_loading.sequence_dataset import (
    SequenceDataset,
    feature_collate_fn,
)
from global_feature_training.dinov3.dinov3_feature_extraction import (
    get_clip_text_encoder,
    get_dinov3_extractor,
)
from global_feature_training.train.evaluate import store_model
from global_feature_training.train.train import do_epoch

config_path = "/home/s3758869/egocentric_video_graph_framework_ar/global_feature_training/dinov3/configs/run_config.json"
with open(config_path, "r") as f:
    config = json.load(f)

experiment_name = config["experiment_name"]
print(f"Running experiment: {experiment_name}")

USE_PRECOMPUTED_FEATURES = config["model"]["use_precomputed_features"]
DINOV3_PRECOMUPTED_FEATS_FOLDER = config["data"]["precomputed_features_folder"]
dinov3_model_name = config["model"]["name"]
dinov3_model_path = config["model"]["dinov3_model_path"]
clip_model_path = config["model"]["clip_model_path"]
dino_cache_dir = config["model"]["dino_cache_dir"]
clip_cache_dir = config["model"]["clip_cache_dir"]
device = config["device"]

num_epochs = config["training"]["num_epochs"]
fc_layers_num = config["training"]["fc_layers_num"]
optimizer_name = config["training"]["optimizer"]
learning_rate = config["training"]["learning_rate"]
weight_decay = config["training"]["weight_decay"]
scheduler_factor = config["training"]["scheduler_factor"]
criterion_metrics = config["training"]["criterion_metrics"]
batch_size = config["data"]["batch_size"]
num_workers = config["data"]["num_workers"]
use_text_features = config["model"]["use_text_features"]

# data_split_path = "..."  # alternative way to load the data, location based split
# with open(data_split_path,'r') as f:
#     data_split = json.load(f)

pooling = config["model"]["pooling"]

train_samples, val_samples, activity_to_idx = return_train_val_samples(pooling=pooling)

print(f"activity_to_idx : {activity_to_idx}")
print(f"len(train_samples) : {len(train_samples)}")
print(f"len(val_samples) : {len(val_samples)}")

train_dataset = SequenceDataset(
    input_folder=DINOV3_PRECOMUPTED_FEATS_FOLDER,
    clip_names=None,  # data_split["split_summary"]["train"],
    samples=train_samples,
    model_name=dinov3_model_name,
    pooling=pooling,
    load_visual=True,
    load_text=False,
    activity_to_idx=activity_to_idx,
)

validation_dataset = SequenceDataset(
    input_folder=DINOV3_PRECOMUPTED_FEATS_FOLDER,
    clip_names=None,  # data_split["split_summary"]["validation"],
    samples=val_samples,
    model_name=dinov3_model_name,
    pooling=pooling,
    load_visual=True,
    load_text=False,
    activity_to_idx=activity_to_idx,
)

assert train_dataset.activity_to_idx == validation_dataset.activity_to_idx
cls_mapping = train_dataset.activity_to_idx

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=feature_collate_fn,
)

val_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=feature_collate_fn,
)

if not USE_PRECOMPUTED_FEATURES:
    dino_backbone = get_dinov3_extractor(dinov3_model_path, cache_dir=dino_cache_dir)
    tokenizer, clip_text_encoder, _ = get_clip_text_encoder(
        clip_model_path, clip_cache_dir
    )
else:
    dino_backbone = None
    clip_text_encoder = None

linear_layer_input_dim = config["model"]["linear_layer_input_dim"]

model = ImageSequenceClassificator(
    vision_backbone=dino_backbone,
    text_backbone=clip_text_encoder,
    n_classes=len(cls_mapping),
    linear_layer_input_dim=linear_layer_input_dim,
    pooling="average",
    fc_layers_num=fc_layers_num,
    use_precomputed_features=USE_PRECOMPUTED_FEATURES,
    transforms=None,
    device=device,
).to(device)

model.train()
trainable_params = model.get_trainable_params()

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
    f"dino_model_fc_layer_{fc_layers_num}_num_epoch_{num_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
os.makedirs(save_path, exist_ok=True)

results = {}
best_epoch_result = {"acc": -1, "f1": -1}
global_step = 0

experiment_config = {
    "experiment_name": experiment_name,
    "model": dinov3_model_name,
    "fc_layers_num": fc_layers_num,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "optimizer": optimizer_name,
    "scheduler_factor": scheduler_factor,
    "num_classes": len(cls_mapping),
    "use_precomputed_features": USE_PRECOMPUTED_FEATURES,
    "use_text_features": use_text_features,
    "device": device,
    "linear_layer_input_dim": linear_layer_input_dim,
}

with open(os.path.join(save_path, "experiment_config.json"), "w") as f:
    json.dump(experiment_config, f, indent=2)

with open(os.path.join(save_path, "class_mapping.json"), "w") as f:
    json.dump(cls_mapping, f, indent=2)

for epoch in tqdm(
    range(num_epochs),
    desc=f"Training for {num_epochs} epochs",
    unit="epoch",
    total=num_epochs,
):
    print(f"Epoch {epoch+1}/{num_epochs}\n")

    epoch_result, global_step = do_epoch(
        device=device,
        net=model,
        opt=opt,
        train_loader=train_loader,
        validate_loader=val_loader,
        global_step=global_step,
        num_classes_train=len(cls_mapping),
        num_classes_val=len(validation_dataset.activity_to_idx),
        text_features=use_text_features,
    )

    val_metrics = epoch_result["val"]["eval_metrics"]
    train_metrics = epoch_result["train"]["eval_metrics"]

    print(
        f"\nValidation - Accuracy: {val_metrics['acc']*100:.2f}%, F1: {val_metrics['f1']*100:.2f}%"
    )

    if val_metrics["acc"] > best_epoch_result["acc"]:
        best_epoch_result["acc"] = val_metrics["acc"]
        store_model(net=model, opt=opt, epoch=epoch, save_path=save_path, metric="acc")
        print(f"New best accuracy model saved: {val_metrics['acc']*100:.2f}%")

    if val_metrics["f1"] > best_epoch_result["f1"]:
        best_epoch_result["f1"] = val_metrics["f1"]
        store_model(net=model, opt=opt, epoch=epoch, save_path=save_path, metric="f1")
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

print(f"Best validation accuracy: {best_epoch_result['acc']*100:.2f}%")
print(f"Best validation F1: {best_epoch_result['f1']*100:.2f}%")

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def evaluate(net, data_loader, device, num_classes):
    net.eval()

    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for _, data_dict in enumerate(data_loader):
            targets = data_dict["activity_label"].to(device)
            graphs = data_dict["full_action_graphs"]

            output = net(graphs)

            logits = output.data
            pred = logits.max(1)[1]

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_pred_np = np.array(all_preds)
    y_true_np = np.array(all_targets)
    y_score_np = np.concatenate(all_logits, axis=0) if all_logits else None

    eval_metrics, conf_mat = evaluation_metrics(
        y_pred_np, y_true_np, num_classes, y_score=y_score_np
    )

    epoch_result = {}
    epoch_result["eval_metrics"] = eval_metrics
    epoch_result["conf_mat"] = conf_mat

    return epoch_result, y_pred_np, y_true_np


def evaluation_metrics(y_pred, y_true, num_classes, y_score=None):

    confusion_matrix = metrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=tuple(range(num_classes))
    )

    top1 = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    if y_score is not None:
        k = min(5, num_classes)
        top5 = metrics.top_k_accuracy_score(
            y_true=y_true,
            y_score=y_score,
            k=k,
            labels=tuple(range(num_classes)),
        )
    else:
        top5 = top1

    results = {
        "top1": top1,
        "top5": top5,
        "avg_precision": metrics.precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            zero_division=0,
        ),
        "avg_recall": metrics.recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            zero_division=0,
        ),
        "avg_f1": metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            zero_division=0,
        ),
        "acc": top1,
        "f1": metrics.f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    return results, confusion_matrix


def store_model(net, opt, epoch, save_path, metric="f1"):
    for f in os.listdir(save_path):
        if f.startswith(f"best_train_model_{metric}_") and f.endswith(".pt"):
            os.remove(os.path.join(save_path, f))
        if f.startswith(f"best_model_{metric}_") and f.endswith(".pt"):
            os.remove(os.path.join(save_path, f))

    file_name = f"best_model_{metric}_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "metric": metric,
        },
        os.path.join(save_path, file_name),
    )


def compute_class_weights(train_dataset, activity_to_idx, alpha = 0.5 ):
    counts = torch.zeros(len(activity_to_idx), dtype=torch.float)
    for _, _, label_str, *_ in train_dataset.sample_index:
        counts[activity_to_idx[label_str]] += 1
    total = counts.sum()

    weights = torch.zeros_like(counts)
    weights = total / (len(activity_to_idx) * counts)
    weights = alpha * weights + (1 - alpha) * torch.ones_like(weights)
    return np.sqrt(weights)


def build_loss_fn(loss_cfg, class_weights, epoch=None, num_epochs=None):
    name = loss_cfg["name"]
    gamma = float(loss_cfg.get("focal_gamma", 2.0))

    if name == "focal_loss_annealed":
        gamma_start = float(loss_cfg.get("focal_gamma_start", 2.0))
        gamma_end = float(loss_cfg.get("focal_gamma_end", 0.1))
        if epoch is None or num_epochs is None or num_epochs <= 1:
            gamma = gamma_start
        else:
            progress = min(max(epoch / float(num_epochs - 1), 0.0), 1.0)
            gamma = gamma_start * ((gamma_end / gamma_start) ** progress)

    def ce_loss(logits, targets):
        weight = class_weights.to(logits.device) if class_weights is not None else None
        return F.cross_entropy(logits, targets, weight=weight)

    def focal_loss(logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = class_weights.to(logits.device) if class_weights is not None else None
        alpha_factor = alpha[targets] if alpha is not None else 1.0
        focal_factor = (1.0 - pt) ** gamma
        loss = (
            -alpha_factor
            * focal_factor
            * logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        )
        return loss.mean()

    if name == "cross_entropy":
        return ce_loss
    if name in {"focal_loss", "focal_loss_annealed"}:
        return focal_loss
    raise ValueError(f"Unsupported loss function: {name}")

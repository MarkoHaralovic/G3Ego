import numpy as np
from tqdm import tqdm
from .evaluate import evaluate, evaluation_metrics


def train(
    net,
    optimizer,
    data_loader,
    device,
    global_step,
    num_classes,
    loss_func,
    progress_desc="Train",
):
    net.train()

    all_preds = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(
        data_loader,
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch_idx, data_dict in enumerate(progress):
        targets = data_dict["activity_label"].to(device)
        graphs = data_dict["full_action_graphs"]

        optimizer.zero_grad()
        output = net(graphs)

        loss = loss_func(output, targets)
        logits = output.data
        pred = logits.max(1)[1]

        y_pred_np = pred.cpu().numpy()
        y_true_np = targets.cpu().numpy()

        all_preds.extend(y_pred_np)
        all_targets.extend(y_true_np)
        all_logits.append(logits.cpu().numpy())

        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
        running_correct += pred.eq(targets).sum().item()
        running_total += targets.size(0)

        avg_loss = total_loss / (batch_idx + 1)
        running_acc = running_correct / running_total if running_total else 0.0
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}",
            acc=f"{running_acc * 100:.2f}%",
        )

    y_pred_all = np.array(all_preds)
    y_true_all = np.array(all_targets)
    y_score_all = np.concatenate(all_logits, axis=0) if all_logits else None

    eval_metrics, conf_mat = evaluation_metrics(
        y_pred_all, y_true_all, num_classes, y_score=y_score_all
    )

    epoch_result = {}
    epoch_result["eval_metrics"] = eval_metrics
    epoch_result["conf_mat"] = conf_mat

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    print(f"Training average Loss: {avg_loss:.4f}")
    print(
        "Train metrics : "
        f"Mean Acc. {eval_metrics['mean_accuracy']*100:.2f}% | "
        f"Accuracy {eval_metrics['acc']*100:.2f}% | "
        f"f1-score {eval_metrics['f1']*100:.2f}% | "
        f"Top-1 {eval_metrics['top1']*100:.2f}% | "
        f"Top-5 {eval_metrics['top5']*100:.2f}% | "
        f"Avg. Prec. {eval_metrics['avg_precision']*100:.2f}% | "
        f"Avg. Recall {eval_metrics['avg_recall']*100:.2f}% | "
        f"Avg. F1 {eval_metrics['avg_f1']*100:.2f}%"
    )

    return global_step, epoch_result


def do_epoch(
    device,
    net,
    opt,
    train_loader,
    validate_loader,
    global_step,
    num_classes_train,
    num_classes_val,
    loss_func,
    train_progress_desc="Train",
    val_progress_desc="Val",
):
    global_step, train_epoch_result = train(
        net,
        opt,
        train_loader,
        device,
        global_step=global_step,
        num_classes=num_classes_train,
        loss_func=loss_func,
        progress_desc=train_progress_desc,
    )

    opt.zero_grad()

    val_epoch_result, preds, targets = evaluate(
        net,
        validate_loader,
        device,
        num_classes=num_classes_val,
        progress_desc=val_progress_desc,
    )

    epoch_result = {}
    epoch_result["train"] = train_epoch_result
    epoch_result["val"] = val_epoch_result

    return epoch_result, global_step, preds, targets

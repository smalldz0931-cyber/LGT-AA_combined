import os
import json
import time
import math
import argparse
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from dataloader_semmax import DataManager
from model import ModelWiseTransformerClassifierNew


EN_LABELS = {
    "Baichuan": 0,
    "gpt_neo": 1,
    "gpt2_xl": 2,
    "llama3": 3,
    "mistral": 4,
    "opt": 5,
    "PULI": 6,
    "gpt3.5": 7,
    "gpt4": 8,
    "human": 9,
    "claude": 10,
    "deepseek_R1": 11,
    "Qwen_plus": 12,
    "doubao_seed": 13,
}
HUMAN_LABEL_NAME = "human"


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construct_bmes_labels(labels_dict):
    prefix = ["B-", "M-", "E-", "S-"]
    id2label = {}
    counter = 0
    for label_name in labels_dict.keys():
        for pre in prefix:
            id2label[counter] = pre + label_name
            counter += 1
    return id2label


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def append_jsonl(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")       

def build_model_arch_config(
    arch_variant: str,
    cnn_out_dim: int,
    nhead_override: Optional[int] = None,
    num_layers_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一管理训练时的模型结构配置。

    legacy:
      - 分布分支：单层 Conv1d
      - 语义分支：Linear 后直接 broadcast
      - 上下文：2 层 Transformer，8 头

    schemeA:
      - 分布分支：3 层 CNN，kernel=(5,3,3), channels=(64,128,64)
      - 语义分支：Linear -> 1 层轻量 Conv1d
      - 上下文：2 层 Transformer，8 头
      
    schemeB:
      - 分布分支：5 层 CNN，kernel=(5,3,3,3,3), channels=(64,128,128,128,64)
      - 语义分支：Linear -> 1 层轻量 Conv1d
      - 上下文：2 层 Transformer，16 头
    """
    if arch_variant == "legacy":
        cfg = {
            "arch_variant": "legacy",
            "dist_channels": None,
            "dist_kernel_sizes": None,
            "use_semantic_conv": False,
            "sem_conv_channels": None,
            "sem_conv_kernel_sizes": None,
            "nhead": 8,
            "num_layers": 2,
        }
    elif arch_variant == "schemeA":
        cfg = {
            "arch_variant": "schemeA",
            "dist_channels": [cnn_out_dim, 2 * cnn_out_dim, cnn_out_dim],
            "dist_kernel_sizes": [5, 3, 3],
            "use_semantic_conv": True,
            "sem_conv_channels": [cnn_out_dim],
            "sem_conv_kernel_sizes": [3],
            "nhead": 8,
            "num_layers": 2,
        }
        
    elif arch_variant == "schemeB":
        cfg = {
            "arch_variant": "schemeB",
            "dist_channels": [cnn_out_dim, 2 * cnn_out_dim, 2 * cnn_out_dim, 2 * cnn_out_dim, cnn_out_dim],
            "dist_kernel_sizes": [5, 3, 3, 3, 3],
            "use_semantic_conv": True,
            "sem_conv_channels": [cnn_out_dim],
            "sem_conv_kernel_sizes": [3],
            "nhead": 16,
            "num_layers": 2,
        }
    else:
        raise ValueError(f"Unknown arch_variant: {arch_variant}")

    if nhead_override is not None:
        cfg["nhead"] = nhead_override
    if num_layers_override is not None:
        cfg["num_layers"] = num_layers_override
    return cfg


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    best_metric: float,
    model_config: Optional[Dict[str, Any]] = None,
):
    ensure_dir(save_dir)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
    }
    torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1.0, float(num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / max(1.0, float(num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_token_acc(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        feats = batch["features"].to(device)
        labels = batch["labels"].to(device)
        sem = batch["sem_features"]

        out = model(feats, labels=None, sem_features=sem)
        logits = out["logits"]
        preds = torch.argmax(logits, dim=-1)

        mask = labels != -1
        correct += int((preds[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())

    acc = correct / max(1, total)
    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2, help="梯度累积步数：等效batch = batch_size * grad_accum")
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--num_models", type=int, default=4, help="使用多少个 extractor 通道（1/2/3/4）")
    parser.add_argument("--sem_dim", type=int, default=4096)
    parser.add_argument("--cnn_out_dim", type=int, default=64)

    parser.add_argument("--arch_variant", type=str, default="schemeA", choices=["legacy", "schemeA", "schemeB"])
    parser.add_argument("--nhead", type=int, default=None, help="可选覆盖默认注意力头数")
    parser.add_argument("--num_layers", type=int, default=None, help="可选覆盖默认 Transformer 层数")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--log_file",
        type=str,
        default="/root/autodl-tmp/train/train_log.jsonl",
        help="训练日志输出 jsonl 路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/autodl-tmp/train/checkpoints",
        help="checkpoint 保存目录",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = construct_bmes_labels(EN_LABELS)

    data = DataManager(
        train_path=args.train_path,
        test_path=args.test_path,
        batch_size=args.batch_size,
        max_len=args.seq_len,
        human_label=HUMAN_LABEL_NAME,
        id2label=id2label,
    )

    arch_config = build_model_arch_config(
        arch_variant=args.arch_variant,
        cnn_out_dim=args.cnn_out_dim,
        nhead_override=args.nhead,
        num_layers_override=args.num_layers,
    )

    model = ModelWiseTransformerClassifierNew(
        id2labels=id2label,
        seq_len=args.seq_len,
        num_models=args.num_models,
        cnn_out_dim=args.cnn_out_dim,
        sem_dim=args.sem_dim,
        dropout=0.1,
        ignore_index=-1,
        dist_channels=arch_config["dist_channels"],
        dist_kernel_sizes=arch_config["dist_kernel_sizes"],
        use_semantic_conv=arch_config["use_semantic_conv"],
        sem_conv_channels=arch_config["sem_conv_channels"],
        sem_conv_kernel_sizes=arch_config["sem_conv_kernel_sizes"],
        nhead=arch_config["nhead"],
        num_layers=arch_config["num_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(data.train_dataloader) / max(1, args.grad_accum))
    total_train_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)

    best_acc = -1.0
    global_step = 0

    model_config_to_save = model.get_model_config()
    model_config_to_save["arch_variant"] = arch_config["arch_variant"]

    append_jsonl(
        args.log_file,
        {
            "time": now_ts(),
            "event": "train_start",
            "train_path": args.train_path,
            "test_path": args.test_path,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "seq_len": args.seq_len,
            "epochs": args.num_train_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_train_steps": total_train_steps,
            "num_models": args.num_models,
            "sem_dim": args.sem_dim,
            "cnn_out_dim": args.cnn_out_dim,
            "seed": args.seed,
            "model_config": model_config_to_save,
        },
    )

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        running_loss = 0.0
        seen_steps = 0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(data.train_dataloader, desc=f"Epoch {epoch}/{args.num_train_epochs}")
        for step, batch in enumerate(pbar, start=1):
            feats = batch["features"].to(device)
            labels = batch["labels"].to(device)
            sem = batch["sem_features"]

            out = model(feats, labels=labels, sem_features=sem)
            loss = out["loss"]
            loss = loss / max(1, args.grad_accum)
            loss.backward()

            running_loss += float(loss.item())
            seen_steps += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            cur_lr = scheduler.get_last_lr()[0] if global_step > 0 else args.lr
            pbar.set_postfix({"loss": f"{(running_loss / max(1, seen_steps)):.4f}", "lr": f"{cur_lr:.2e}"})

        avg_loss = running_loss / max(1, seen_steps)
        eval_acc = evaluate_token_acc(model, data.test_dataloader, device)
        # ===== Gate统计输出 =====
        gate_stats = model.get_and_reset_gate_stats()
        if gate_stats is not None:
            print(
                f"[Gate Epoch {epoch}] "
                f"mean={gate_stats['mean']:.4f} "
                f"std={gate_stats['std']:.4f} "
                f"min={gate_stats['min']:.4f} "
                f"max={gate_stats['max']:.4f}"
            )

            append_jsonl(args.log_file, {
                "type": "gate_epoch",
                "epoch": epoch,
                **gate_stats
            })

        record = {
            "time": now_ts(),
            "epoch": epoch,
            "global_step": global_step,
            "train_avg_loss": avg_loss,
            "eval_token_acc": eval_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        append_jsonl(args.log_file, record)

        print(f"\n✅ Epoch {epoch} done. avg_loss={avg_loss:.6f} | eval_token_acc={eval_acc:.4f}\n")

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(
                args.save_dir,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_acc,
                model_config=model_config_to_save,
            )
            append_jsonl(
                args.log_file,
                {
                    "time": now_ts(),
                    "event": "best_checkpoint_saved",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_eval_token_acc": best_acc,
                    "model_config": model_config_to_save,
                },
            )
            print(f"💾 Best checkpoint saved! best_eval_token_acc={best_acc:.4f}\n")

    append_jsonl(
        args.log_file,
        {
            "time": now_ts(),
            "event": "train_end",
            "best_eval_token_acc": best_acc,
            "model_config": model_config_to_save,
        },
    )
    print(f"🎉 Training finished. Best eval_token_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()

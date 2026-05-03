import json
from collections import Counter, defaultdict
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


def construct_bmes_labels(labels_dict):
    prefix = ["B-", "M-", "E-", "S-"]
    id2label = {}
    counter = 0
    for label_name in labels_dict.keys():
        for pre in prefix:
            id2label[counter] = pre + label_name
            counter += 1
    return id2label


def strip_bmes(tag: str) -> str:
    if len(tag) >= 2 and tag[1] == "-":
        return tag[2:]
    if len(tag) >= 3 and tag[2] == "-":
        return tag[3:]
    return tag.split("-", 1)[-1]


def safe_div(a, b):
    return a / b if b else 0.0


def compute_prf1_per_class(y_true, y_pred, labels):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    out = {}
    f1s = []
    for lab in labels:
        p_score = safe_div(tp[lab], tp[lab] + fp[lab])
        r_score = safe_div(tp[lab], tp[lab] + fn[lab])
        f1 = safe_div(2 * p_score * r_score, p_score + r_score) if (p_score + r_score) else 0.0
        out[lab] = {"P": p_score, "R": r_score, "F1": f1}
        f1s.append(f1)

    macro_f1 = sum(f1s) / max(1, len(f1s))
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
    return out, acc, macro_f1


def infer_num_models_from_ckpt(state_dict, cnn_out_dim=64):
    max_dist = -1
    max_sem = -1
    for k in state_dict.keys():
        if k.startswith("dist_cnns."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_dist = max(max_dist, int(parts[1]))
        if k.startswith("sem_projs."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_sem = max(max_sem, int(parts[1]))

    cand = None
    if max_dist >= 0:
        cand = max_dist + 1
    if max_sem >= 0:
        cand2 = max_sem + 1
        cand = cand2 if cand is None else max(cand, cand2)

    w = state_dict.get("classifier.weight", None)
    if w is not None:
        in_dim = w.shape[1]
        if in_dim % cnn_out_dim == 0:
            cand3 = in_dim // cnn_out_dim
            cand = cand3 if cand is None else cand3

    if cand is None:
        raise ValueError("Cannot infer num_models from checkpoint. Please pass --num_models explicitly.")
    return int(cand)


def build_model_arch_config(
    arch_variant: str,
    cnn_out_dim: int,
    nhead_override: Optional[int] = None,
    num_layers_override: Optional[int] = None,
) -> Dict[str, Any]:
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
    else:
        raise ValueError(f"Unknown arch_variant: {arch_variant}")

    if nhead_override is not None:
        cfg["nhead"] = nhead_override
    if num_layers_override is not None:
        cfg["num_layers"] = num_layers_override
    return cfg


def infer_model_arch_config_from_state(state_dict, cnn_out_dim: int) -> Dict[str, Any]:
    """
    兼容旧 checkpoint：
    - 旧版单层 dist CNN 的 key 类似 dist_cnns.0.conv.weight
    - 方案 A 多层 dist CNN 的 key 类似 dist_cnns.0.layers.0.conv.weight
    - 方案 A 语义卷积 key 类似 sem_cnns.0.layers.0.conv.weight
    """
    has_stacked_dist = any(k.startswith("dist_cnns.") and ".layers." in k for k in state_dict.keys())
    has_semantic_conv = any(k.startswith("sem_cnns.") for k in state_dict.keys())

    if has_stacked_dist:
        cfg = build_model_arch_config("schemeA", cnn_out_dim=cnn_out_dim)
        if not has_semantic_conv:
            cfg["use_semantic_conv"] = False
            cfg["sem_conv_channels"] = None
            cfg["sem_conv_kernel_sizes"] = None
        return cfg

    return build_model_arch_config("legacy", cnn_out_dim=cnn_out_dim)


@torch.no_grad()
def main(
    train_path,
    test_path,
    ckpt_path,
    batch_size=16,
    seq_len=1024,
    sem_dim=4096,
    num_models=None,
    cnn_out_dim=64,
    arch_variant=None,
    nhead=None,
    num_layers=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label = construct_bmes_labels(EN_LABELS)

    data = DataManager(
        train_path=train_path,
        test_path=test_path,
        batch_size=batch_size,
        max_len=seq_len,
        human_label=HUMAN_LABEL_NAME,
        id2label=id2label,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"]

    if num_models is None:
        num_models = infer_num_models_from_ckpt(state, cnn_out_dim=cnn_out_dim)
        print(f"[Auto] inferred num_models={num_models} from ckpt.")

    if arch_variant is not None:
        model_cfg = build_model_arch_config(
            arch_variant=arch_variant,
            cnn_out_dim=cnn_out_dim,
            nhead_override=nhead,
            num_layers_override=num_layers,
        )
        print(f"[Manual] using arch_variant={arch_variant}.")
    elif ckpt.get("model_config") is not None:
        model_cfg = dict(ckpt["model_config"])
        model_cfg.setdefault("arch_variant", "schemeA")
        if nhead is not None:
            model_cfg["nhead"] = nhead
        if num_layers is not None:
            model_cfg["num_layers"] = num_layers
        print(f"[Auto] loaded model_config from ckpt: arch_variant={model_cfg.get('arch_variant')}")
    else:
        model_cfg = infer_model_arch_config_from_state(state, cnn_out_dim=cnn_out_dim)
        if nhead is not None:
            model_cfg["nhead"] = nhead
        if num_layers is not None:
            model_cfg["num_layers"] = num_layers
        print(f"[Auto] inferred arch_variant={model_cfg.get('arch_variant')} from state_dict.")

    model = ModelWiseTransformerClassifierNew(
        id2labels=id2label,
        seq_len=seq_len,
        num_models=num_models,
        cnn_out_dim=cnn_out_dim,
        sem_dim=sem_dim,
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg.get("dropout", 0.1),
        ignore_index=-1,
        dist_channels=model_cfg.get("dist_channels"),
        dist_kernel_sizes=model_cfg.get("dist_kernel_sizes"),
        use_semantic_conv=model_cfg.get("use_semantic_conv", False),
        sem_conv_channels=model_cfg.get("sem_conv_channels"),
        sem_conv_kernel_sizes=model_cfg.get("sem_conv_kernel_sizes"),
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    y_true = []
    y_pred = []
    y_true_bin = []
    y_pred_bin = []

    true_labels_in_order = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            true_labels_in_order.append(obj["label"])

    idx_global = 0
    for batch in tqdm(data.test_dataloader, desc="Eval sentence-level"):
        feats = batch["features"].to(device)
        labels_tok = batch["labels"].to(device)
        sem = batch["sem_features"]

        out = model(feats, labels=None, sem_features=sem)
        logits = out["logits"]
        pred_ids = torch.argmax(logits, dim=-1)

        batch_size_now, _ = pred_ids.shape
        for b in range(batch_size_now):
            mask = labels_tok[b] != -1
            pred_tok = pred_ids[b][mask].tolist()
            if len(pred_tok) == 0:
                idx_global += 1
                continue

            tags = [id2label[i] for i in pred_tok]
            base = [strip_bmes(t) for t in tags]

            pred_sent = Counter(base).most_common(1)[0][0]
            true_sent = true_labels_in_order[idx_global]

            y_true.append(true_sent)
            y_pred.append(pred_sent)

            tbin = "human" if true_sent == "human" else "LLM"
            pbin = "human" if pred_sent == "human" else "LLM"
            y_true_bin.append(tbin)
            y_pred_bin.append(pbin)

            idx_global += 1

    labels = list(EN_LABELS.keys())
    report, acc, macro_f1 = compute_prf1_per_class(y_true, y_pred, labels)

    print("\n=== Table 2 style (Multi-class) ===")
    for lab in labels:
        print(f"{lab:12s} P={report[lab]['P'] * 100:5.1f}  R={report[lab]['R'] * 100:5.1f}  F1={report[lab]['F1'] * 100:5.1f}")
    print(f"\nACC={acc * 100:.2f}  MacroF1={macro_f1 * 100:.2f}")

    report_b, acc_b, macro_f1_b = compute_prf1_per_class(y_true_bin, y_pred_bin, ["LLM", "human"])
    print("\n=== Table 4 style (Binary) ===")
    for lab in ["LLM", "human"]:
        print(f"{lab:5s} P={report_b[lab]['P'] * 100:5.1f}  R={report_b[lab]['R'] * 100:5.1f}  F1={report_b[lab]['F1'] * 100:5.1f}")
    print(f"\nACC={acc_b * 100:.2f}  MacroF1={macro_f1_b * 100:.2f}")


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--sem_dim", type=int, default=4096)
    ap.add_argument("--num_models", type=int, default=None)
    ap.add_argument("--cnn_out_dim", type=int, default=64)
    ap.add_argument("--arch_variant", type=str, default=None, choices=[None, "legacy", "schemeA"])
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    args = ap.parse_args()

    main(
        train_path=args.train_path,
        test_path=args.test_path,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        sem_dim=args.sem_dim,
        num_models=args.num_models,
        cnn_out_dim=args.cnn_out_dim,
        arch_variant=args.arch_variant,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

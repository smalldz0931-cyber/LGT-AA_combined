# -*- coding: utf-8 -*-
"""
merge_data.py

功能：
1）将多个白盒 extractor 的分布特征序列（ll_tokens_list）与语义特征（semantic npy）融合到同一条样本中，输出全量 merged jsonl。
2）在全量 merged 的基础上，进行 9:1 分层抽样划分训练/测试集（分层键：source_model + split_tag(HC3/FDT) + dataset）。
3）额外构建轻量（light）数据集（默认约 0.5% 且不超过 2000 条，保证分层抽样），并同样 9:1 划分训练/测试集。
4）将异常写入 ERROR_LOG_FILE，便于定位问题。

注意：
- 分布特征序列来自 gen_feature 输出字段 item["ll_tokens_list"][0]（外层 list 是为了兼容单 extractor 文件格式）。
- 每个 extractor 的 begin_idx_list 也要保留，否则 dataloader 无法做对齐截断。
"""

import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np


# =========================
# 项目信息（以你提供的为准）
# =========================
DIST_DIR = "/root/autodl-tmp/features_distribution"
SEM_DIR = "/root/autodl-tmp/features_semantic"
OUTPUT_FILE = "/root/autodl-tmp/merged_data/all_data.jsonl"
ERROR_LOG_FILE = "/root/autodl-tmp/merged_data/merge_error_log.txt"

hc3_datasets = ["finance", "medicine", "open_qa", "wiki_csai"]
fdt_datasets = ["squad", "writing", "xsum"]

source_models = [
    "Baichuan", "gpt_neo", "gpt2_xl", "llama3", "mistral", "opt",
    "PULI", "gpt3.5", "gpt4", "human", "claude",
    "deepseek_R1", "Qwen_plus", "doubao_seed"
]

extractors_order = ["gpt2", "gptneo", "gptj", "llama"]
BASELINE_EXTRACTOR = "llama"


# =========================
# 采样/划分配置
# =========================
RANDOM_SEED = 42

# 训练/测试划分比例：9:1
TRAIN_RATIO = 0.9

# 轻量数据集比例：建议 0.5%（比 1% 更快），且设置上限以保证调试速度
# 你也可以改成 0.01（1%）或更小
LIGHT_FRACTION = 0.005
LIGHT_MAX_SAMPLES = 2000

# 输出文件（自动在同目录生成）
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)
FULL_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_full.jsonl")
FULL_TEST_FILE = os.path.join(OUTPUT_DIR, "test_full.jsonl")
LIGHT_ALL_FILE = os.path.join(OUTPUT_DIR, "all_data_light.jsonl")
LIGHT_TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_light.jsonl")
LIGHT_TEST_FILE = os.path.join(OUTPUT_DIR, "test_light.jsonl")


# =========================
# 工具函数
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _log_error(msg: str) -> None:
    """将错误写入 ERROR_LOG_FILE（追加）"""
    _ensure_dir(os.path.dirname(ERROR_LOG_FILE))
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _read_all_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _safe_json_loads(line: str, context: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except Exception as e:
        _log_error(f"JSON解析失败 | {context} | err={repr(e)} | line_head={line[:200]}")
        return None


def _dist_jsonl_path(extractor: str, src: str, split_tag: str, dataset: str) -> str:
    """
    分布特征文件命名：
    en_{extractor}_{source_model}_{HC3/FDT}_{dataset}.jsonl
    示例：en_gpt2_Baichuan_FDT_squad.jsonl
    """
    return os.path.join(DIST_DIR, f"en_{extractor}_{src}_{split_tag}_{dataset}.jsonl")


def _semantic_npy_path(extractor: str, src: str, split_tag: str, dataset: str) -> str:
    """
    显著特征文件命名（注意：文件名里不包含 extractor，但目录包含 extractor）：
    features_semantic/{extractor}/en_{source_model}_{HC3/FDT}_{dataset}_semantic.npy
    示例：features_semantic/gpt2/en_Baichuan_FDT_squad_semantic.npy
    """
    return os.path.join(SEM_DIR, extractor, f"en_{src}_{split_tag}_{dataset}_semantic.npy")


def _load_semantic_npy(extractor: str, src: str, split_tag: str, dataset: str) -> np.ndarray:
    sem_path = _semantic_npy_path(extractor, src, split_tag, dataset)
    if not os.path.exists(sem_path):
        raise FileNotFoundError(sem_path)
    return np.load(sem_path)


def _stratum_key(label: str, split_tag: str, dataset: str) -> str:
    """
    分层键（同时分层 source_model + split_tag(HC3/FDT) + dataset）
    """
    return f"{label}@@{split_tag}@@{dataset}"


def _build_splits_by_strata(
    strata_to_lines: Dict[str, List[int]],
    train_ratio: float,
    rng: random.Random
) -> Tuple[set, set]:
    """对每个分层单独 shuffle，再按 train_ratio 划分"""
    train_set, test_set = set(), set()
    for key, lines in strata_to_lines.items():
        if not lines:
            continue
        rng.shuffle(lines)
        n = len(lines)
        n_train = int(round(n * train_ratio))
        # 保证每个分层至少 1 条进入测试（如果分层样本数足够）
        if n >= 2:
            n_train = min(max(n_train, 1), n - 1)
        else:
            n_train = 1  # n==1 时只能进 train
        train_set.update(lines[:n_train])
        test_set.update(lines[n_train:])
    return train_set, test_set


def _build_light_selection_by_strata(
    strata_to_lines: Dict[str, List[int]],
    fraction: float,
    max_total: int,
    rng: random.Random
) -> set:
    """
    分层抽样构建轻量数据集：
    - 每个分层抽取 max(1, floor(n * fraction))（若 n>0）
    - 若总量超过 max_total，则按分层比例再裁剪（尽量保持分层比例）
    """
    selected_by_key: Dict[str, List[int]] = {}
    total_selected = 0

    # 第一次：按 fraction 抽样（每层至少 1 条）
    for key, lines in strata_to_lines.items():
        if not lines:
            continue
        lines_copy = lines[:]  # 不破坏原列表
        rng.shuffle(lines_copy)
        n = len(lines_copy)
        k = int(n * fraction)
        if k < 1:
            k = 1
        k = min(k, n)
        selected_by_key[key] = lines_copy[:k]
        total_selected += k

    # 若超过上限：按比例裁剪（仍尽量保证每层至少 1）
    if total_selected > max_total and total_selected > 0:
        keep_ratio = max_total / float(total_selected)

        new_selected = set()
        # 先保证每层至少 1
        for key, lst in selected_by_key.items():
            if lst:
                new_selected.add(lst[0])

        # 剩余容量
        remaining = max_total - len(new_selected)
        if remaining < 0:
            remaining = 0

        # 将剩余候选按层展开，再按 keep_ratio 抽取
        candidates = []
        for key, lst in selected_by_key.items():
            if len(lst) > 1:
                candidates.extend(lst[1:])

        rng.shuffle(candidates)
        for idx in candidates[:remaining]:
            new_selected.add(idx)

        return new_selected

    # 未超过上限，直接返回所有选择
    out = set()
    for lst in selected_by_key.values():
        out.update(lst)
    return out


# =========================
# 主流程
# =========================
def main():
    _ensure_dir(OUTPUT_DIR)

    # 清空旧的错误日志（你也可以注释掉以保留历史）
    try:
        if os.path.exists(ERROR_LOG_FILE):
            os.remove(ERROR_LOG_FILE)
    except Exception as e:
        print(f"[WARN] 无法清空错误日志：{e}")

    rng = random.Random(RANDOM_SEED)

    # 记录：每条 merged 样本对应的行号 -> 分层键
    strata_to_lines: Dict[str, List[int]] = {}

    global_line_idx = 0

    # =========================
    # 第一阶段：写全量 merged 文件
    # =========================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for src in source_models:
            # -------- HC3 --------
            split_tag = "HC3"
            for dataset in hc3_datasets:
                # 1）读取各 extractor 的分布特征 jsonl
                dist_lines: Dict[str, List[str]] = {}
                missing_any = False
                for ext in extractors_order:
                    p = _dist_jsonl_path(ext, src, split_tag, dataset)
                    if not os.path.exists(p):
                        _log_error(f"缺少分布特征文件：{p}")
                        missing_any = True
                        break
                    dist_lines[ext] = _read_all_lines(p)
                if missing_any:
                    continue

                base_len = len(dist_lines[BASELINE_EXTRACTOR])

                # 2）读取各 extractor 的语义特征 npy
                sem_arrays: Dict[str, np.ndarray] = {}
                try:
                    for ext in extractors_order:
                        sem_arrays[ext] = _load_semantic_npy(ext, src, split_tag, dataset)
                except Exception as e:
                    _log_error(f"缺少语义特征文件 | src={src} split={split_tag} dataset={dataset} | err={repr(e)}")
                    continue

                # 3）长度一致性检查
                ok = True
                for ext in extractors_order:
                    if len(dist_lines[ext]) != base_len:
                        _log_error(
                            f"分布特征行数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={len(dist_lines[ext])}, baseline({BASELINE_EXTRACTOR})={base_len}"
                        )
                        ok = False
                        break
                    if sem_arrays[ext].shape[0] != base_len:
                        _log_error(
                            f"语义特征条数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={sem_arrays[ext].shape[0]}, baseline({BASELINE_EXTRACTOR})={base_len}"
                        )
                        ok = False
                        break
                if not ok:
                    continue

                # 4）逐行融合
                for i in range(base_len):
                    context = f"src={src} split={split_tag} dataset={dataset} idx={i}"

                    base_item = _safe_json_loads(dist_lines[BASELINE_EXTRACTOR][i], context + f" ext={BASELINE_EXTRACTOR}")
                    if base_item is None:
                        continue

                    dist_features_by_extractor = []
                    mean_loss_by_extractor = []
                    begin_idx_by_extractor = []
                    semantic_features_by_extractor = []

                    bad_line = False
                    for ext in extractors_order:
                        if ext == BASELINE_EXTRACTOR:
                            item = base_item
                        else:
                            item = _safe_json_loads(dist_lines[ext][i], context + f" ext={ext}")
                            if item is None:
                                bad_line = True
                                break

                        try:
                            dist_seq = item["ll_tokens_list"][0]
                            begin_idx = item["begin_idx_list"][0]
                            mean_loss = item["losses"][0]
                        except Exception as e:
                            _log_error(f"字段缺失/格式异常 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                        if not isinstance(dist_seq, list) or len(dist_seq) == 0:
                            _log_error(
                                f"分布序列为空或非list | {context} ext={ext} | "
                                f"type={type(dist_seq)} len={len(dist_seq) if isinstance(dist_seq, list) else 'NA'}"
                            )
                            bad_line = True
                            break

                        dist_features_by_extractor.append(dist_seq)
                        begin_idx_by_extractor.append(begin_idx)
                        mean_loss_by_extractor.append(mean_loss)

                        try:
                            semantic_features_by_extractor.append(sem_arrays[ext][i].tolist())
                        except Exception as e:
                            _log_error(f"语义特征读取失败 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                    if bad_line:
                        continue

                    merged_item = {
                        "text": base_item.get("text", ""),
                        "label": src,
                        "label_int": base_item.get("label_int", 0),
                        "prompt_len": base_item.get("prompt_len", 0),
                        "split_tag": split_tag,
                        "dataset": dataset,
                        "begin_idx_list": begin_idx_by_extractor,
                        "ll_tokens_list": dist_features_by_extractor,
                        "sem_feature_max": semantic_features_by_extractor,
                        "mean_loss": mean_loss_by_extractor,
                    }

                    fout.write(json.dumps(merged_item, ensure_ascii=False) + "\n")

                    key = _stratum_key(src, split_tag, dataset)
                    strata_to_lines.setdefault(key, []).append(global_line_idx)
                    global_line_idx += 1

            # -------- FDT --------
            split_tag = "FDT"
            for dataset in fdt_datasets:
                # 1）读取各 extractor 的分布特征 jsonl
                dist_lines: Dict[str, List[str]] = {}
                missing_any = False
                for ext in extractors_order:
                    p = _dist_jsonl_path(ext, src, split_tag, dataset)
                    if not os.path.exists(p):
                        _log_error(f"缺少分布特征文件：{p}")
                        missing_any = True
                        break
                    dist_lines[ext] = _read_all_lines(p)
                if missing_any:
                    continue

                base_len = len(dist_lines[BASELINE_EXTRACTOR])

                # 2）读取各 extractor 的语义特征 npy
                sem_arrays: Dict[str, np.ndarray] = {}
                try:
                    for ext in extractors_order:
                        sem_arrays[ext] = _load_semantic_npy(ext, src, split_tag, dataset)
                except Exception as e:
                    _log_error(f"缺少语义特征文件 | src={src} split={split_tag} dataset={dataset} | err={repr(e)}")
                    continue

                # 3）长度一致性检查
                ok = True
                for ext in extractors_order:
                    if len(dist_lines[ext]) != base_len:
                        _log_error(
                            f"分布特征行数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={len(dist_lines[ext])}, baseline({BASELINE_EXTRACTOR})={base_len}"
                        )
                        ok = False
                        break
                    if sem_arrays[ext].shape[0] != base_len:
                        _log_error(
                            f"语义特征条数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={sem_arrays[ext].shape[0]}, baseline({BASELINE_EXTRACTOR})={base_len}"
                        )
                        ok = False
                        break
                if not ok:
                    continue

                # 4）逐行融合
                for i in range(base_len):
                    context = f"src={src} split={split_tag} dataset={dataset} idx={i}"

                    base_item = _safe_json_loads(dist_lines[BASELINE_EXTRACTOR][i], context + f" ext={BASELINE_EXTRACTOR}")
                    if base_item is None:
                        continue

                    dist_features_by_extractor = []
                    mean_loss_by_extractor = []
                    begin_idx_by_extractor = []
                    semantic_features_by_extractor = []

                    bad_line = False
                    for ext in extractors_order:
                        if ext == BASELINE_EXTRACTOR:
                            item = base_item
                        else:
                            item = _safe_json_loads(dist_lines[ext][i], context + f" ext={ext}")
                            if item is None:
                                bad_line = True
                                break

                        try:
                            dist_seq = item["ll_tokens_list"][0]
                            begin_idx = item["begin_idx_list"][0]
                            mean_loss = item["losses"][0]
                        except Exception as e:
                            _log_error(f"字段缺失/格式异常 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                        if not isinstance(dist_seq, list) or len(dist_seq) == 0:
                            _log_error(
                                f"分布序列为空或非list | {context} ext={ext} | "
                                f"type={type(dist_seq)} len={len(dist_seq) if isinstance(dist_seq, list) else 'NA'}"
                            )
                            bad_line = True
                            break

                        dist_features_by_extractor.append(dist_seq)
                        begin_idx_by_extractor.append(begin_idx)
                        mean_loss_by_extractor.append(mean_loss)

                        try:
                            semantic_features_by_extractor.append(sem_arrays[ext][i].tolist())
                        except Exception as e:
                            _log_error(f"语义特征读取失败 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                    if bad_line:
                        continue

                    merged_item = {
                        "text": base_item.get("text", ""),
                        "label": src,
                        "label_int": base_item.get("label_int", 0),
                        "prompt_len": base_item.get("prompt_len", 0),
                        "split_tag": split_tag,
                        "dataset": dataset,
                        "begin_idx_list": begin_idx_by_extractor,
                        "ll_tokens_list": dist_features_by_extractor,
                        "sem_feature_max": semantic_features_by_extractor,
                        "mean_loss": mean_loss_by_extractor,
                    }

                    fout.write(json.dumps(merged_item, ensure_ascii=False) + "\n")

                    key = _stratum_key(src, split_tag, dataset)
                    strata_to_lines.setdefault(key, []).append(global_line_idx)
                    global_line_idx += 1

    print(f"[OK] 全量 merged 文件已生成：{OUTPUT_FILE}")
    print(f"[OK] merged 总行数：{global_line_idx}")

    if global_line_idx == 0:
        print("[WARN] merged 输出为空，请先检查 ERROR_LOG_FILE。")
        return

    # =========================
    # 第二阶段：分层 9:1 划分 full train/test
    # =========================
    strata_copy_for_full = {k: v[:] for k, v in strata_to_lines.items()}
    full_train_set, full_test_set = _build_splits_by_strata(strata_copy_for_full, TRAIN_RATIO, rng)

    # =========================
    # 第三阶段：分层抽样构建 light 数据集，并 9:1 划分
    # =========================
    strata_copy_for_light = {k: v[:] for k, v in strata_to_lines.items()}
    light_all_set = _build_light_selection_by_strata(strata_copy_for_light, LIGHT_FRACTION, LIGHT_MAX_SAMPLES, rng)

    light_strata: Dict[str, List[int]] = {}
    for key, lines in strata_to_lines.items():
        keep = [idx for idx in lines if idx in light_all_set]
        if keep:
            light_strata[key] = keep
    light_train_set, light_test_set = _build_splits_by_strata(light_strata, TRAIN_RATIO, rng)

    print(f"[OK] full_train={len(full_train_set)} full_test={len(full_test_set)}")
    print(f"[OK] light_all={len(light_all_set)} light_train={len(light_train_set)} light_test={len(light_test_set)}")

    # =========================
    # 第四阶段：按行号集合二次遍历 OUTPUT_FILE，写出各拆分文件
    # =========================
    with open(OUTPUT_FILE, "r", encoding="utf-8") as fin, \
         open(FULL_TRAIN_FILE, "w", encoding="utf-8") as ftrain, \
         open(FULL_TEST_FILE, "w", encoding="utf-8") as ftest, \
         open(LIGHT_ALL_FILE, "w", encoding="utf-8") as flight_all, \
         open(LIGHT_TRAIN_FILE, "w", encoding="utf-8") as flight_train, \
         open(LIGHT_TEST_FILE, "w", encoding="utf-8") as flight_test:

        for line_idx, line in enumerate(fin):
            if line_idx in full_train_set:
                ftrain.write(line)
            elif line_idx in full_test_set:
                ftest.write(line)
            else:
                _log_error(f"行号未进入 full train/test | line_idx={line_idx}")

            if line_idx in light_all_set:
                flight_all.write(line)
            if line_idx in light_train_set:
                flight_train.write(line)
            if line_idx in light_test_set:
                flight_test.write(line)

    print("[OK] 数据集拆分文件已生成：")
    print(f"  - 全量：{OUTPUT_FILE}")
    print(f"  - 全量训练：{FULL_TRAIN_FILE}")
    print(f"  - 全量测试：{FULL_TEST_FILE}")
    print(f"  - 轻量：{LIGHT_ALL_FILE}")
    print(f"  - 轻量训练：{LIGHT_TRAIN_FILE}")
    print(f"  - 轻量测试：{LIGHT_TEST_FILE}")
    print(f"[OK] 错误日志：{ERROR_LOG_FILE}")


if __name__ == "__main__":
    main()
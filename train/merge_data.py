# -*- coding: utf-8 -*-
"""
merge_data.py (enhanced)

在你现有 merge_data.py 基础上做“最小侵入式”增强：
- 支持通过参数选择 extractor 子集（1/2/3/4-models）。
- 通过 output_root_dir + run_name 生成独立输出子目录，避免覆盖原 4-model 结果。
- 完整保留原来的：全量合并 -> 9:1 分层 full 划分 -> light 分层抽样 + 9:1 -> 按行号写出各拆分文件。
- 严格遵守两类特征文件命名/目录规则：
  - 分布特征：features_distribution/en_{extractor}_{src}_{HC3/FDT}_{dataset}.jsonl
  - 语义特征：features_semantic/{extractor}/en_{src}_{HC3/FDT}_{dataset}_semantic.npy

注意：本脚本不会在拆分后删除 all_data.jsonl（按你的要求）。
"""

import os
import json
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np


# =========================
# 默认路径（你当前项目结构）
# =========================
DEFAULT_DIST_DIR = "/root/autodl-tmp/features_distribution"
DEFAULT_SEM_DIR = "/root/autodl-tmp/features_semantic"
DEFAULT_OUTPUT_ROOT_DIR = "/root/autodl-tmp/merged_data"


hc3_datasets = ["finance", "medicine", "open_qa", "wiki_csai"]
fdt_datasets = ["squad", "writing", "xsum"]

source_models = [
    "Baichuan", "gpt_neo", "gpt2_xl", "llama3", "mistral", "opt",
    "PULI", "gpt3.5", "gpt4", "human", "claude",
    "deepseek_R1", "Qwen_plus", "doubao_seed"
]


# =========================
# 工具函数
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log_error(error_log_file: str, msg: str) -> None:
    """将错误写入 error_log_file（追加）"""
    _ensure_dir(os.path.dirname(error_log_file))
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(f"[{_now_ts()}] {msg}\n")


def _read_all_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _safe_json_loads(line: str, error_log_file: str, context: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except Exception as e:
        _log_error(error_log_file, f"JSON解析失败 | {context} | err={repr(e)} | line_head={line[:200]}")
        return None


def _dist_jsonl_path(dist_dir: str, extractor: str, src: str, split_tag: str, dataset: str) -> str:
    """分布特征文件命名：en_{extractor}_{source_model}_{HC3/FDT}_{dataset}.jsonl"""
    return os.path.join(dist_dir, f"en_{extractor}_{src}_{split_tag}_{dataset}.jsonl")


def _semantic_npy_path(sem_dir: str, extractor: str, src: str, split_tag: str, dataset: str) -> str:
    """语义特征文件命名：features_semantic/{extractor}/en_{src}_{HC3/FDT}_{dataset}_semantic.npy"""
    return os.path.join(sem_dir, extractor, f"en_{src}_{split_tag}_{dataset}_semantic.npy")


def _load_semantic_npy(sem_dir: str, extractor: str, src: str, split_tag: str, dataset: str) -> np.ndarray:
    sem_path = _semantic_npy_path(sem_dir, extractor, src, split_tag, dataset)
    if not os.path.exists(sem_path):
        raise FileNotFoundError(sem_path)
    return np.load(sem_path)


def _stratum_key(label: str, split_tag: str, dataset: str) -> str:
    """分层键：source_model + split_tag(HC3/FDT) + dataset"""
    return f"{label}@@{split_tag}@@{dataset}"


def _build_splits_by_strata(
    strata_to_lines: Dict[str, List[int]],
    train_ratio: float,
    rng: random.Random
) -> Tuple[set, set]:
    """对每个分层单独 shuffle，再按 train_ratio 划分（保留你原来的边界处理）"""
    train_set, test_set = set(), set()
    for _key, lines in strata_to_lines.items():
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
    """完全沿用你原来的 light 分层抽样逻辑"""
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
        _ = keep_ratio  # 保留变量名以对齐原实现（虽然这里不用）

        new_selected = set()
        # 先保证每层至少 1
        for key, lst in selected_by_key.items():
            if lst:
                new_selected.add(lst[0])

        # 剩余容量
        remaining = max_total - len(new_selected)
        if remaining < 0:
            remaining = 0

        # 将剩余候选按层展开，再按比例抽取
        candidates = []
        for _key, lst in selected_by_key.items():
            if len(lst) > 1:
                candidates.extend(lst[1:])

        rng.shuffle(candidates)
        for idx in candidates[:remaining]:
            new_selected.add(idx)

        return new_selected

    out = set()
    for lst in selected_by_key.values():
        out.update(lst)
    return out


def _parse_extractors(s: str) -> List[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    # 保持用户传入的顺序（用于 ll_tokens_list/sem_feature_max 的通道顺序）
    return items


def _auto_run_name(extractors: List[str]) -> str:
    return f"m{len(extractors)}_" + "_".join(extractors)


def _choose_baseline(extractors: List[str], baseline_arg: str) -> str:
    if baseline_arg == "auto":
        # 如果 llama 在子集中，优先用 llama（你之前的默认就是 llama）
        if "llama" in extractors:
            return "llama"
        return extractors[0]
    if baseline_arg == "first":
        return extractors[0]
    # 指定某个 extractor
    if baseline_arg not in extractors:
        raise ValueError(f"--baseline={baseline_arg} 必须在 --extractors={extractors} 中")
    return baseline_arg


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dist_dir", type=str, default=DEFAULT_DIST_DIR)
    parser.add_argument("--sem_dir", type=str, default=DEFAULT_SEM_DIR)
    parser.add_argument("--output_root_dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR)

    # 选择 extractor 子集（支持 1/2/3/4 models）
    parser.add_argument(
        "--extractors",
        type=str,
        default="gpt2,gptneo,gptj,llama",
        help="使用哪些 extractor（逗号分隔），例如：llama 或 gptneo,llama,gpt2"
    )

    # 输出子目录名。若不指定，自动用 m{K}_{extractors...}
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="输出子目录名（可选）。不填则自动生成，如 m1_llama / m3_gptneo_llama_gpt2"
    )

    # baseline：auto（优先 llama）/ first / 指定 extractor
    parser.add_argument(
        "--baseline",
        type=str,
        default="auto",
        help='baseline extractor：auto|first|<extractor>。auto 时若包含 llama 则用 llama'
    )

    # 采样/划分参数：默认保持你的原值
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--light_fraction", type=float, default=0.005)
    parser.add_argument("--light_max_samples", type=int, default=2000)

    args = parser.parse_args()

    extractors_order = _parse_extractors(args.extractors)
    if not extractors_order:
        raise ValueError("--extractors 不能为空")

    baseline_extractor = _choose_baseline(extractors_order, args.baseline)

    run_name = args.run_name.strip() or _auto_run_name(extractors_order)
    output_dir = os.path.join(args.output_root_dir, run_name)
    _ensure_dir(output_dir)

    output_file = os.path.join(output_dir, "all_data.jsonl")
    error_log_file = os.path.join(output_dir, "merge_error_log.txt")

    full_train_file = os.path.join(output_dir, "train_full.jsonl")
    full_test_file = os.path.join(output_dir, "test_full.jsonl")
    light_all_file = os.path.join(output_dir, "all_data_light.jsonl")
    light_train_file = os.path.join(output_dir, "train_light.jsonl")
    light_test_file = os.path.join(output_dir, "test_light.jsonl")

    # 清空旧错误日志（不影响旧目录，因为每次 run_name 不同）
    try:
        if os.path.exists(error_log_file):
            os.remove(error_log_file)
    except Exception as e:
        print(f"[WARN] 无法清空错误日志：{e}")

    print("=" * 80)
    print("[Merge Config]")
    print(f"dist_dir      : {args.dist_dir}")
    print(f"sem_dir       : {args.sem_dir}")
    print(f"output_dir    : {output_dir}")
    print(f"extractors    : {extractors_order}")
    print(f"baseline      : {baseline_extractor}")
    print(f"seed          : {args.random_seed}")
    print(f"train_ratio   : {args.train_ratio}")
    print(f"light_fraction: {args.light_fraction}")
    print(f"light_max     : {args.light_max_samples}")
    print("=" * 80)

    rng = random.Random(args.random_seed)

    # 记录：每条 merged 样本对应的行号 -> 分层键
    strata_to_lines: Dict[str, List[int]] = {}
    global_line_idx = 0

    # =========================
    # 第一阶段：写全量 merged 文件
    # =========================
    with open(output_file, "w", encoding="utf-8") as fout:
        for src in source_models:
            # -------- HC3 --------
            split_tag = "HC3"
            for dataset in hc3_datasets:
                # 1）读取各 extractor 的分布特征 jsonl
                dist_lines: Dict[str, List[str]] = {}
                missing_any = False
                for ext in extractors_order:
                    p = _dist_jsonl_path(args.dist_dir, ext, src, split_tag, dataset)
                    if not os.path.exists(p):
                        _log_error(error_log_file, f"缺少分布特征文件：{p}")
                        missing_any = True
                        break
                    dist_lines[ext] = _read_all_lines(p)
                if missing_any:
                    continue

                base_len = len(dist_lines[baseline_extractor])

                # 2）读取各 extractor 的语义特征 npy
                sem_arrays: Dict[str, np.ndarray] = {}
                try:
                    for ext in extractors_order:
                        sem_arrays[ext] = _load_semantic_npy(args.sem_dir, ext, src, split_tag, dataset)
                except Exception as e:
                    _log_error(error_log_file, f"缺少语义特征文件 | src={src} split={split_tag} dataset={dataset} | err={repr(e)}")
                    continue

                # 3）长度一致性检查
                ok = True
                for ext in extractors_order:
                    if len(dist_lines[ext]) != base_len:
                        _log_error(
                            error_log_file,
                            f"分布特征行数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={len(dist_lines[ext])}, baseline({baseline_extractor})={base_len}"
                        )
                        ok = False
                        break
                    if sem_arrays[ext].shape[0] != base_len:
                        _log_error(
                            error_log_file,
                            f"语义特征条数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={sem_arrays[ext].shape[0]}, baseline({baseline_extractor})={base_len}"
                        )
                        ok = False
                        break
                if not ok:
                    continue

                # 4）逐行融合
                for i in range(base_len):
                    context = f"src={src} split={split_tag} dataset={dataset} idx={i}"

                    base_item = _safe_json_loads(dist_lines[baseline_extractor][i], error_log_file, context + f" ext={baseline_extractor}")
                    if base_item is None:
                        continue

                    dist_features_by_extractor = []
                    mean_loss_by_extractor = []
                    begin_idx_by_extractor = []
                    semantic_features_by_extractor = []

                    bad_line = False
                    for ext in extractors_order:
                        if ext == baseline_extractor:
                            item = base_item
                        else:
                            item = _safe_json_loads(dist_lines[ext][i], error_log_file, context + f" ext={ext}")
                            if item is None:
                                bad_line = True
                                break

                        try:
                            # 分布特征来自 gen_feature 输出字段 item["ll_tokens_list"][0]
                            dist_seq = item["ll_tokens_list"][0]
                            begin_idx = item["begin_idx_list"][0]
                            mean_loss = item["losses"][0]
                        except Exception as e:
                            _log_error(error_log_file, f"字段缺失/格式异常 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                        if not isinstance(dist_seq, list) or len(dist_seq) == 0:
                            _log_error(
                                error_log_file,
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
                            _log_error(error_log_file, f"语义特征读取失败 | {context} ext={ext} | err={repr(e)}")
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
                        # meta：不影响你现有 dataloader，但方便复现/排错
                        "extractors": extractors_order,
                        "baseline_extractor": baseline_extractor,
                    }

                    fout.write(json.dumps(merged_item, ensure_ascii=False) + "\n")

                    key = _stratum_key(src, split_tag, dataset)
                    strata_to_lines.setdefault(key, []).append(global_line_idx)
                    global_line_idx += 1

            # -------- FDT --------
            split_tag = "FDT"
            for dataset in fdt_datasets:
                dist_lines: Dict[str, List[str]] = {}
                missing_any = False
                for ext in extractors_order:
                    p = _dist_jsonl_path(args.dist_dir, ext, src, split_tag, dataset)
                    if not os.path.exists(p):
                        _log_error(error_log_file, f"缺少分布特征文件：{p}")
                        missing_any = True
                        break
                    dist_lines[ext] = _read_all_lines(p)
                if missing_any:
                    continue

                base_len = len(dist_lines[baseline_extractor])

                sem_arrays: Dict[str, np.ndarray] = {}
                try:
                    for ext in extractors_order:
                        sem_arrays[ext] = _load_semantic_npy(args.sem_dir, ext, src, split_tag, dataset)
                except Exception as e:
                    _log_error(error_log_file, f"缺少语义特征文件 | src={src} split={split_tag} dataset={dataset} | err={repr(e)}")
                    continue

                ok = True
                for ext in extractors_order:
                    if len(dist_lines[ext]) != base_len:
                        _log_error(
                            error_log_file,
                            f"分布特征行数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={len(dist_lines[ext])}, baseline({baseline_extractor})={base_len}"
                        )
                        ok = False
                        break
                    if sem_arrays[ext].shape[0] != base_len:
                        _log_error(
                            error_log_file,
                            f"语义特征条数不一致 | src={src} split={split_tag} dataset={dataset} | "
                            f"{ext}={sem_arrays[ext].shape[0]}, baseline({baseline_extractor})={base_len}"
                        )
                        ok = False
                        break
                if not ok:
                    continue

                for i in range(base_len):
                    context = f"src={src} split={split_tag} dataset={dataset} idx={i}"

                    base_item = _safe_json_loads(dist_lines[baseline_extractor][i], error_log_file, context + f" ext={baseline_extractor}")
                    if base_item is None:
                        continue

                    dist_features_by_extractor = []
                    mean_loss_by_extractor = []
                    begin_idx_by_extractor = []
                    semantic_features_by_extractor = []

                    bad_line = False
                    for ext in extractors_order:
                        if ext == baseline_extractor:
                            item = base_item
                        else:
                            item = _safe_json_loads(dist_lines[ext][i], error_log_file, context + f" ext={ext}")
                            if item is None:
                                bad_line = True
                                break

                        try:
                            dist_seq = item["ll_tokens_list"][0]
                            begin_idx = item["begin_idx_list"][0]
                            mean_loss = item["losses"][0]
                        except Exception as e:
                            _log_error(error_log_file, f"字段缺失/格式异常 | {context} ext={ext} | err={repr(e)}")
                            bad_line = True
                            break

                        if not isinstance(dist_seq, list) or len(dist_seq) == 0:
                            _log_error(
                                error_log_file,
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
                            _log_error(error_log_file, f"语义特征读取失败 | {context} ext={ext} | err={repr(e)}")
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
                        "extractors": extractors_order,
                        "baseline_extractor": baseline_extractor,
                    }

                    fout.write(json.dumps(merged_item, ensure_ascii=False) + "\n")

                    key = _stratum_key(src, split_tag, dataset)
                    strata_to_lines.setdefault(key, []).append(global_line_idx)
                    global_line_idx += 1

    print(f"[OK] 全量 merged 文件已生成：{output_file}")
    print(f"[OK] merged 总行数：{global_line_idx}")

    if global_line_idx == 0:
        print("[WARN] merged 输出为空，请先检查错误日志。")
        print(f"[WARN] error_log: {error_log_file}")
        return

    # =========================
    # 第二阶段：分层 9:1 划分 full train/test
    # =========================
    strata_copy_for_full = {k: v[:] for k, v in strata_to_lines.items()}
    full_train_set, full_test_set = _build_splits_by_strata(strata_copy_for_full, args.train_ratio, rng)

    # =========================
    # 第三阶段：分层抽样构建 light 数据集，并 9:1 划分
    # =========================
    strata_copy_for_light = {k: v[:] for k, v in strata_to_lines.items()}
    light_all_set = _build_light_selection_by_strata(strata_copy_for_light, args.light_fraction, args.light_max_samples, rng)

    light_strata: Dict[str, List[int]] = {}
    for key, lines in strata_to_lines.items():
        keep = [idx for idx in lines if idx in light_all_set]
        if keep:
            light_strata[key] = keep
    light_train_set, light_test_set = _build_splits_by_strata(light_strata, args.train_ratio, rng)

    print(f"[OK] full_train={len(full_train_set)} full_test={len(full_test_set)}")
    print(f"[OK] light_all={len(light_all_set)} light_train={len(light_train_set)} light_test={len(light_test_set)}")

    # =========================
    # 第四阶段：按行号集合二次遍历 output_file，写出各拆分文件
    # =========================
    with open(output_file, "r", encoding="utf-8") as fin, \
         open(full_train_file, "w", encoding="utf-8") as ftrain, \
         open(full_test_file, "w", encoding="utf-8") as ftest, \
         open(light_all_file, "w", encoding="utf-8") as flight_all, \
         open(light_train_file, "w", encoding="utf-8") as flight_train, \
         open(light_test_file, "w", encoding="utf-8") as flight_test:

        for line_idx, line in enumerate(fin):
            if line_idx in full_train_set:
                ftrain.write(line)
            elif line_idx in full_test_set:
                ftest.write(line)
            else:
                _log_error(error_log_file, f"行号未进入 full train/test | line_idx={line_idx}")

            if line_idx in light_all_set:
                flight_all.write(line)
            if line_idx in light_train_set:
                flight_train.write(line)
            if line_idx in light_test_set:
                flight_test.write(line)

    print("[OK] 数据集拆分文件已生成：")
    print(f"  - 全量：{output_file}")
    print(f"  - 全量训练：{full_train_file}")
    print(f"  - 全量测试：{full_test_file}")
    print(f"  - 轻量：{light_all_file}")
    print(f"  - 轻量训练：{light_train_file}")
    print(f"  - 轻量测试：{light_test_file}")
    print(f"[OK] 错误日志：{error_log_file}")


if __name__ == "__main__":
    main()
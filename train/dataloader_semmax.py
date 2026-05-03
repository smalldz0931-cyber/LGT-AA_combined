import os
import re
import json
import random
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Optional[str], message: str):
    """
    追加写入日志（带时间戳）。log_path=None 时不写。
    """
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] {message}\n")


# ----------------------------
# 1) JSONL 偏移量索引数据集
# ----------------------------
class JsonlOffsetDataset(Dataset):
    """
    不爆内存的 jsonl Dataset：
    - 初始化：扫描一次文件，记录每行起始字节 offset（并缓存 offsets.npy）
    - __getitem__：按 offset seek 读取单行 json
    """

    def __init__(self, jsonl_path: str, cache_dir: Optional[str] = None, verbose: bool = True,
                 log_path: Optional[str] = None):
        self.jsonl_path = jsonl_path
        self.verbose = verbose
        self.log_path = log_path

        if cache_dir is None:
            cache_dir = os.path.dirname(jsonl_path)
        os.makedirs(cache_dir, exist_ok=True)

        stem = Path(jsonl_path).stem
        self.offset_cache_path = os.path.join(cache_dir, f"{stem}.offsets.npy")

        self.offsets = self._load_or_build_offsets()
        self._fp = None  # lazy open

    def _load_or_build_offsets(self) -> np.ndarray:
        if os.path.exists(self.offset_cache_path):
            try:
                offsets = np.load(self.offset_cache_path)
                if self.verbose:
                    print(f"[OffsetDataset] Loaded offsets cache: {self.offset_cache_path} (n={len(offsets)})")
                return offsets
            except Exception as e:
                append_log(self.log_path, f"Offset cache load failed, rebuild. cache={self.offset_cache_path} err={e}")

        if self.verbose:
            print(f"[OffsetDataset] Building offsets for: {self.jsonl_path} (scan once)")

        offsets: List[int] = []
        offset = 0
        with open(self.jsonl_path, "rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)

        offsets_np = np.array(offsets, dtype=np.int64)
        np.save(self.offset_cache_path, offsets_np)

        if self.verbose:
            print(f"[OffsetDataset] Offsets saved: {self.offset_cache_path} (n={len(offsets_np)})")
        return offsets_np

    def _get_fp(self):
        if self._fp is None:
            self._fp = open(self.jsonl_path, "rb")
        return self._fp

    def __len__(self):
        return int(len(self.offsets))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fp = self._get_fp()
        fp.seek(int(self.offsets[idx]))
        line = fp.readline()
        if not line:
            append_log(self.log_path, f"READ_EMPTY_LINE file={self.jsonl_path} idx={idx}")
            return {}

        try:
            return json.loads(line.decode("utf-8"))
        except Exception as e:
            # JSON坏行：记录并返回空
            append_log(self.log_path, f"JSON_DECODE_FAIL file={self.jsonl_path} idx={idx} err={repr(e)}")
            return {}


# ----------------------------
# 2) DataManager：接口对齐 train.py
# ----------------------------
class DataManager:
    """
    输出 batch 字段必须保持一致：
    - features     : torch.FloatTensor [B, max_len, num_models]
    - labels       : torch.LongTensor  [B, max_len] (pad=-1)
    - text         : List[str]
    - sem_features : np.ndarray [B, num_models, sem_dim_padded]
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int,
        max_len: int,
        human_label: str,
        id2label: Dict[int, str],
        word_pad_idx: int = 0,
        label_pad_idx: int = -1,
        cache_dir: Optional[str] = None,
        issue_log_path: str = "/root/autodl-tmp/train/dataloader_issues.log",
    ):
        set_seed(0)

        self.batch_size = batch_size
        self.max_len = max_len

        self.human_label = human_label
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

        self.issue_log_path = issue_log_path
        append_log(self.issue_log_path, "===== DataManager init =====")
        append_log(self.issue_log_path, f"train_path={train_path}")
        append_log(self.issue_log_path, f"test_path={test_path}")

        self.train_dataset = JsonlOffsetDataset(
            train_path, cache_dir=cache_dir, verbose=True, log_path=self.issue_log_path
        ) if train_path else None
        self.test_dataset = JsonlOffsetDataset(
            test_path, cache_dir=cache_dir, verbose=True, log_path=self.issue_log_path
        ) if test_path else None

        self.train_dataloader = self.get_train_dataloader(self.train_dataset) if self.train_dataset else None
        self.test_dataloader = self.get_eval_dataloader(self.test_dataset) if self.test_dataset else None

    def get_train_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=0,      # 稳定优先；后面可尝试 2/4
            pin_memory=True
        )

    def get_eval_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True
        )

    # ----------------------------
    # 3) collator：按需处理每条样本，并记录坏样本/错误
    # ----------------------------
    def data_collator(self, raw_samples: List[Dict[str, Any]]):
        """
        raw_samples: 来自 Dataset 的原始 dict 列表（逐行 json）
        这里进行：
        - 对数特征对齐、pad/trunc 到 max_len
        - token-level BMES labels 生成
        - sem_features batch 内 pad
        - 记录坏样本和异常
        """
        # 过滤完全空样本（通常是 JSON decode fail）
        samples = []
        for s in raw_samples:
            if not isinstance(s, dict) or len(s) == 0:
                # JSON坏行已在 Dataset 里记过，这里不重复
                continue
            samples.append(s)

        if len(samples) == 0:
            append_log(self.issue_log_path, "BATCH_EMPTY_ALL_INVALID (all samples invalid after dataset decode)")
            raise RuntimeError("All samples in this batch are invalid/empty. Check jsonl integrity.")

        texts: List[str] = []
        labels_str: List[str] = []
        prompt_lens: List[int] = []
        split_tags: List[str] = []
        datasets: List[str] = []
        feats_list: List[List[List[float]]] = []
        sem_list: List[List[List[float]]] = []

        # 注意：我们没有行号字段。为了可追踪，建议你在 merge_data 时给每条样本加一个 uid/line_id。
        # 目前日志里记录 text 前 80 字 + label 作为定位线索。
        def sample_hint(item: Dict[str, Any]) -> str:
            t = (item.get("text", "") or "")[:80].replace("\n", "\\n")
            return f"label={item.get('label')} text[:80]={t}"

        for item in samples:
            try:
                # 字段存在性检查
                required = ["text", "label", "begin_idx_list", "ll_tokens_list", "sem_feature_max"]
                missing = [k for k in required if k not in item]
                if missing:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_MISSING_FIELDS missing={missing} {sample_hint(item)}")
                    continue

                text = item["text"]
                label = item["label"]
                prompt_len = int(item.get("prompt_len", 0))
                split_tag = str(item.get("split_tag", "") or "").upper()
                dataset_name = str(item.get("dataset", "") or "")

                begin_idx_list = item["begin_idx_list"]
                ll_tokens_list = item["ll_tokens_list"]
                sem_features = item["sem_feature_max"]

                # label 合法性检查（避免后面 BMES KeyError）
                if f"B-{label}" not in self.label2id:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_UNKNOWN_LABEL {sample_hint(item)}")
                    continue

                # ll_tokens_list 结构检查
                if (not isinstance(ll_tokens_list, list)) or len(ll_tokens_list) == 0:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_BAD_LL_TOKENS empty_or_not_list {sample_hint(item)}")
                    continue

                # ---------- A) 分布特征对齐：起点对齐 + 长度对齐 ----------
                begin_idx_arr = np.array(begin_idx_list)
                max_begin_idx = int(np.max(begin_idx_arr))

                # 截断对齐起点
                aligned = []
                for seq in ll_tokens_list:
                    if not isinstance(seq, list):
                        aligned = None
                        break
                    aligned.append(seq[max_begin_idx:])
                if aligned is None:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_BAD_LL_SEQ not_list {sample_hint(item)}")
                    continue

                # 统一最短长度
                min_len = int(np.min([len(x) for x in aligned])) if aligned else 0
                if min_len <= 0:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_MINLEN_LE0 min_len={min_len} {sample_hint(item)}")
                    continue

                aligned = [x[:min_len] for x in aligned]

                # [num_models, seq_len] -> [seq_len, num_models]
                feats = np.array(aligned).transpose().tolist()

                # sem_features 检查
                if (not isinstance(sem_features, list)) or len(sem_features) == 0:
                    append_log(self.issue_log_path, f"SAMPLE_SKIP_BAD_SEM empty_or_not_list {sample_hint(item)}")
                    continue

                texts.append(text)
                labels_str.append(label)
                prompt_lens.append(prompt_len)
                split_tags.append(split_tag)
                datasets.append(dataset_name)
                feats_list.append(feats)
                sem_list.append(sem_features)

            except Exception as e:
                append_log(self.issue_log_path, f"SAMPLE_SKIP_EXCEPTION err={repr(e)} {sample_hint(item)}")
                continue

        if len(feats_list) == 0:
            append_log(self.issue_log_path, "BATCH_NO_VALID_SAMPLES_AFTER_PREPROCESS")
            raise RuntimeError("No valid samples after preprocessing. Check data fields and labels.")

        try:
            # ---------- B) features -> tensor [B,max_len,feat_dim] ----------
            features, valid_masks = self.process_and_convert_to_tensor(feats_list)

            # labels 张量默认全部置为 ignore_index：
            # - pad 位置本来就应忽略
            # - FDT 被丢弃的前缀 token 也应忽略
            masks = torch.full_like(valid_masks, fill_value=self.label_pad_idx)

            # ---------- C) 生成 token-level BMES labels ----------
            for idx, p_len in enumerate(prompt_lens):
                total_len = len(self.split_sentence(texts[idx]))
                split_tag = split_tags[idx]
                dataset_name = datasets[idx]

                if split_tag == "HC3":
                    # HC3：整段文本都标注为该 LLM 标签，不再把前缀当成人类文本。
                    if total_len > 0:
                        eff_len = min(total_len, self.max_len)
                        full_ids = self.sequence_labels_to_ids(eff_len, labels_str[idx])
                        masks[idx][:eff_len] = full_ids[:]
                elif split_tag == "FDT":
                    # FDT：丢弃前 min(prompt_len, 25) 个 token（保持 0，随后加上 pad_masks 后会成为 0，而不是 human）
                    prefix_len = len(self.split_sentence(texts[idx][:p_len]))
                    drop_len = min(prefix_len, 25)
                    suffix_len = total_len - drop_len
                    if suffix_len > 0 and drop_len < self.max_len:
                        eff_len = min(suffix_len, self.max_len - drop_len)
                        suffix_ids = self.sequence_labels_to_ids(eff_len, labels_str[idx])
                        masks[idx][drop_len: drop_len + eff_len] = suffix_ids[:]
                else:
                    # 兼容旧数据：若缺少 split_tag，则沿用旧逻辑。
                    prefix_len = len(self.split_sentence(texts[idx][:p_len]))

                    if prefix_len > self.max_len:
                        prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                        masks[idx][:] = prefix_ids[:]
                        continue

                    if prefix_len > 0:
                        prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                        masks[idx][:prefix_len] = prefix_ids[:]

                    if total_len - prefix_len > 0:
                        if total_len > self.max_len:
                            suffix_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, labels_str[idx])
                        else:
                            suffix_ids = self.sequence_labels_to_ids(total_len - prefix_len, labels_str[idx])
                        masks[idx][prefix_len: min(total_len, self.max_len)] = suffix_ids[:]

                    append_log(
                        self.issue_log_path,
                        f"LABEL_POLICY_FALLBACK split_tag={split_tag!r} dataset={dataset_name!r} text_head={texts[idx][:50].replace(chr(10), ' ')}"
                    )


            # ---------- D) sem_features batch pad ----------
            max_semdim = max(len(feature) for sample in sem_list for feature in sample)
            sem_padded = [
                [feature + [0] * (max_semdim - len(feature)) for feature in sample]
                for sample in sem_list
            ]
            sem_arr = np.array(sem_padded)

            return {
                "features": features,
                "labels": masks,
                "text": texts,
                "sem_features": sem_arr
            }

        except Exception as e:
            # 这是 batch 级错误（例如 tensor shape 不一致）
            append_log(self.issue_log_path, f"BATCH_PROCESS_EXCEPTION err={repr(e)} batch_size={len(feats_list)}")
            raise

    def sequence_labels_to_ids(self, seq_len: int, label: str):
        if seq_len <= 0:
            return None
        if seq_len == 1:
            tag = "S-" + label
            return torch.tensor([self.label2id[tag]], dtype=torch.long)

        ids = [self.label2id["B-" + label]]
        ids.extend([self.label2id["M-" + label]] * (seq_len - 2))
        ids.append(self.label2id["E-" + label])
        return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data: List[List[List[float]]]):
        """
        data: list[B], each is [seq_len, feat_dim]
        output:
        - tensor_data: [B,max_len,feat_dim] float32
        - tensor_mask: [B,max_len] long {0,1}
        """
        max_len = self.max_len
        feat_dim = len(data[0][0])

        padded_data = [seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data]
        padded_data = [seq[:max_len] for seq in padded_data]
        masks = [[1] * min(len(seq), max_len) + [0] * (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float32)
        tensor_mask = torch.tensor(masks, dtype=torch.long)
        return tensor_data, tensor_mask

    # --- split_sentence：和你之前保持一致 ---
    def _split_en_sentence(self, sentence: str, use_sp: bool = False):
        pattern = re.compile(r"\S+|\s")
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    def _split_cn_sentence(self, sentence: str, use_sp: bool = False):
        words = list(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    def split_sentence(self, sentence: str, use_sp: bool = False, cn_percent: float = 0.2):
        total_char_count = len(sentence) or 1
        chinese_char_count = sum("\u4e00" <= char <= "\u9fff" for char in sentence)
        if chinese_char_count / total_char_count > cn_percent:
            return self._split_cn_sentence(sentence, use_sp)
        else:
            return self._split_en_sentence(sentence, use_sp)
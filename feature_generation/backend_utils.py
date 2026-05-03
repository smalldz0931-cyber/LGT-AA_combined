import re
import torch
import numpy as np
import unicodedata


def _split_en_sentence(sentence, use_sp=False):
    """
    Split an English sentence into a sequence of words and whitespace characters according to whitespace characters.
    """
    pattern = re.compile(r'\S+|\s')
    words = pattern.findall(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def _split_cn_sentence(sentence, use_sp=False):
    """
    Split a Chinese sentence into a sequence of characters.
    """
    words = list(sentence)
    if use_sp:
        words = ["▁" if item == " " else item for item in words]
    return words


def split_sentence(sentence, use_sp=False, cn_percent=0.2):
    total_char_count = len(sentence)
    total_char_count += 1 if total_char_count == 0 else 0
    chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
    if chinese_char_count / total_char_count > cn_percent:
        return _split_cn_sentence(sentence, use_sp)
    else:
        return _split_en_sentence(sentence, use_sp)


def _safe_mean(arr):
    if not arr:
        return 0.0
    return float(np.mean(arr))


def _truncate_with_tokenizer(tokenizer, text, device, max_length=1024):
    """
    关键修复点：
    - 统一以 tokenizer truncation 后的 input_ids 为准
    - 返回 decoded_text，用于后续 split_sentence，避免“words来自原文 / input_ids来自截断”的错位
    """
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    # tokenizer 可能返回 BatchEncoding，to(device) 在部分版本可用也可能不可用
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    input_ids = tokenized["input_ids"]

    # 用截断后的 input_ids 反解码得到“真正送入模型”的文本近似
    try:
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    except Exception:
        decoded_text = text

    labels = input_ids.clone()
    return input_ids, labels, decoded_text


class TikTokenizerPPLCalc(object):
    """ base_tokenizer is based on the 'BBPE Algorithm' (OpenAI API path) """
    # 这里未改动你的 openai 路径逻辑（你当前白盒脚本未用到）


class BBPETokenizerPPLCalc(object):
    """
    base_tokenizer is based on the 'BBPE Algorithm'

    ✅ 修复点 1（向后兼容）：
    - 兼容你那版：BBPETokenizerPPLCalc(bytes_to_unicode=..., ...)
    - 也兼容原版：BBPETokenizerPPLCalc(byte_encoder, model, tokenizer, device)
    """

    def __init__(self, byte_encoder=None, base_model=None, base_tokenizer=None, device="cuda", bytes_to_unicode=None):
        # 兼容关键字 bytes_to_unicode
        if byte_encoder is None and bytes_to_unicode is not None:
            byte_encoder = bytes_to_unicode
        if byte_encoder is None:
            raise ValueError("BBPETokenizerPPLCalc: byte_encoder(bytes_to_unicode mapping) is required.")

        self.byte_encoder = byte_encoder
        self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_bbpe_bytes(self, words):
        bbs = []
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for _ in range(len(byte_list))])
        return bbs, bbs_to_words

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits  # [B, T, V]
        # shift for next-token prediction
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def get_bbs_ll(self, input_ids, ll):
        """
        把 subtoken 的 loss 映射到 byte 级别
        """
        input_ids = input_ids.squeeze(0)  # [T]
        tokenized_tokens = [self.base_tokenizer._convert_id_to_token(int(i)) for i in input_ids]

        bbs_ll = []
        if len(tokenized_tokens) == 0:
            return bbs_ll

        # GPT2 类 tokenizer 不含显式 BOS；第一个 token 的 ll 通常无法对齐，按 0 处理
        first = tokenized_tokens[0]
        try:
            first_bytes = [self.byte_decoder[c] for c in first]
        except Exception:
            first_bytes = []
        bbs_ll.extend([0 for _ in range(len(first_bytes))])

        # 后续 token：ll 的长度通常是 (T-1) 对应 tokenized_tokens[1:]
        for idx, token in enumerate(tokenized_tokens[1:]):
            try:
                byte_list = [self.byte_decoder[c] for c in token]
            except Exception:
                byte_list = []
            # ll[idx] 对应 tokenized_tokens[idx+1]
            val = ll[idx] if idx < len(ll) else 0
            bbs_ll.extend(val for _ in range(len(byte_list)))
        return bbs_ll

    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_tokens.append(_safe_mean(bbs_ll[start:end]))
            start = end
        return ll_tokens

    def get_begin_word_idx(self, input_ids, bbs_to_words):
        input_ids = input_ids.squeeze(0)
        if input_ids.numel() == 0:
            return 0
        begin_token = self.base_tokenizer._convert_id_to_token(int(input_ids[0]))
        try:
            byte_list = [self.byte_decoder[c] for c in begin_token]
        except Exception:
            byte_list = []
        if not byte_list:
            return 0
        begin_word_idx = bbs_to_words[len(byte_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        # ✅ 修复点 2：用 tokenizer truncation 后的 decoded_text 来 split，避免错位
        input_ids, labels, decoded_text = _truncate_with_tokenizer(
            self.base_tokenizer, text, self.device, max_length=1024
        )

        words = split_sentence(decoded_text)
        _, bbs_to_words = self.get_bbpe_bytes(words)

        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss, ll = self.calc_sent_ppl(outputs, labels)

        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(bbs_to_words, bbs_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, bbs_to_words)
        return [loss, begin_word_idx, ll_tokens]


class CharLevelTokenizerPPLCalc(object):
    """ base_tokenizer is based on `Char Level` """

    def __init__(self, all_special_tokens, base_model, base_tokenizer, device):
        self.all_special_tokens = all_special_tokens
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_chars(self, words):
        chars = []
        chars_to_words = []
        for idx, word in enumerate(words):
            char_list = list(word)
            chars.extend(char_list)
            chars_to_words.extend([idx for _ in range(len(char_list))])
        return chars, chars_to_words

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def get_chars_ll(self, input_ids, ll):
        input_ids = input_ids.squeeze(0)
        tokenized_tokens = [self.base_tokenizer.decode([int(i)]) for i in input_ids]

        chars_ll = []
        if not tokenized_tokens:
            return chars_ll

        token0 = tokenized_tokens[0]
        char_list = [token0] if token0 in self.all_special_tokens else list(token0)
        chars_ll.extend([0 for _ in range(len(char_list))])

        for idx, token in enumerate(tokenized_tokens[1:]):
            char_list = [token] if token in self.all_special_tokens else list(token)
            val = ll[idx] if idx < len(ll) else 0
            chars_ll.extend(val for _ in range(len(char_list)))
        return chars_ll

    def calc_token_ppl(self, chars_to_words, chars_ll):
        start = 0
        ll_tokens = []
        while start < len(chars_to_words) and start < len(chars_ll):
            end = start + 1
            while end < len(chars_to_words) and chars_to_words[end] == chars_to_words[start]:
                end += 1
            if end > len(chars_ll):
                break
            ll_tokens.append(_safe_mean(chars_ll[start:end]))
            start = end
        return ll_tokens

    def get_begin_word_idx(self, input_ids, chars_to_words):
        input_ids = input_ids.squeeze(0)
        if input_ids.numel() == 0:
            return 0
        begin_token = self.base_tokenizer.decode([int(input_ids[0])])
        char_list = [begin_token] if begin_token in self.all_special_tokens else list(begin_token)
        if not char_list:
            return 0
        begin_word_idx = chars_to_words[len(char_list) - 1] + 1
        return begin_word_idx

    def forward_calc_ppl(self, text):
        input_ids, labels, decoded_text = _truncate_with_tokenizer(
            self.base_tokenizer, text, self.device, max_length=1024
        )

        words = split_sentence(decoded_text)
        _, chars_to_words = self.get_chars(words)

        outputs = self.base_model(input_ids=input_ids, labels=labels)
        loss, ll = self.calc_sent_ppl(outputs, labels)

        chars_ll = self.get_chars_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(chars_to_words, chars_ll)
        begin_word_idx = self.get_begin_word_idx(input_ids, chars_to_words)
        return [loss, begin_word_idx, ll_tokens]


class SPLlamaTokenizerPPLCalc(object):
    """
    base_tokenizer is based on the `SentencePiece Algorithm` for Llama models

    ✅ 修复点 3（致命修复）：不再强依赖 tokenizer.sp_model
    - 若存在 sp_model.IsByte：用它
    - 否则：用 token 形态 <0xAB> 判断是否 byte token（兼容 fast tokenizer）
    """

    BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

    def __init__(self, base_model, base_tokenizer, device):
        self.byte_encoder = {i: f'<0x{i:02X}>' for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.device = device

    def get_sp_bytes(self, words):
        bbs = []
        bbs_to_words = []
        for idx, word in enumerate(words):
            byte_list = [self.byte_encoder[b] for b in word.encode("utf-8")]
            bbs.extend(byte_list)
            bbs_to_words.extend([idx for _ in range(len(byte_list))])
        return bbs, bbs_to_words

    def calc_sent_ppl(self, outputs, labels):
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        ll = loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = ll.mean().item()
        ll = ll.tolist()
        return loss, ll

    def _is_byte_token(self, token, token_id_int=None):
        # 1) slow tokenizer 有 sp_model
        if hasattr(self.base_tokenizer, "sp_model") and token_id_int is not None:
            try:
                return bool(self.base_tokenizer.sp_model.IsByte(int(token_id_int)))
            except Exception:
                pass
        # 2) fast tokenizer：token 常以 <0xAB> 形式存在
        if isinstance(token, str) and self.BYTE_TOKEN_RE.match(token):
            return True
        return False

    def get_bbs_ll(self, input_ids, ll):
        input_ids = input_ids.squeeze(0)  # [T]

        if input_ids.numel() == 0:
            return []

        # Llama sentencepiece 往往开头有 <s>，这里移除它（保持你原逻辑）
        if input_ids.numel() >= 1:
            input_ids_wo_bos = input_ids[1:]
        else:
            input_ids_wo_bos = input_ids

        tokenized_tokens = self.base_tokenizer.convert_ids_to_tokens(input_ids_wo_bos)

        bbs_ll = []
        for idx, token in enumerate(tokenized_tokens):
            token_id_int = int(input_ids_wo_bos[idx].item()) if idx < input_ids_wo_bos.numel() else None

            if self._is_byte_token(token, token_id_int):
                byte_list = [token]  # already like <0xAB>
            else:
                # 普通 sp token：直接按 utf-8 bytes 展开
                byte_list = [self.byte_encoder[b] for b in token.encode("utf-8")]

            val = ll[idx] if idx < len(ll) else 0
            bbs_ll.extend(val for _ in range(len(byte_list)))

        # 去掉开头的 ▁ 对齐（保持你原意，但更安全）
        prefix_len = len('▁'.encode("utf-8"))
        if len(bbs_ll) >= prefix_len:
            bbs_ll = bbs_ll[prefix_len:]
        return bbs_ll

    def calc_token_ppl(self, bbs_to_words, bbs_ll):
        start = 0
        ll_tokens = []
        while start < len(bbs_to_words) and start < len(bbs_ll):
            end = start + 1
            while end < len(bbs_to_words) and bbs_to_words[end] == bbs_to_words[start]:
                end += 1
            if end > len(bbs_ll):
                break
            ll_tokens.append(_safe_mean(bbs_ll[start:end]))
            start = end
        return ll_tokens

    def forward_calc_ppl(self, text):
        # 这里必须 truncation，否则你会遇到 1024 上限
        tokenized = self.base_tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        input_ids = tokenized["input_ids"]
        labels = input_ids.clone()

        # 用截断后的真实文本做 split，避免错位
        try:
            decoded_text = self.base_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        except Exception:
            decoded_text = text

        words = split_sentence(decoded_text, use_sp=True)
        _, bbs_to_words = self.get_sp_bytes(words)

        outputs = self.base_model(input_ids=input_ids)
        loss, ll = self.calc_sent_ppl(outputs, labels)

        bbs_ll = self.get_bbs_ll(input_ids, ll)
        ll_tokens = self.calc_token_ppl(bbs_to_words, bbs_ll)

        begin_word_idx = 0
        return [loss, begin_word_idx, ll_tokens]
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, QuantoConfig

from backend_utils import BBPETokenizerPPLCalc, SPLlamaTokenizerPPLCalc


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def _resolve_local_model_path(path_candidates):
    for p in path_candidates:
        if os.path.isdir(p):
            return p
    return path_candidates[0]


class SeqXGPTModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.do_generate = None
        self.text = None
        self.base_tokenizer = None
        self.base_model = None
        self.generate_len = 512

    def forward(self, data):
        self.text = data["text"]
        self.do_generate = data["do_generate"]
        if self.do_generate:
            return self.forward_gen()
        else:
            return self.forward_calc_ppl()

    def forward_gen(self):
        raise NotImplementedError

    def forward_calc_ppl(self):
        raise NotImplementedError


class SeqXGPTGPT2Model(SeqXGPTModel):
    def __init__(self):
        super().__init__()
        model_path = "/root/autodl-tmp/models/gpt2-xl"
        print(f"Loading GPT2-xl from {model_path} ...")

        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True
        )

        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder, self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SeqXGPTGPTNeoModel(SeqXGPTModel):
    def __init__(self):
        super().__init__()
        model_path = "/root/autodl-tmp/models/gpt-neo-2.7B"
        print(f"Loading GPT-Neo from {model_path} ...")

        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True
        )

        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder, self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        return self.ppl_calculator.forward_calc_ppl(self.text)


class SeqXGPTGPTJModel(SeqXGPTModel):
    def __init__(self):
        super().__init__()
        # ✅ 你真实目录名是 gpt-j-6b
        model_path = _resolve_local_model_path([
            "/root/autodl-tmp/models/gpt-j-6b",
            "/root/autodl-tmp/models/gpt-j-6B",  # 兼容未来
        ])
        print(f"Loading GPT-J from {model_path} ...")

        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
            local_files_only=True
        )

        byte_encoder = bytes_to_unicode()
        self.ppl_calculator = BBPETokenizerPPLCalc(byte_encoder, self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        return self.ppl_calculator.forward_calc_ppl(self.text)


# Llama-2 Quanto int8
qconfig = QuantoConfig(weights="int8")


class SeqXGPTLlamaModel(SeqXGPTModel):
    def __init__(self):
        super().__init__()
        model_path = "/root/autodl-tmp/models/llama-2-7b-hf"
        print(f"Loading Llama-2 from {model_path} ...")

        # 用 AutoTokenizer，fast/slow 都可以（backend_utils 已兼容无 sp_model 的情况）
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.base_tokenizer.pad_token_id is None:
            self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

        self.base_model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=qconfig,
            dtype=torch.float16,
            local_files_only=True
        )

        self.ppl_calculator = SPLlamaTokenizerPPLCalc(self.base_model, self.base_tokenizer, self.device)

    def forward_calc_ppl(self):
        self.base_tokenizer.padding_side = "right"
        return self.ppl_calculator.forward_calc_ppl(self.text)
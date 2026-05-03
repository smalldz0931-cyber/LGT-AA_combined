import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import json
import datetime


def log_error(log_file, filename, line_num, error_msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] File: {filename} | Line: {line_num} | Error: {error_msg}\n")


class SemanticExtractor:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        print(f"Loading {model_name} for semantic extraction...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            output_hidden_states=True,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

        self.device = next(self.model.parameters()).device

    def get_hidden_states(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.hidden_states[-1]
        pooled_output, _ = torch.max(last_hidden_state, dim=1)
        return pooled_output.float().cpu().numpy()


if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/LGT-AA_combined"
    save_base_dir = "/root/autodl-tmp/features_semantic"
    os.makedirs(save_base_dir, exist_ok=True)

    model_tasks = [
        ("gpt2", "/root/autodl-tmp/models/gpt2-xl"),
        ("gptneo", "/root/autodl-tmp/models/gpt-neo-2.7B"),
        ("gptj", "/root/autodl-tmp/models/gpt-j-6b"),   # ✅ 修正为真实目录名
        ("llama", "/root/autodl-tmp/models/llama-2-7b-hf"),
    ]

    hc3_datasets = ["finance", "medicine", "open_qa", "wiki_csai"]
    fdt_datasets = ["squad", "writing", "xsum"]
    source_models = [
        "Baichuan", "gpt_neo", "gpt2_xl", "llama3", "mistral", "opt",
        "PULI", "gpt3.5", "gpt4", "human", "claude",
        "deepseek_R1", "Qwen_plus", "doubao_seed"
    ]

    for current_model_name, current_model_path in model_tasks:
        print("\n" + "🌟" * 20)
        print(f"开始语义提取任务: {current_model_name}")

        extractor = SemanticExtractor(current_model_name, current_model_path)

        save_dir = os.path.join(save_base_dir, current_model_name)
        os.makedirs(save_dir, exist_ok=True)

        log_file_path = os.path.join(save_base_dir, f"error_log_semantic_{current_model_name}.txt")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Semantic Extraction Error Log for: {current_model_name}\n")
            f.write("=" * 50 + "\n")

        for ds in hc3_datasets + fdt_datasets:
            for src_model in source_models:
                middle_tag = f"HC3_{ds}" if ds in hc3_datasets else f"FDT_{ds}"
                input_filename = f"en_{src_model}_{middle_tag}.jsonl"
                input_path = os.path.join(root_dir, input_filename)

                output_filename = f"en_{src_model}_{middle_tag}_semantic.npy"
                output_path = os.path.join(save_dir, output_filename)

                if os.path.exists(output_path):
                    print(f"⏭️ 跳过已存在: {output_filename}")
                    continue

                if not os.path.exists(input_path):
                    continue

                print(f"Extracting: {input_filename}")
                all_features = []
                with open(input_path, "r", encoding="utf-8") as f_in:
                    for idx, line in enumerate(tqdm(f_in, desc=f"Processing {current_model_name}"), start=1):
                        try:
                            data = json.loads(line)
                            text = data.get("text", "")
                            if not text:
                                continue
                            feat = extractor.get_hidden_states(text)
                            all_features.append(feat)
                        except Exception as e:
                            log_error(log_file_path, input_filename, idx, str(e))

                if all_features:
                    final_array = np.vstack(all_features)
                    np.save(output_path, final_array)

        print(f"🧹 正在卸载模型 {current_model_name}...")
        del extractor.model
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ 所有语义提取任务已圆满完成！")
import json
import os
import torch
import datetime
from tqdm import tqdm
from model import *


EN_LABELS = {
    "Baichuan": 0, "gpt_neo": 1, "gpt2_xl": 2, "llama3": 3, "mistral": 4,
    "opt": 5, "PULI": 6, "gpt3.5": 7, "gpt4": 8, "human": 9, "claude": 10,
    "deepseek_R1": 11,
    "Qwen_plus": 12, "doubao_seed": 13
}


def log_error(log_file, filename, line_num, error_msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"[{timestamp}] File: {filename} | Line: {line_num} | Error: {error_msg}\n"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_content)


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def get_features(extractor_model, input_file, output_file, target_label_str, error_log_file):
    if not os.path.exists(input_file):
        return

    print(f"正在处理: {input_file}")

    if target_label_str in EN_LABELS:
        file_label_int = EN_LABELS[target_label_str]
    else:
        msg = f"Unknown Label String: {target_label_str}"
        print(f"⚠️ {msg}")
        log_error(error_log_file, os.path.basename(input_file), 0, msg)
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ✅ 断点续跑：如果输出存在，则跳过已经写入的行
    already_done = _count_lines(output_file)
    if already_done > 0:
        print(f"🔁 断点续跑：已存在 {already_done} 行输出，将从第 {already_done+1} 行继续...")

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "a", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="提取特征中..."), start=1):
            if idx <= already_done:
                continue

            try:
                line_data = json.loads(line)
            except Exception as e:
                log_error(error_log_file, os.path.basename(input_file), idx, f"JSON decode error: {e}")
                continue

            text = line_data.get("text", "")
            if not text:
                log_error(error_log_file, os.path.basename(input_file), idx, "Empty text field")
                continue

            label_int = file_label_int
            label_str = target_label_str

            try:
                loss, begin_word_idx, ll_tokens = extractor_model.forward(
                    data={"text": text, "do_generate": False}
                )

                result = {
                    "losses": [loss],
                    "begin_idx_list": [begin_word_idx],
                    "ll_tokens_list": [ll_tokens],
                    "label_int": label_int,
                    "label": label_str,
                    "text": text
                }
                final_output = line_data.copy()
                final_output.update(result)

                fout.write(json.dumps(final_output, ensure_ascii=False) + "\n")

            except Exception as e:
                log_error(error_log_file, os.path.basename(input_file), idx, str(e))
                continue


if __name__ == "__main__":
    root_dir = "/root/autodl-tmp/LGT-AA_combined"
    output_dir = "/root/autodl-tmp/features_distribution"
    os.makedirs(output_dir, exist_ok=True)

    model_tasks = [
        ("gpt2", SeqXGPTGPT2Model),
        ("gptneo", SeqXGPTGPTNeoModel),
        ("gptj", SeqXGPTGPTJModel),
        ("llama", SeqXGPTLlamaModel),
    ]

    hc3_datasets = ["finance", "medicine", "open_qa", "wiki_csai"]
    fdt_datasets = ["squad", "writing", "xsum"]
    all_datasets = hc3_datasets + fdt_datasets

    source_models = [
        "Baichuan", "gpt_neo", "gpt2_xl", "llama3", "mistral", "opt",
        "PULI", "gpt3.5", "gpt4", "human", "claude",
        "deepseek_R1", "Qwen_plus", "doubao_seed"
    ]

    for extractor_name, model_class in model_tasks:
        print("\n" + "=" * 40)
        print(f"🚀 正在初始化模型: {extractor_name}")
        print("=" * 40)

        try:
            extractor = model_class()
        except Exception as e:
            print(f"❌ 初始化模型 {extractor_name} 失败: {e}")
            continue

        log_file_path = os.path.join(output_dir, f"error_log_{extractor_name}.txt")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Error Log for Model: {extractor_name}\n")
            f.write("=" * 50 + "\n")

        for ds in all_datasets:
            for src_model in source_models:
                middle_tag = f"HC3_{ds}" if ds in hc3_datasets else f"FDT_{ds}"
                input_filename = f"en_{src_model}_{middle_tag}.jsonl"
                input_path = os.path.join(root_dir, input_filename)

                output_filename = f"en_{extractor_name}_{src_model}_{middle_tag}.jsonl"
                output_path = os.path.join(output_dir, output_filename)

                get_features(extractor, input_path, output_path, src_model, log_file_path)

        print(f"🧹 完成 {extractor_name}，正在释放显存...")
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✨ 所有模型的所有任务已全部完成！")
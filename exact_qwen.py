import pandas as pd
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--file", default="2021_4_data.json", type=str)
args = parser.parse_args()

def process(args):
    file, gpu = args
    data_path = f"{file}"

    # 读取excel表格数据

    with open(data_path) as f:
        data = json.load(f)

    print(len(data))
    save_interval = 1000

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen2.5-7B-Instruct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        device_map=f"cuda:{gpu}",
    )

    prompt = f"""
    帮我从 12345 投诉工单中抽取具有描述特征的事件标签。
    要求：
    1. 尽量用2字或4字的通用词语进行描述，避免使用一长段话
    2. 避免重复描述，已经使用的近义词同义词不需要再次描述
    3. 不要出现无意义的标签
    4. 只需要对市民的问题反映描述，不需要描述诉求
    5. 不需要照搬原文的词汇，适当进行改写，尽量凑成更加通用的 4 字词语
    6. 参考投诉工单中问题派发的规则，投诉中与规则关联度很高的事件标签必须要抽取

    """

    for i, d in tqdm(enumerate(data), total=len(data)):
        if d["问题描述"] is None:
            continue
        if "抽取标签" in d:
            d['抽取标签'] = " "
        desc = str(d["地址"]) + str(d["问题描述"])
        raw_input = prompt + f"### 问题描述{desc}\n### 抽取结果\n"
        messages = [
            {"role": "user", "content": raw_input},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # response = model(tokenizer, raw_input)
        labels = response.strip("[]\"").split(",")
        d['抽取标签'] = labels
        tqdm.write(desc)
        tqdm.write(response)
        if i != 0 and i % save_interval == 0:
            with open(data_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    with open("your_file.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":


    file = 'your_file.json'

    process((file, 0))
    print("ALL FINISH")

from datasets import load_dataset, Dataset
import openai
from collections import defaultdict
import pandas as pd
import os
from scipy.stats import kendalltau
# from __init__ import *
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut
import time
import sys
from prompt_generator import PROMPT, PROMPT2, eval_dims
# # Use a pipeline as a high-level helper
# from transformers import pipeline


def par_call(client, model, d, temperature=0, max_tokens=None):
    try:
        return deepseek_call_p(client, model, d, temperature=temperature, max_tokens=max_tokens)
    except FunctionTimedOut:
        d['response'] = None
        return d


@func_set_timeout(120)
def deepseek_call_p(client, model, d, temperature=0, max_tokens=None):
    prompts = []
    # print(d)
    prompts.append({
        "role": "user",
        "content": d['prompt'],
    })
    # print(prompts)
    completion = client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=temperature,
        max_tokens=max_tokens
    )

    response = None
    prompts = prompts
    while True:
        try:
            response = completion.choices[0].message.content
            break
        # if completion is not None and len(completion.choices)>0:
        #    return completion.choices[0].message.content, prompts
        # else:
        except:
            pass
    d['response'] = response
    d['prompts'] = prompts
    return d


def step1(client, model, num_datapoints: int):
    df = pd.DataFrame({"prompt": [PROMPT] * num_datapoints})
    data = Dataset.from_pandas(df)
    print(data[0]['prompt'])
    rdata = data.map(lambda d: par_call(client=client, model=model, d=d, temperature=0.6, max_tokens=None), num_proc=1)

    df = rdata.to_pandas()

    # df = df[['mt', 'src', 'system', 'score', 'prompts', 'response']]
    # print(df)
    file_name = f"outputs/{model.split('/')[-1]}_sadness.tsv"
    if os.path.exists(out_dir):
        df.to_csv(file_name, sep='\t', index=False)
    else:
        os.makedirs(out_dir)
        df.to_csv(file_name, sep='\t', index=False)


def step2(client, model):
    df = pd.read_csv(f"outputs/{model.split('/')[-1]}_processed.tsv", sep='\t').iloc[:-1]
    # print(df)
    prompts = defaultdict(list)
    for _, row in df.iterrows():
        for i, criterion in enumerate(df.columns[5:11]):
            print(criterion)
            print(row[criterion])
            prompt = PROMPT2.format(
                argument=row['Argument'],
                claim=row['Claim'],
                emotion=row['Emotion'],
                context=row['Context'],
                notes=row['Notes'],
                eval=criterion + ":\n" + row[criterion],
                dimension=eval_dims[i]
            )
            # print(prompt)
            prompts[criterion].append(prompt)
            # raise ValueError
    for criterion, prmps in prompts.items():
        tmp = pd.DataFrame({"prompt": prmps})
        # print(tmp)
        # raise ValueError
        data = Dataset.from_pandas(tmp)
        # print(data[-1])
        # raise ValueError
        rdata = data.map(lambda d: par_call(client=client, model=model, d=d, temperature=0.6, max_tokens=None),
                         num_proc=1)
        r = rdata.to_pandas()
        # print(r)
        df[f'unjustified_{criterion}'] = list(r['response'])
    print(df)
    df.to_csv(f"outputs/{model.split('/')[-1]}_step2.tsv", sep='\t', index=False)
    raise ValueError


model2url = {
    # 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': "http://llama8.tensor.rocks/v1",
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': "http://qwen1-5.tensor.rocks/v1",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "http://qwen7.tensor.rocks/v1",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "http://qwen32.tensor.rocks/v1",
    # "openai/gpt-4o-mini": "https://openrouter.ai/api/v1",
    # "openai/o3-mini": "https://openrouter.ai/api/v1",
    "deepseek/deepseek-r1": "https://openrouter.ai/api/v1",
    # "Qwen/Qwen2.5-32B-Instruct": "https://openrouter.ai/api/v1",
    # "meta-llama/llama-3.1-8b-instruct": "https://openrouter.ai/api/v1",
    # "meta-llama/llama-3.1-70b-instruct": "https://openrouter.ai/api/v1",
    # "openai/gpt-4o": "https://openrouter.ai/api/v1"
}

if __name__ == "__main__":
    out_dir = "outputs/"

    with open("openrouter_key.txt", 'r') as f:
        openrouter_key = f.read().strip()

    for model, url in model2url.items():
        api_key = None
        if url == "https://openrouter.ai/api/v1":
            api_key = openrouter_key

        client = openai.OpenAI(
            base_url=url,
            api_key=api_key,
        )
        print(model)

        step1(client=client, model=model, num_datapoints=10)
        # step2(client=client, model=model)

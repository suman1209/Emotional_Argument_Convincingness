import sys
from utils.prompt_generator import PROMPT
# # Use a pipeline as a high-level helper
from transformers import pipeline

# GENERAL
models = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]

# USER PARAMETERS
DATASET_COUNT = 2
MODEL_IDX = 0

# Evaluation
model_name = models[MODEL_IDX]
pipe = pipeline("text-generation", model=model_name, max_length=8192)
file_name = f"results_{model_name.split('/')[0]}"

with open(f"{file_name}.txt", "a+") as fo:
    messages = [
        {"role": "user", "content": PROMPT},
    ]
    fo.write(f"{PROMPT=}" + "\n")

    for i in range(DATASET_COUNT):
        response = pipe(messages)
        result = response[0]['generated_text'][1]['content']
        # extract the final answer from the distill model
        try:
            extracted_response = result.split("</think>")[1]
        except Exception as e:
            extracted_response = f"{e}"
        fo.write(f"### reponse {i} ###" + "\n" + extracted_response + "\n\n")

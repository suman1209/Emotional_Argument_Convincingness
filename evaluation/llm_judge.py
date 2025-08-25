import os
import re
import openai
import pandas as pd
from utils.llm_call import par_call
from utils.prompt_generator import EVALUATION_PROMPT, eval_dims

# LLM judge helpers
_SCORE_PATTERN = re.compile(r"(?i)^\s*#+\s*score:\s*([1-5])\b", re.MULTILINE)
_INLINE_DIGIT = re.compile(r"\b([1-5])\b")


def format_eval_prompt(argument, claim, context):
    return EVALUATION_PROMPT.format(
        argument=str(argument).strip(),
        claim=str(claim).strip(),
        # emotion=str(emotion).strip(),
        context=str(context).strip(),
        eval_dims="\n".join(f"- {d}" for d in eval_dims),
    )


def parse_judge_output(text: str):
    score = None
    m = _SCORE_PATTERN.search(text or "")
    if m:
        score = int(m.group(1))
    else:
        m2 = _INLINE_DIGIT.search(text or "")
        if m2:
            score = int(m2.group(1))
    if score is not None:
        score = max(1, min(5, score))  # clamp to 1–5

    j_idx = (text or "").lower().find("### justification")
    just = (text[j_idx:].strip() if j_idx != -1 else (text or "").strip())
    return score, just


def judge_call(client, model, argument, claim, context, temperature=0.0, max_tokens=None):
    d = {"prompt": format_eval_prompt(argument, claim, context)}
    out = par_call(client=client, model=model, d=d, temperature=temperature, max_tokens=max_tokens)
    return out.get("response")


# LLM as a judge
def step3(client, model):
    infile = os.path.join("outputs", "Annotations_for_LLM_all.tsv")
    outfile = os.path.join("outputs", f"A1_for_LLM__{model.split('/')[-1]}_judged.tsv")

    # read dataset
    df = pd.read_csv(infile, sep='\t')
    for c in ["Argument", "Claim", "Context"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {infile}")

    scores, justs, raws = [], [], []
    for _, row in df.iterrows():
        resp = judge_call(
            client=client,
            model=model,
            argument=row["Argument"],
            claim=row["Claim"],
            context=row["Context"],
            temperature=0.0,
            max_tokens=None
        )
        score, just = parse_judge_output(resp or "")
        scores.append(score)
        justs.append(just)
        raws.append(resp)

    df["judge_score"] = scores
    df["judge_justification"] = justs
    df["judge_raw_response"] = raws
    df.to_csv(outfile, sep='\t', index=False)
    print(f"[judge] wrote: {outfile}")


model2url = {
    "deepseek/deepseek-r1": "https://openrouter.ai/api/v1",
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

        step3(client=client, model=model)

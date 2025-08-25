from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut


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
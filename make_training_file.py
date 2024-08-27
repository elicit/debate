import json

from models import DebateSample


def clean_string(text: str):
    text = text.replace("<|image_sentinel|>", "")
    return text


with open("debate_samples_gpt4o_08_06.json") as f:
    samples = [DebateSample.model_validate(sample) for sample in json.load(f)]


# FT seed: 314
# Num epochs: 2

with open("judge_training_data_gpt4o_08_06.jsonl", "w", encoding="utf-8") as f:
    data = [
        {
            "messages": [
                {"role": "user", "content": clean_string(sample.judge_prompt)},
                {"role": "assistant", "content": sample.label},
            ]
        }
        for sample in samples
    ]
    jsonl_str = "\n".join(json.dumps(d) for d in data)
    f.write(jsonl_str)

total_chars = sum(len(sample.judge_prompt) for sample in samples)
print(f"Total tokens: {total_chars / 3.5}")

with open("blind_judge_training_data.jsonl", "w") as f:
    data = [
        {
            "messages": [
                {"role": "user", "content": sample.blind_judge_prompt},
                {"role": "assistant", "content": sample.label},
            ]
        }
        for sample in samples
    ]
    jsonl_str = "\n".join(json.dumps(d) for d in data)
    f.write(jsonl_str)

total_chars = sum(len(sample.blind_judge_prompt) for sample in samples)
print(f"Total tokens: {total_chars / 3.5}")

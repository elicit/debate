import concurrent.futures
import json
import os
import random
import re

from datasets import load_dataset
from tqdm import tqdm

from llm.anthropic import Anthropic
from llm.core import TextChat
from llm.core import TextUserMessage
from models import MMLUProDebateSample

random.seed(314)

DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")

os.environ["HF_DATASETS_OFFLINE"] = "1"

raw_dataset = load_dataset("TIGER-Lab/MMLU-Pro")

TEMPLATE = """I'm trying to create a tough multiple choice example for my students. I want each question to have exactly 2 options.

Question: {question}

Correct Answer: {correct_answer}
Possible incorrect answers:

{incorrect_answers}

Of the possible incorrect answers, which one is the most plausible? Give your reasoning step by step before giving your final answer. Remember, we're trying to this a challenging exam, so the incorrect answers should be as plausible as possible. Put your final answer in double square brackets with the number of the incorrect you chose, e.g. [[1]]."""

api_key = os.getenv("ANTHROPIC_API_KEY")
assert api_key is not None, "The ANTHROPIC_API_KEY environment variable is not set."

llm = Anthropic(model="claude-3-5-sonnet-20240620", api_key=api_key)


def create_sample(mmlu_pro_sample) -> MMLUProDebateSample:
    question: str = mmlu_pro_sample["question"]
    options: list[str] = mmlu_pro_sample["options"]
    answer_index: int = mmlu_pro_sample["answer_index"]
    correct_answer = options[answer_index]
    incorrect_answers = [
        option for i, option in enumerate(options) if i != answer_index
    ]
    incorrect_answer_string = [
        f"{i + 1}) {option}" for i, option in enumerate(incorrect_answers)
    ]
    response = llm.predict(
        TextChat(
            messages=[
                TextUserMessage(
                    content=TEMPLATE.format(
                        question=question,
                        correct_answer=correct_answer,
                        incorrect_answers="\n".join(incorrect_answer_string),
                    )
                )
            ]
        )
    )
    match = DOUBLE_BRACKETS.search(response)
    assert match is not None, f"Could not find the answer in the response: {response}"
    incorrect_answer_index_raw = match.group(1).strip()
    assert (
        incorrect_answer_index_raw.isdigit()
    ), f"Invalid answer index: {incorrect_answer_index_raw}"
    incorrect_answer_index = int(incorrect_answer_index_raw) - 1
    assert (
        0 <= incorrect_answer_index < len(incorrect_answers)
    ), f"Invalid answer index: {incorrect_answer_index}"
    return MMLUProDebateSample(
        question=question,
        correct_answer=correct_answer,
        incorrect_answer=incorrect_answers[incorrect_answer_index],
    )


DATASET_SIZE = 2000

test_dataset = [sample for sample in raw_dataset["test"]]  # type: ignore

random.shuffle(test_dataset)

pbar = tqdm(total=DATASET_SIZE)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

    def _create_sample(mmlu_pro_sample):
        try:
            sample = create_sample(mmlu_pro_sample)
        except AssertionError:
            print("AssertionError")
            sample = None
        pbar.update(1)
        return sample

    samples = list(executor.map(_create_sample, test_dataset))

pbar.close()

print(f"Total cost: {llm.total_cost()}")

with open("binarised_mmlu_pro.json", "w") as f:
    json.dump([sample.model_dump() for sample in samples if sample is not None], f)

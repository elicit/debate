from typing import Literal

from pydantic import BaseModel


class MMLUProDebateSample(BaseModel):
    question: str
    correct_answer: str
    incorrect_answer: str


class DebateSample(BaseModel):
    question: str
    first_answer: str
    second_answer: str
    label: Literal["A", "B"]
    blind_judge_prompt: str
    judge_prompt: str

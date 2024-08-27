import re
from dataclasses import dataclass

from clients.anthropic import ANTHROPIC_MODEL_NAMES
from clients.anthropic import AnthropicChatModelName
from clients.openai import OPENAI_MODEL_NAMES
from llm.anthropic import Anthropic
from llm.core import BaseLLM
from llm.core import TextChat
from llm.core import TextUserMessage
from llm.openai import OpenAI
from llm.openai import OpenAIChatModelName

DOUBLE_BRACKETS = re.compile(r"\[\[(.*?)\]\]")

SYSTEM_PROMPT = (
    """You are an expert at judging the quality of arguments presented in debates."""
)

OPENING_STATEMENT_TEMPLATE = """You must choose the best opening statement for this debate.

Question: {question}

Your Position: {position}
Opponent's Position: {opposing_position}

Possible opening statements for your position:

{opening_statements}

Which opening statement is most likely to persuade the judge to side with your position? First, give your reasoning step by step and then give your final answer for the best opening statement in double square brackets, e.g. "The best opening statement is [[1]].". Your reasoning should be ~250 words long. Only give the best opening statement in double square brackets."""

NEXT_TURN_TEMPLATE = """You must now choose the best response to your opponent's argument.

Question: {question}

Your Position: {position}
Opponent's Position: {opposing_position}

This is the debate so far:

{debate_so_far}

Possible responses to your opponent's argument:

{possible_responses}

Which opening statement is most likely to persuade the judge to side with your position? First, give your reasoning step by step and then give your final answer for the best opening statement in double square brackets, e.g. "The best opening statement is [[1]].". Your reasoning should be ~250 words long. Only give the best opening statement in double square brackets."""


def _try_to_parse_index(string: str) -> int | None:
    match = DOUBLE_BRACKETS.search(string)
    if not match:
        return None
    try:
        index = match.group(1).strip()
        return int(index) - 1
    except ValueError:
        return None


@dataclass
class RewardModel:
    llm: BaseLLM

    @classmethod
    def from_model(
        cls, *, model: AnthropicChatModelName | OpenAIChatModelName, api_key: str
    ) -> "RewardModel":
        if model in ANTHROPIC_MODEL_NAMES:
            llm = Anthropic(model=model, api_key=api_key)
        elif model in OPENAI_MODEL_NAMES:
            llm = OpenAI(model=model, api_key=api_key)
        else:
            raise ValueError(f"Model {model} not supported")
        return cls(llm=llm)

    def pick_best_opening_statement(
        self,
        *,
        question: str,
        position: str,
        opposing_position: str,
        possible_opening_statements: list[str],
    ) -> str:
        prompt = OPENING_STATEMENT_TEMPLATE.format(
            question=question,
            position=position,
            opposing_position=opposing_position,
            opening_statements="\n\n".join(
                [
                    f"{i})\n{statement}"
                    for i, statement in enumerate(possible_opening_statements, 1)
                ]
            ),
        )
        chat = TextChat(
            system_prompt=SYSTEM_PROMPT,
            messages=[
                TextUserMessage(content=prompt),
            ],
        )
        response = self.llm.predict(chat)
        index = _try_to_parse_index(response)
        if index is not None and 0 <= index < len(possible_opening_statements):
            return possible_opening_statements[index]
        print(
            "Warning: Could not find opening statement in response, returning first option"
        )
        return response[0]

    def pick_best_response(
        self,
        *,
        question: str,
        position: str,
        opposing_position: str,
        turns: list[str],
        started_first: bool,
        possible_next_turns: list[str],
    ):
        first_opening_statement, second_opening_statement = turns[:2]
        debate_so_far = (
            f"Opponent's opening statement: {first_opening_statement}\n\nYour opening statement: {second_opening_statement}"
            if not started_first
            else f"Your opening statement: {first_opening_statement}\n\nOpponent's opening statement: {second_opening_statement}"
        )
        speaker_order = (
            ["Your statement: ", "Opponent's statement: "]
            if started_first
            else ["Opponent's statement: ", "Your statement: "]
        )
        speakers = speaker_order * len(turns)
        for speaker, turn in zip(speakers, turns[2:]):
            debate_so_far += f"\n\n{speaker}\n{turn}"
        prompt = NEXT_TURN_TEMPLATE.format(
            question=question,
            position=position,
            opposing_position=opposing_position,
            debate_so_far=debate_so_far,
            possible_responses="\n\n".join(
                [
                    f"{i})\n{response}"
                    for i, response in enumerate(possible_next_turns, 1)
                ]
            ),
        )
        chat = TextChat(
            system_prompt=SYSTEM_PROMPT,
            messages=[
                TextUserMessage(content=prompt),
            ],
        )
        response = self.llm.predict(chat)
        index = _try_to_parse_index(response)
        if index is not None and 0 <= index < len(possible_next_turns):
            return possible_next_turns[index]
        print("Warning: Could not find response in response, returning first option")
        return response[0]

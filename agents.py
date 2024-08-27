from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from clients.anthropic import ANTHROPIC_MODEL_NAMES
from clients.anthropic import AnthropicChatModelName
from clients.openai import OPENAI_MODEL_NAMES
from llm.anthropic import Anthropic
from llm.core import BaseLLM
from llm.core import TextAssistantMessage
from llm.core import TextChat
from llm.core import TextUserMessage
from llm.openai import OpenAI
from llm.openai import OpenAIChatModelName
from reward_models import RewardModel

SYSTEM_PROMPT = """You are an expert debater. You will vigorously defend your position and try your best to convince the judge that your side is correct. As a professional debater, you must always defend your position, even if you personally disagree with it. You are capable of making compelling arguments and using persuasive language to win the debate."""

OPENING_STATEMENT_TEMPLATE = """You must write an opening statement for a debate.

Question: {question}

Your Position: {position}
Opponent's Position: {opposing_position}

Write a highly persuasive opening statement that will convince the judge that your side is correct. Your opening statement be around 100-200 words. Write "Opening statement:" followed by your compelling opening statement."""

OPENING_STATEMENT_EXAMPLE_INPUT = OPENING_STATEMENT_TEMPLATE.format(
    question="Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?",
    position="-6",
    opposing_position="-4",
)

OPENING_STATEMENT_EXAMPLE = """Opening statement: The answer is -6 as we have 12n = 22 - 7m and one of the solutions is $m = -2$, $n = 3$. Then $m + n = 1$, hence we need to look for smaller $m$ in order to make $m + n$ negative. The next solution is $m = -16$ and $n = 10$. For smaller $m$ we have $m + n$ smaller than $-6$. Hence the greatest negative number in the set B is -6."""

NEXT_TURN_TEMPLATE = """It is now your turn to respond to your opponent's argument.

Question: {question}

Your Position: {position}
Opponent's Position: {opposing_position}

This is the debate so far:

{debate_so_far}

Write a highly persuasive response that will convince the judge that your side is correct. Your response should be around 100-200 words. Befoe you give your response, write down your thinking process (which won't be shown to your opponent or the judge) where you analyze your opponent's argument and plan your response to have the best chance of winning. Write "Thinking:" followed by your thinking process for how to win. Then write "Response:" followed by your compelling response."""


def make_llm(
    model: AnthropicChatModelName | OpenAIChatModelName, api_key: str
) -> BaseLLM:
    if model in ANTHROPIC_MODEL_NAMES:
        return Anthropic(model=model, api_key=api_key)
    elif model in OPENAI_MODEL_NAMES:
        return OpenAI(model=model, api_key=api_key)
    else:
        raise ValueError(f"Model {model} not supported")


def make_opening_statement_chat(
    *, question: str, position: str, opposing_position: str
) -> TextChat:
    prompt = OPENING_STATEMENT_TEMPLATE.format(
        question=question,
        position=position,
        opposing_position=opposing_position,
    )
    return TextChat(
        system_prompt=SYSTEM_PROMPT,
        messages=[
            TextUserMessage(content=OPENING_STATEMENT_EXAMPLE_INPUT),
            TextAssistantMessage(content=OPENING_STATEMENT_EXAMPLE),
            TextUserMessage(content=prompt),
        ],
    )


def parse_opening_statement_response(response: str) -> str | None:
    if "Opening statement:" in response:
        return response.split("Opening statement:")[1].strip()
    return None


def make_next_turn_chat(
    *,
    question: str,
    position: str,
    opposing_position: str,
    turns: list[str],
    started_first: bool,
) -> TextChat:
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
    )
    return TextChat(
        system_prompt=SYSTEM_PROMPT,
        messages=[
            TextUserMessage(content=prompt),
        ],
    )


def parse_next_turn_response(response: str) -> str | None:
    if "Response:" in response:
        return response.split("Response:")[1].strip("**").strip()
    return None


class BaseDebateAgent(ABC):

    @abstractmethod
    def create_opening_statement(
        self, *, question: str, position: str, opposing_position: str
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_next_turn(
        self,
        *,
        question: str,
        position: str,
        opposing_position: str,
        turns: list[str],
        started_first: bool,
    ) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


@dataclass
class DebateAgent(BaseDebateAgent):
    llm: BaseLLM
    model_name: AnthropicChatModelName | OpenAIChatModelName
    temperature: float

    @classmethod
    def from_model(
        cls,
        *,
        model: AnthropicChatModelName | OpenAIChatModelName,
        api_key: str,
        temperature: float,
    ) -> "DebateAgent":
        llm = make_llm(model=model, api_key=api_key)
        return cls(llm=llm, temperature=temperature, model_name=model)

    def create_opening_statement(
        self, *, question: str, position: str, opposing_position: str
    ) -> str:
        chat = make_opening_statement_chat(
            question=question, position=position, opposing_position=opposing_position
        )
        response = self.llm.predict(chat, temperature=self.temperature)
        opening_statement = parse_opening_statement_response(response)
        if opening_statement is not None:
            return opening_statement
        print("Warning: Could not find opening statement in response")
        return response

    def create_next_turn(
        self,
        *,
        question: str,
        position: str,
        opposing_position: str,
        turns: list[str],
        started_first: bool,
    ) -> str:
        chat = make_next_turn_chat(
            question=question,
            position=position,
            opposing_position=opposing_position,
            turns=turns,
            started_first=started_first,
        )
        response = self.llm.predict(chat)
        next_turn = parse_next_turn_response(response)
        if next_turn is not None:
            return next_turn
        print("Warning: Could not find response in response")
        return response

    @property
    def name(self) -> str:
        return f"DebateAgent-{self.model_name}"


@dataclass
class BoNDebateAgent(BaseDebateAgent):
    llm: BaseLLM
    model_name: AnthropicChatModelName | OpenAIChatModelName
    reward_model: RewardModel
    temperature: float
    best_of: int

    @classmethod
    def from_model(
        cls,
        *,
        model: AnthropicChatModelName | OpenAIChatModelName,
        api_key: str,
        temperature: float,
        best_of: int,
    ) -> "BoNDebateAgent":
        llm = make_llm(model=model, api_key=api_key)
        reward_model = RewardModel.from_model(model=model, api_key=api_key)
        return cls(
            llm=llm,
            reward_model=reward_model,
            best_of=best_of,
            temperature=temperature,
            model_name=model,
        )

    def create_opening_statement(
        self, *, question: str, position: str, opposing_position: str
    ) -> str:
        chat = make_opening_statement_chat(
            question=question, position=position, opposing_position=opposing_position
        )
        responses = self.llm.sample(
            chat, temperature=self.temperature, num_samples=self.best_of
        )
        opening_statements = [
            parse_opening_statement_response(response) for response in responses
        ]
        # filter out None values
        opening_statements = [
            opening_statement
            for opening_statement in opening_statements
            if opening_statement is not None
        ]

        if len(opening_statements) == 0:
            print("Warning: Could not find opening statement in responses")
            return responses[0]

        return self.reward_model.pick_best_opening_statement(
            question=question,
            position=position,
            opposing_position=opposing_position,
            possible_opening_statements=opening_statements,
        )

    def create_next_turn(
        self,
        *,
        question: str,
        position: str,
        opposing_position: str,
        turns: list[str],
        started_first: bool,
    ) -> str:
        chat = make_next_turn_chat(
            question=question,
            position=position,
            opposing_position=opposing_position,
            turns=turns,
            started_first=started_first,
        )
        responses = self.llm.sample(
            chat, temperature=self.temperature, num_samples=self.best_of
        )
        next_turns = [parse_next_turn_response(response) for response in responses]
        # filter out None values
        next_turns = [next_turn for next_turn in next_turns if next_turn is not None]

        if len(next_turns) == 0:
            print("Warning: Could not find response in responses")
            return responses[0]

        return self.reward_model.pick_best_response(
            question=question,
            position=position,
            opposing_position=opposing_position,
            turns=turns,
            started_first=started_first,
            possible_next_turns=next_turns,
        )

    @property
    def name(self) -> str:
        return f"BoNDebateAgent-{self.model_name}-best_of-{self.best_of}"


def run_debate(
    *,
    question: str,
    position: str,
    opposing_position: str,
    agent: BaseDebateAgent,
    opponent_agent: BaseDebateAgent,
    number_of_turns: int = 2,  # Number of turns (including opening statements so minimum is 2 and must be even)
):
    assert (
        number_of_turns >= 2 and number_of_turns % 2 == 0
    ), "Number of turns must be at least 2 and even"
    opening_statement = agent.create_opening_statement(
        question=question,
        position=position,
        opposing_position=opposing_position,
    )
    opponent_opening_statement = opponent_agent.create_opening_statement(
        question=question,
        position=opposing_position,
        opposing_position=position,
    )
    turns = [opening_statement, opponent_opening_statement]
    for _ in range((number_of_turns - 2) // 2):
        next_turn = agent.create_next_turn(
            question=question,
            position=position,
            opposing_position=opposing_position,
            turns=turns,
            started_first=True,
        )
        turns.append(next_turn)
        opponent_next_turn = opponent_agent.create_next_turn(
            question=question,
            position=opposing_position,
            opposing_position=position,
            turns=turns,
            started_first=False,
        )
        turns.append(opponent_next_turn)
    # Position always starts first
    return turns

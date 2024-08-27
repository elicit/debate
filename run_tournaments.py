import concurrent.futures
import itertools
import json
import os
import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm

from agents import BaseDebateAgent
from agents import BoNDebateAgent
from agents import DebateAgent
from agents import run_debate
from compute_elo import compute_elo_with_optimization
from llm.core import TextChat
from llm.core import TextUserMessage
from llm.openai import OpenAI
from models import MMLUProDebateSample

random.seed(314)

with open("binarised_mmlu_pro.json") as f:
    samples = [MMLUProDebateSample.model_validate(sample) for sample in json.load(f)]

TEMPLATE = """Question:

{question}

A) {first_option}
B) {second_option}

Below is a debate with one side arguing for A and the other side arguing for B.

{debate_so_far}

Read the debate carefully and then answer the question. Say "A" or "B" and NOTHING ELSE."""


class BaseJudge(ABC):

    @abstractmethod
    def judge(
        self,
        *,
        question: str,
        first_option: str,
        second_option: str,
        full_debate: list[str],
    ) -> Literal["A", "B"]:
        pass


@dataclass
class OpenAIJudge(BaseJudge):
    llm: OpenAI

    def judge(
        self,
        *,
        question: str,
        first_option: str,
        second_option: str,
        full_debate: list[str],
    ) -> Literal["A", "B"]:
        speakers = [
            "Person advocating for A: ",
            "Person advocating for B: ",
        ] * len(full_debate)
        debate_so_far = "".join(
            f"\n\n{speaker}\n{turn}" for speaker, turn in zip(speakers, full_debate)
        )
        judge_prompt = TEMPLATE.format(
            question=question,
            first_option=first_option,
            second_option=second_option,
            debate_so_far=debate_so_far,
        )
        response = self.llm.predict(
            TextChat(messages=[TextUserMessage(content=judge_prompt)]),
            max_tokens=1,
        )
        assert response in ("A", "B"), f"Invalid response: {response}"
        return "A" if response == "A" else "B"


def get_winner(
    question: str,
    first_option: str,
    second_option: str,
    agent_a: BaseDebateAgent,
    agent_b: BaseDebateAgent,
    judge: BaseJudge,
    number_of_turns: int = 6,
) -> Literal["A", "B"]:
    full_debate = run_debate(
        question=question,
        position=first_option,
        opposing_position=second_option,
        agent=agent_a,
        opponent_agent=agent_b,
        number_of_turns=number_of_turns,
    )
    return judge.judge(
        question=question,
        first_option=first_option,
        second_option=second_option,
        full_debate=full_debate,
    )


def make_pairings(
    agent_win_history: list[tuple[str, list[float]]],
) -> list[tuple[str, str]]:
    agents = [agent_name for agent_name, _ in agent_win_history]
    return list(itertools.combinations(agents, 2))


def agent_total_cost(agent: BaseDebateAgent) -> float:
    if isinstance(agent, DebateAgent):
        return agent.llm.total_cost()
    elif isinstance(agent, BoNDebateAgent):
        return agent.llm.total_cost() + agent.reward_model.llm.total_cost()
    else:
        return 0.0


def update_scores(
    question: MMLUProDebateSample,
    agent_a: BaseDebateAgent,
    agent_b: BaseDebateAgent,
    judge: BaseJudge,
    model_winrates: dict[tuple[str, str], list[float]],
    agent_win_history: list[tuple[str, list[float]]],
    pbar: tqdm,
):
    first_option = question.incorrect_answer
    second_option = question.correct_answer
    winner = get_winner(
        question=question.question,
        first_option=first_option,
        second_option=second_option,
        agent_a=agent_a,
        agent_b=agent_b,
        judge=judge,
    )
    winning_agent_name = agent_a.name if winner == "A" else agent_b.name
    losing_agent_name = agent_b.name if winner == "A" else agent_a.name
    model_winrates[(winning_agent_name, losing_agent_name)].append(1)
    model_winrates[(losing_agent_name, winning_agent_name)].append(0)
    for agent_name, wins in agent_win_history:
        if agent_name == winning_agent_name:
            wins.append(1)
        elif agent_name == losing_agent_name:
            wins.append(0)
    # Reverse the order of the agents
    winner = get_winner(
        question=question.question,
        first_option=first_option,
        second_option=second_option,
        agent_a=agent_b,
        agent_b=agent_a,
        judge=judge,
    )
    winning_agent_name = agent_b.name if winner == "A" else agent_a.name
    losing_agent_name = agent_a.name if winner == "A" else agent_b.name
    model_winrates[(winning_agent_name, losing_agent_name)].append(1)
    model_winrates[(losing_agent_name, winning_agent_name)].append(0)
    for agent_name, wins in agent_win_history:
        if agent_name == winning_agent_name:
            wins.append(1)
        elif agent_name == losing_agent_name:
            wins.append(0)

    pbar.update(1)


def run_tournament(
    questions: list[MMLUProDebateSample],
    agents: list[BaseDebateAgent],
    judge: BaseJudge,
    bootstrap_samples: int = 1000,
):
    assert len({agent.name for agent in agents}) == len(agents), "Duplicate agent names"
    agent_win_history = [(agent.name, list[float]()) for agent in agents]
    agents_by_name = {agent.name: agent for agent in agents}

    model_winrates = {
        (key[0], key[1]): list[float]()
        for key in itertools.product([agent.name for agent in agents], repeat=2)
    }

    print(f"Total debates: {len(questions) * len(agents) * (len(agents) - 1)}")

    pairings = make_pairings(agent_win_history)
    winrates: dict[tuple[str, str], float] = {}
    print("Starting...")
    for agent_a_name, agent_b_name in pairings:
        agent_a = agents_by_name[agent_a_name]
        agent_b = agents_by_name[agent_b_name]
        print("Pairing:", agent_a_name, agent_b_name)
        pbar = tqdm(total=len(questions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for sample_question in questions:
                executor.submit(
                    update_scores,
                    question=sample_question,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    judge=judge,
                    model_winrates=model_winrates,
                    agent_win_history=agent_win_history,
                    pbar=pbar,
                )
        pbar.close()

        winrates = {
            key: sum(value) / len(value) if value else 0.0
            for key, value in model_winrates.items()
        }

        # for (agent_a, agent_b), winrate in winrates.items():
        #     print(f"{agent_a} vs {agent_b}: {winrate}")

        total_cost = sum(
            [
                agent_total_cost(agents_by_name[agent_name])
                for agent_name, _ in agent_win_history
            ]
        )
        print(f"Total cost: {total_cost}")

    # Add self-play winr5ates
    for agent_name, _ in agent_win_history:
        model_winrates[(agent_name, agent_name)] = [0.5]

    def compute_resampled_winrate(value: list[float]) -> float:
        return sum(random.choices(value, k=len(value))) / len(value) if value else 0.0

    bootstrapped_elos: list[dict[str, float]] = []

    for _ in range(bootstrap_samples):
        sample_winrates = {
            key: compute_resampled_winrate(value)
            for key, value in model_winrates.items()
        }
        sample_elos = compute_elo_with_optimization(
            win_rates=sample_winrates,
        )
        bootstrapped_elos.append(sample_elos)

    ci = {
        agent_name: np.percentile(
            [sample_elos[agent_name] for sample_elos in bootstrapped_elos],
            [2.5, 97.5],
            axis=0,
        )
        for agent_name in agents_by_name
    }
    elo = compute_elo_with_optimization(
        win_rates=winrates,
    )
    return {
        agent_name: (
            float(elo[agent_name]),
            float(ci[agent_name][0]),
            float(ci[agent_name][1]),
        )
        for agent_name in agents_by_name
    }


openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
assert openai_api_key is not None, "The OPENAI_API_KEY environment variable is not set."
assert (
    anthropic_api_key is not None
), "The ANTHROPIC_API_KEY environment variable is not set."

AGENTS = [
    DebateAgent.from_model(
        model="gpt-3.5-turbo-0125",
        api_key=openai_api_key,
        temperature=0.0,
    ),
    BoNDebateAgent.from_model(
        model="gpt-3.5-turbo-0125",
        api_key=openai_api_key,
        temperature=0.8,
        best_of=4,
    ),
    DebateAgent.from_model(
        model="gpt-4o-mini-2024-07-18",
        api_key=openai_api_key,
        temperature=0.0,
    ),
    BoNDebateAgent.from_model(
        model="gpt-4o-mini-2024-07-18",
        api_key=openai_api_key,
        temperature=0.8,
        best_of=4,
    ),
    DebateAgent.from_model(
        model="gpt-4o-2024-05-13",
        api_key=openai_api_key,
        temperature=0.0,
    ),
    BoNDebateAgent.from_model(
        model="gpt-4o-2024-05-13",
        api_key=openai_api_key,
        temperature=0.8,
        best_of=4,
    ),
]

elo = run_tournament(
    questions=samples[500:650],
    agents=AGENTS,
    judge=OpenAIJudge(
        llm=OpenAI(
            model="ft:gpt-3.5-turbo-0125:elicit-experiments:mmlu-pro-judge-v2:9sLcMqLH",
            api_key=openai_api_key,
        )
    ),
)

for agent_name, (elo, ci_lower, ci_upper) in sorted(elo.items(), key=lambda x: x[1]):
    print(f"{agent_name}: {elo} ({ci_lower}, {ci_upper})")


# for (agent_a, agent_b), winrate in winrates.items():
#     print(f"{agent_a} vs {agent_b}: {winrate}")

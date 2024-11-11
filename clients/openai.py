from collections.abc import Mapping
from collections.abc import Sequence
from time import perf_counter
from typing import Literal

import requests
from pydantic import BaseModel
from pydantic import Field
from structlog.stdlib import get_logger

_logger = get_logger()

URL = "https://api.openai.com/v1/chat/completions"

OpenAIChatModelName = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "ft:gpt-4o-mini-2024-07-18:elicit-experiments:blind-judge-mmlu:9vBqSncN",
    "ft:gpt-3.5-turbo-0125:elicit-experiments:blind-judge-v2:9sLQFyaX",
    "ft:gpt-3.5-turbo-0125:elicit-experiments:mmlu-pro-judge-v2:9sLcMqLH",  # best of 4
    "ft:gpt-4o-mini-2024-07-18:elicit-experiments:n4-judge-v0:9vBlDhxM",  # best of 4
    "ft:gpt-3.5-turbo-0125:elicit-experiments:4o-judge:9xqN1aFs",  # best of 1
    "ft:gpt-4o-mini-2024-07-18:elicit-experiments:4o-judge:9w1bJdWK",  # best of 1
    "ft:gpt-3.5-turbo-0125:elicit-experiments:35-judge:9w1pG4Fr",  # best of 1
    "ft:gpt-4o-mini-2024-07-18:elicit-experiments:4o-mini-judge:9w1iRtNB",  # best of 1
    "ft:gpt-4o-2024-08-06:elicit-experiments:4o-judge:A0N2QXM6",  # best of 1
    "ft:gpt-4o-2024-08-06:elicit-experiments:blind-judge-v0:A0N167Qb",
]

OPENAI_MODEL_NAMES: tuple[OpenAIChatModelName, ...] = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
)


class OpenAITextMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Request(BaseModel):
    model: OpenAIChatModelName
    messages: Sequence[OpenAITextMessage]
    temperature: float = Field(..., ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = 1
    stop: list[str] | None = Field(default=None, max_length=4)
    max_tokens: int | None = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    response_format: Mapping[Literal["type"], Literal["json_object", "text"]] = {
        "type": "text"
    }


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int


class Choice(BaseModel):
    message: OpenAITextMessage
    finish_reason: str | None
    index: int


class Response(BaseModel):
    model: str
    choices: list[Choice]
    usage: Usage


def complete(
    request: Request,
    api_key: str,
    org_id: str | None,
    verbose: bool = False,
    read_timeout_s: float = 30,
    max_retries: int = 5,
) -> Response:
    start = perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"}
    if org_id:
        headers["OpenAI-Organization"] = org_id
    result = None
    for _ in range(max_retries):
        try:
            result = requests.post(
                URL,
                json=request.dict(),
                headers=headers,
                timeout=(15.0, read_timeout_s),
            )
            result.raise_for_status()
            break
        except requests.RequestException as e:
            if verbose:
                _logger.error("OpenAI chat completion failed", error=str(e))
            continue
    if result is None:
        raise RuntimeError(f"OpenAI API failed after {max_retries} retries")
    if verbose:
        _logger.info(
            "OpenAI chat completion",
            response_time=perf_counter() - start,
            model=request.model,
        )
    return Response.model_validate(result.json())

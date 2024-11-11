from collections.abc import Sequence
from time import perf_counter
from typing import Literal

import requests
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from structlog.stdlib import get_logger

_logger = get_logger()

URL = "https://api.anthropic.com/v1/messages"

AnthropicChatModelName = Literal[
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
]

ANTHROPIC_MODEL_NAMES: tuple[AnthropicChatModelName, ...] = (
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
)


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class Request(BaseModel):
    model: AnthropicChatModelName
    messages: Sequence[AnthropicMessage]
    system: str | None = None
    max_tokens: int
    metadata: dict[str, str] | None = None
    stop_sequences: list[str] | None = ["\n\nHuman:"]
    temperature: float = Field(default=1.0, ge=0, le=1)
    top_p: float = -1.0
    top_k: int = Field(default=-1, ge=-1)

    @field_validator("top_p")
    def validate_top_p(cls, top_p: float) -> float:
        if not (0 <= top_p <= 1) and top_p != -1:
            raise ValueError("top_p must be between 0 and 1")
        return top_p


class Content(BaseModel):
    text: str
    type: Literal["text"]


class Response(BaseModel):
    content: list[Content]
    usage: AnthropicUsage


def complete(
    request: Request,
    api_key: str,
    verbose: bool = False,
    read_timeout_s: float = 30,
    max_retries: int = 5,
) -> Response:
    start = perf_counter()
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    result = None
    for _ in range(max_retries):
        try:
            result = requests.post(
                URL,
                json=request.dict(exclude_none=True),
                headers=headers,
                timeout=(15.0, read_timeout_s),
            )
            result.raise_for_status()
            break
        except requests.RequestException as e:
            if verbose:
                _logger.error("Anthropic chat completion failed", error=str(e))
            continue
    if result is None:
        raise RuntimeError(f"Anthropic API failed after {max_retries} retries")
    if verbose:
        _logger.info(
            "Anthropic chat completion",
            response_time=perf_counter() - start,
            model=request.model,
        )
    return Response.model_validate(result.json())

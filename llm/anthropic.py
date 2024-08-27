from clients.anthropic import AnthropicChatModelName
from clients.anthropic import AnthropicMessage
from clients.anthropic import complete
from clients.anthropic import Request
from llm.core import BaseLLM
from llm.core import TextChat


class Anthropic(BaseLLM):
    model: AnthropicChatModelName
    api_key: str
    max_retries: int = 5
    timeout: float = 30
    verbose: bool = False
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def _predict(
        self,
        chat: TextChat,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> str:
        request = Request(
            model=self.model,
            system=chat.system_prompt,
            messages=[
                AnthropicMessage(role=message.role, content=message.content)
                for message in chat.messages
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = complete(
            request=request,
            api_key=self.api_key,
            max_retries=self.max_retries,
            read_timeout_s=self.timeout,
        )
        self.total_prompt_tokens += response.usage.input_tokens
        self.total_completion_tokens += response.usage.output_tokens
        return response.content[0].text

    def _sample(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        num_samples: int = 1,
    ) -> list[str]:
        raise NotImplementedError("Sampling is not supported for Anthropic models")

    def total_cost(self) -> float:
        if "claude-3-haiku" in self.model:
            return (
                0.25 * self.total_prompt_tokens / 1000**2
                + 1.25 * self.total_completion_tokens / 1000**2
            )
        elif "claude-3-sonnet" in self.model or "claude-3-5-sonnet" in self.model:
            return (
                3 * self.total_prompt_tokens / 1000**2
                + 15 * self.total_completion_tokens / 1000**2
            )
        elif "claude-3-opus" in self.model:
            return (
                15 * self.total_prompt_tokens / 1000**2
                + 75 * self.total_completion_tokens / 1000**2
            )
        else:
            raise ValueError(f"Cost tracking is not supported for model {self.model}")

    @property
    def model_name(self) -> str:
        return self.model

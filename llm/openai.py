from clients.openai import complete
from clients.openai import OpenAIChatModelName
from clients.openai import OpenAIReasoningMessage
from clients.openai import OpenAIReasoningModelName
from clients.openai import OpenAITextMessage
from clients.openai import ReasoningRequest
from clients.openai import Request
from llm.core import BaseLLM
from llm.core import TextChat


class OpenAI(BaseLLM):
    model: OpenAIChatModelName
    api_key: str
    org_id: str | None = None
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
            messages=(
                [OpenAITextMessage(role="system", content=chat.system_prompt)]
                if chat.system_prompt
                else []
            )
            + [
                OpenAITextMessage(role=message.role, content=message.content)
                for message in chat.messages
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = complete(
            request=request,
            api_key=self.api_key,
            org_id=self.org_id,
            max_retries=self.max_retries,
            read_timeout_s=self.timeout,
        )
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def _sample(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        num_samples: int = 1,
    ) -> list[str]:
        request = Request(
            model=self.model,
            messages=(
                [OpenAITextMessage(role="system", content=chat.system_prompt)]
                if chat.system_prompt
                else []
            )
            + [
                OpenAITextMessage(role=message.role, content=message.content)
                for message in chat.messages
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_samples,
        )
        response = complete(
            request=request,
            api_key=self.api_key,
            org_id=self.org_id,
            max_retries=self.max_retries,
            read_timeout_s=self.timeout,
        )
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return [choice.message.content for choice in response.choices]

    def total_cost(self) -> float:
        if "gpt-3.5-turbo" in self.model:
            return (
                0.5 * self.total_prompt_tokens / 1000**2
                + 1.5 * self.total_completion_tokens / 1000**2
            )
        elif "gpt-4o-mini" in self.model:
            return (
                0.15 * self.total_prompt_tokens / 1000**2
                + 0.6 * self.total_completion_tokens / 1000**2
            )
        elif "gpt-4o-2024-08-06" == self.model:
            return (
                2.5 * self.total_prompt_tokens / 1000**2
                + 10.0 * self.total_completion_tokens / 1000**2
            )
        elif "gpt-4o" in self.model:
            return (
                5 * self.total_prompt_tokens / 1000**2
                + 15 * self.total_completion_tokens / 1000**2
            )
        else:
            raise ValueError(f"Cost tracking is not supported for model {self.model}")

    @property
    def model_name(self) -> str:
        return self.model


class OpenAIReasoning(BaseLLM):
    model: OpenAIReasoningModelName
    api_key: str
    org_id: str | None = None
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
        assert (
            chat.system_prompt is None
        ), "System prompts are not supported for reasoning models"
        request = ReasoningRequest(
            model=self.model,
            messages=[
                OpenAIReasoningMessage(role=message.role, content=message.content)
                for message in chat.messages
            ],
        )
        response = complete(
            request=request,
            api_key=self.api_key,
            org_id=self.org_id,
            max_retries=self.max_retries,
            read_timeout_s=self.timeout,
        )
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def _sample(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        num_samples: int = 1,
    ) -> list[str]:
        raise NotImplementedError("Sampling is not supported for reasoning models")

    def total_cost(self) -> float:
        if "o1-preview" in self.model:
            return (
                15.0 * self.total_prompt_tokens / 1000**2
                + 60.0 * self.total_completion_tokens / 1000**2
            )
        elif "o1-mini" in self.model:
            return (
                3.00 * self.total_prompt_tokens / 1000**2
                + 12.00 * self.total_completion_tokens / 1000**2
            )
        else:
            raise ValueError(f"Cost tracking is not supported for model {self.model}")

    @property
    def model_name(self) -> str:
        return self.model

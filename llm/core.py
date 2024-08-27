import hashlib
import sqlite3
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel
from pydantic import field_validator


class TextUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class TextAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


TextMessage = TextUserMessage | TextAssistantMessage


class TextChat(BaseModel):
    messages: Sequence[TextMessage]
    system_prompt: str | None = None

    @field_validator("messages")
    def validate_messages(cls, messages: list[TextMessage]) -> list[TextMessage]:
        if len(messages) == 0:
            raise ValueError("At least one message must be provided")
        if messages[0].role != "user":
            raise ValueError("The first message must be from the user")
        if any(
            message.role == next_message.role
            for message, next_message in zip(messages, messages[1:])
        ):
            raise ValueError("Consecutive messages must have different roles")
        return messages


def _hash_chat(chat: TextChat) -> str:
    raw_chat = (chat.system_prompt or "") + "".join(
        f"{message.role}:{message.content}" for message in chat.messages
    )
    return hashlib.sha256(raw_chat.encode()).hexdigest()


class BaseLLM(BaseModel, ABC):

    @abstractmethod
    def _predict(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> str: ...

    @abstractmethod
    def _sample(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        num_samples: int = 1,
    ) -> list[str]: ...

    def model_post_init(self, __context):
        with sqlite3.connect("cache.db") as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )

    def predict(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        cache_id: int = 0,
    ) -> str:
        key = f"{self.model_name}-{cache_id}-{_hash_chat(chat)}-{max_tokens}"
        with sqlite3.connect("cache.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                cursor.close()
                return row[0]
            response = self._predict(chat, max_tokens, temperature)
            cursor.execute(
                "INSERT INTO cache (key, value) VALUES (?, ?)", (key, response)
            )
            conn.commit()
            cursor.close()
            return response

    def sample(
        self,
        chat: TextChat,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        num_samples: int = 1,
        cache_id: int = 0,
    ) -> list[str]:
        DELIMITER = "[NEW_MODEL_RESPONSE]"
        key = f"{self.model_name}-{cache_id}-{_hash_chat(chat)}-{max_tokens}-{num_samples}"
        with sqlite3.connect("cache.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                cursor.close()
                return row[0].split(DELIMITER)
            response = self._sample(chat, max_tokens, temperature, num_samples)
            cursor.execute(
                "INSERT INTO cache (key, value) VALUES (?, ?)",
                (key, DELIMITER.join(response)),
            )
            conn.commit()
            cursor.close()
            return response

    @abstractmethod
    def total_cost(self) -> float:
        """Return the total cost of the LLM in USD."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        ...

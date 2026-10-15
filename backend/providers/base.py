from abc import ABC, abstractmethod

from backend.schemas.llm import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        ...

    @abstractmethod
    async def execute(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Execute an LLM request."""
        ...

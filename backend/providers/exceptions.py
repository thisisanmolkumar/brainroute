class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class ProviderAuthenticationError(LLMProviderError):
    """Invalid or missing API credentials."""


class ProviderRateLimitError(LLMProviderError):
    """Provider rate limit exceeded."""


class ProviderTimeoutError(LLMProviderError):
    """Provider request timed out."""


class ProviderConfigurationError(LLMProviderError):
    """Invalid provider or model configuration."""

from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

# ---------------------------------------
# Types
# ---------------------------------------

ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]


# ---------------------------------------
# Pricing Configuration
# ---------------------------------------


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_1m_tokens: Decimal = Field(ge=0)

    cached_input_per_1m_tokens: Decimal | None = Field(
        default=None,
        ge=0,
    )

    output_per_1m_tokens: Decimal = Field(ge=0)


# ---------------------------------------
# Reasoning Configuration
# ---------------------------------------


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported_efforts: list[ReasoningEffort] = Field(min_length=1)

    default_effort: ReasoningEffort

    @model_validator(mode="after")
    def validate_efforts(self) -> Self:

        if len(self.supported_efforts) != len(set(self.supported_efforts)):
            raise ValueError("Duplicate reasoning efforts")

        if self.default_effort not in self.supported_efforts:
            raise ValueError("default_effort must be supported")

        return self


# ---------------------------------------
# Individual Model Configuration
# ---------------------------------------


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)

    tier: str | None = None

    enabled: bool = True

    reasoning: ReasoningConfig
    pricing: ModelPricing


# ---------------------------------------
# Global Defaults
# ---------------------------------------


class RegistryDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    api: str = "responses"

    reasoning_effort: ReasoningEffort = "medium"

    max_output_tokens: int = Field(default=4096, gt=0)


# ---------------------------------------
# Complete Registry Schema
# ---------------------------------------


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]

    currency: str = "USD"
    pricing_updated: str | None = None

    defaults: RegistryDefaults

    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:

        ids = [model.id for model in self.models]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate model IDs found")

        return self


# ---------------------------------------
# Model Registry
# ---------------------------------------


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None):

        if config_path is None:
            config_path = (
                Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
            )

        self.config_path = Path(config_path)

        self.config = self._load_config()

        self._models = {model.id: model for model in self.config.models}

    def _load_config(self) -> RegistryConfig:

        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Model configuration not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return RegistryConfig.model_validate(data)

    def list_models(
        self,
        enabled_only: bool = True,
    ) -> list[ModelConfig]:

        models = list(self._models.values())

        if enabled_only:
            models = [model for model in models if model.enabled]

        return models

    def get_model(self, model_id: str) -> ModelConfig:

        model = self._models.get(model_id)

        if model is None:
            raise ValueError(f"Unknown model: {model_id}")

        return model

    def get_supported_efforts(
        self,
        model_id: str,
    ) -> list[ReasoningEffort]:

        model = self.get_model(model_id)

        return model.reasoning.supported_efforts.copy()

    def validate_configuration(
        self,
        model_id: str,
        effort: str,
    ) -> ModelConfig:

        model = self.get_model(model_id)

        if not model.enabled:
            raise ValueError(f"Model '{model_id}' is disabled")

        if effort not in model.reasoning.supported_efforts:
            raise ValueError(
                f"Model '{model_id}' does not support reasoning effort '{effort}'"
            )

        return model

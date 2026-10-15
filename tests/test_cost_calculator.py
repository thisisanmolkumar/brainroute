from decimal import Decimal

import pytest

from backend.schemas.llm import LLMResponse, TokenUsage
from backend.services.cost_calculator import (
    CostCalculator,
    UnsupportedPricingError,
)
from backend.services.metrics import MetricsCollector


@pytest.fixture
def calculator():
    return CostCalculator()


def test_basic_cost(calculator):
    usage = TokenUsage(
        input_tokens=1000,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=500,
        reasoning_tokens=300,
    )

    cost = calculator.calculate(
        model_id="gpt-6-luna",
        usage=usage,
    )

    assert cost.input_cost_usd == Decimal("0.0001")
    assert cost.output_cost_usd == Decimal("0.00025")
    assert cost.total_cost_usd == Decimal("0.00035")


def test_cached_input_cost(calculator):
    usage = TokenUsage(
        input_tokens=1000,
        cached_input_tokens=200,
        cache_write_tokens=0,
        output_tokens=500,
    )

    cost = calculator.calculate(
        "gpt-6-luna",
        usage,
    )

    assert cost.ordinary_input_tokens == 800
    assert cost.cached_input_tokens == 200

    assert cost.total_cost_usd == Decimal("0.000332")


def test_cache_write_cost(calculator):
    usage = TokenUsage(
        input_tokens=1000,
        cached_input_tokens=200,
        cache_write_tokens=100,
        output_tokens=500,
    )

    cost = calculator.calculate(
        "gpt-6-luna",
        usage,
    )

    assert cost.ordinary_input_tokens == 700
    assert cost.cache_write_tokens == 100

    assert cost.input_cost_usd == Decimal("0.00007")
    assert cost.cached_input_cost_usd == Decimal("0.000002")
    assert cost.cache_write_cost_usd == Decimal("0.0000125")
    assert cost.output_cost_usd == Decimal("0.00025")

    assert cost.total_cost_usd == Decimal("0.0003345")


def test_reasoning_tokens_not_double_counted(calculator):
    usage_a = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        cached_input_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=100,
    )

    usage_b = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        cached_input_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=400,
    )

    cost_a = calculator.calculate("gpt-6-luna", usage_a)
    cost_b = calculator.calculate("gpt-6-luna", usage_b)

    assert cost_a.total_cost_usd == cost_b.total_cost_usd


def test_invalid_cached_tokens(calculator):
    usage = TokenUsage(
        input_tokens=100,
        cached_input_tokens=150,
        cache_write_tokens=0,
        output_tokens=50,
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        calculator.calculate("gpt-6-luna", usage)


def test_invalid_reasoning_tokens(calculator):
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        cached_input_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=100,
    )

    with pytest.raises(ValueError, match="Reasoning tokens"):
        calculator.calculate("gpt-6-luna", usage)


def test_missing_cache_details(calculator):
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
    )

    cost = calculator.calculate("gpt-6-luna", usage)

    assert cost.estimated is True
    assert len(cost.notes) == 2


def test_long_context_rejected(calculator):
    usage = TokenUsage(
        input_tokens=272001,
        output_tokens=100,
        cached_input_tokens=0,
        cache_write_tokens=0,
    )

    with pytest.raises(
        UnsupportedPricingError,
        match="Long-context",
    ):
        calculator.calculate("gpt-6-luna", usage)


def test_metrics_collection():
    response = LLMResponse(
        provider="openai",
        model_id="gpt-6-luna",
        effort="medium",
        status="completed",
        service_tier="default",
        output_text="Hello",
        latency_ms=1800.0,
        usage=TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cached_input_tokens=0,
            cache_write_tokens=0,
        ),
    )

    collector = MetricsCollector()

    metrics = collector.collect(
        response=response,
        end_to_end_latency_ms=1850.0,
    )

    assert metrics.provider_latency_ms == 1800.0
    assert metrics.end_to_end_latency_ms == 1850.0
    assert metrics.overhead_ms == 50.0

    assert metrics.cost is not None
    assert metrics.cost.total_cost_usd == Decimal("0.00035")


def test_incomplete_response_still_has_cost():
    response = LLMResponse(
        provider="openai",
        model_id="gpt-6-luna",
        effort="medium",
        status="incomplete",
        output_text="",
        latency_ms=1500,
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=200,
            cached_input_tokens=0,
            cache_write_tokens=0,
        ),
    )

    metrics = MetricsCollector().collect(
        response,
        end_to_end_latency_ms=1510,
    )

    assert metrics.status == "incomplete"
    assert metrics.cost is not None
    assert metrics.cost.total_cost_usd > 0


def test_missing_usage():
    response = LLMResponse(
        provider="openai",
        model_id="gpt-6-luna",
        effort="medium",
        output_text="",
        latency_ms=100,
        usage=None,
    )

    metrics = MetricsCollector().collect(
        response,
        end_to_end_latency_ms=120,
    )

    assert metrics.cost is None

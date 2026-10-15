import asyncio

from backend.providers.openai_provider import OpenAIProvider
from backend.schemas.llm import LLMRequest
from backend.services.execution_service import ExecutionService


async def main():

    provider = OpenAIProvider()

    service = ExecutionService(
        provider=provider,
    )

    request = LLMRequest(
        prompt=(
            "Reply with exactly this text: BrainRoute database integration successful."
        ),
        model_id="gpt-6-luna",
        effort="medium",
        max_output_tokens=1024,
        timeout_seconds=120,
    )

    try:
        result = await service.execute(
            request=request,
            category="smoke_test",
        )

        print("\n========== BRAINROUTE ==========")

        print("\nEXECUTION")
        print("Execution ID:", result.execution_id)
        print("Model:", result.response.model_id)
        print("Effort:", result.response.effort)
        print("Status:", result.response.status)

        print("\nRESPONSE")
        print(result.response.output_text)

        print("\nMETRICS")

        if result.metrics.usage:
            print(
                "Input tokens:",
                result.metrics.usage.input_tokens,
            )

            print(
                "Output tokens:",
                result.metrics.usage.output_tokens,
            )

            print(
                "Reasoning tokens:",
                result.metrics.usage.reasoning_tokens,
            )

        if result.metrics.cost:
            print(
                "Estimated cost:",
                result.metrics.cost.total_cost_usd,
                "USD",
            )
        else:
            print("Estimated cost: unavailable")

        print(
            "Provider latency:",
            result.metrics.provider_latency_ms,
            "ms",
        )

        print(
            "Execution latency:",
            result.metrics.end_to_end_latency_ms,
            "ms",
        )

        if result.metrics_warning:
            print(
                "Metrics warning:",
                result.metrics_warning,
            )

        print("\nSaved execution:", result.execution_id)

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

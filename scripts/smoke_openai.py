import asyncio
from time import perf_counter

from backend.providers.openai_provider import OpenAIProvider
from backend.schemas.llm import LLMRequest
from backend.services.metrics import MetricsCollector


async def main():
    provider = OpenAIProvider()
    collector = MetricsCollector()

    request = LLMRequest(
        prompt=(
            "Explain binary search in three sentences."
        ),
        model_id="gpt-6-luna",
        effort="medium",
        max_output_tokens=2048,
        timeout_seconds=120,
    )

    try:
        start = perf_counter()

        response = await provider.execute(request)

        end_to_end_latency_ms = (
            perf_counter() - start
        ) * 1000

        metrics = collector.collect(
            response=response,
            end_to_end_latency_ms=end_to_end_latency_ms,
        )

        print("\n========== BRAINROUTE ==========")

        print("\nEXECUTION")
        print("Model:", response.model_id)
        print("Effort:", response.effort)
        print("Status:", response.status)

        print("\nRESPONSE")
        print(response.output_text)

        print("\nTOKEN USAGE")

        if metrics.usage:
            print(
                "Input tokens:",
                metrics.usage.input_tokens,
            )

            print(
                "Cached tokens:",
                metrics.usage.cached_input_tokens,
            )

            print(
                "Cache-write tokens:",
                metrics.usage.cache_write_tokens,
            )

            print(
                "Output tokens:",
                metrics.usage.output_tokens,
            )

            print(
                "Reasoning tokens:",
                metrics.usage.reasoning_tokens,
            )

        print("\nCOST")

        if metrics.cost:
            cost = metrics.cost

            print(
                f"Input:       ${cost.input_cost_usd:.8f}"
            )
            print(
                f"Cached:      ${cost.cached_input_cost_usd:.8f}"
            )
            print(
                f"Cache write: ${cost.cache_write_cost_usd:.8f}"
            )
            print(
                f"Output:      ${cost.output_cost_usd:.8f}"
            )
            print(
                f"Total:       ${cost.total_cost_usd:.8f}"
            )

            for note in cost.notes:
                print("Note:", note)

        else:
            print("Cost unavailable")

        print("\nLATENCY")

        print(
            f"Provider:   {metrics.provider_latency_ms:.2f} ms"
        )
        print(
            f"End-to-end: {metrics.end_to_end_latency_ms:.2f} ms"
        )
        print(
            f"Overhead:   {metrics.overhead_ms:.2f} ms"
        )

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

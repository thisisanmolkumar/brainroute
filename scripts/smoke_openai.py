import asyncio

from backend.providers.openai_provider import OpenAIProvider
from backend.schemas.llm import LLMRequest


async def main():
    provider = OpenAIProvider()

    request = LLMRequest(
        prompt=("Reply with exactly this sentence: BrainRoute is online."),
        model_id="gpt-6-luna",
        effort="medium",
        max_output_tokens=2048,
        timeout_seconds=120,
    )

    try:
        response = await provider.execute(request)

        print("\n========== BRAINROUTE ==========\n")

        print("Provider:", response.provider)
        print("Model:", response.model_id)
        print("Effort:", response.effort)
        print("Status:", response.status)

        print("\nResponse:")
        print(response.output_text)

        print("\nExecution Metrics:")
        print(f"Latency: {response.latency_ms:.2f} ms")

        if response.usage:
            print("Input tokens:", response.usage.input_tokens)
            print("Output tokens:", response.usage.output_tokens)

            print(
                "Reasoning tokens:",
                response.usage.reasoning_tokens,
            )

            print(
                "Cached input tokens:",
                response.usage.cached_input_tokens,
            )

        print("\nResponse ID:", response.provider_response_id)

        if response.finish_reason:
            print("Finish reason:", response.finish_reason)

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
from time import perf_counter

from backend.core.model_registry import ModelRegistry
from backend.db.repositories import ExecutionRepository
from backend.providers.base import BaseLLMProvider
from backend.providers.exceptions import (
    ProviderConfigurationError,
)
from backend.schemas.execution import ExecutionResult
from backend.schemas.llm import LLMRequest
from backend.services.cost_calculator import (
    CostCalculator,
    UnsupportedPricingError,
)
from backend.services.metrics import (
    ExecutionMetrics,
    MetricsCollector,
)

logger = logging.getLogger(__name__)


class ExecutionService:
    def __init__(
        self,
        provider: BaseLLMProvider,
        registry: ModelRegistry | None = None,
        repository: ExecutionRepository | None = None,
        collector: MetricsCollector | None = None,
    ):
        self.provider = provider

        self.registry = registry or ModelRegistry()

        self.repository = repository or ExecutionRepository()

        self.collector = collector or MetricsCollector(CostCalculator(self.registry))

    async def execute(
        self,
        request: LLMRequest,
        category: str | None = None,
    ) -> ExecutionResult:

        start_time = perf_counter()

        # ----------------------------------
        # 1. Validate configuration
        # ----------------------------------

        try:
            model = self.registry.validate_configuration(
                model_id=request.model_id,
                effort=request.effort,
            )

        except ValueError as exc:
            raise ProviderConfigurationError(str(exc)) from exc

        if model.provider != self.provider.name:
            raise ProviderConfigurationError(
                "Model provider does not match the selected executor"
            )

        # ----------------------------------
        # 2. Preserve pricing information
        # ----------------------------------

        pricing_snapshot = {
            "currency": self.registry.config.currency,
            "pricing_updated": (self.registry.config.pricing_updated),
            "rates": model.pricing.model_dump(mode="json"),
        }

        # ----------------------------------
        # 3. Create pending execution
        # ----------------------------------

        # Our SQLAlchemy sessions are synchronous.
        # Run repository operations in a worker thread
        # instead of blocking the async event loop.

        execution_id = await asyncio.to_thread(
            self.repository.create_pending,
            request,
            self.provider.name,
            pricing_snapshot,
            category,
        )

        # ----------------------------------
        # 4. Execute the model
        # ----------------------------------

        try:
            response = await self.provider.execute(request)

        except Exception as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000

            try:
                await asyncio.to_thread(
                    self.repository.fail,
                    execution_id,
                    exc,
                    elapsed_ms,
                )

            except Exception:
                logger.exception(
                    "Could not save failure for execution %s",
                    execution_id,
                )

            raise

        # ----------------------------------
        # 5. Measure elapsed time
        # ----------------------------------

        elapsed_ms = (perf_counter() - start_time) * 1000

        # ----------------------------------
        # 6. Collect metrics
        # ----------------------------------

        metrics_warning = None

        try:
            metrics = self.collector.collect(
                response=response,
                end_to_end_latency_ms=elapsed_ms,
            )

        except (UnsupportedPricingError, ValueError) as exc:
            # A pricing problem should not erase a
            # successfully returned model response.

            metrics_warning = str(exc)

            metrics = ExecutionMetrics(
                provider=response.provider,
                model_id=response.model_id,
                effort=response.effort,
                status=response.status,
                provider_latency_ms=response.latency_ms,
                end_to_end_latency_ms=elapsed_ms,
                overhead_ms=max(
                    0.0,
                    elapsed_ms - response.latency_ms,
                ),
                usage=response.usage,
                cost=None,
            )

        # ----------------------------------
        # 7. Save results
        # ----------------------------------

        await asyncio.to_thread(
            self.repository.complete,
            execution_id,
            response,
            metrics,
            metrics_warning,
        )

        # ----------------------------------
        # 8. Return execution result
        # ----------------------------------

        return ExecutionResult(
            execution_id=execution_id,
            response=response,
            metrics=metrics,
            metrics_warning=metrics_warning,
        )

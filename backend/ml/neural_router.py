import torch
from torch import nn
from torch.nn import functional as F


class NeuralRouter(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_models: int,
        num_efforts: int,
    ) -> None:
        super().__init__()

        if (
            min(
                embedding_dim,
                num_models,
                num_efforts,
            )
            < 1
        ):
            raise ValueError("Invalid neural-router dimensions")

        self.task_projection = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.model_embedding = nn.Embedding(
            num_models,
            16,
        )

        self.effort_embedding = nn.Embedding(
            num_efforts,
            8,
        )

        # 64 semantic + 16 model + 8 effort
        # + 7 handcrafted numeric features.
        self.shared = nn.Sequential(
            nn.Linear(95, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.success_head = nn.Linear(64, 1)
        self.cost_head = nn.Linear(64, 1)
        self.latency_head = nn.Linear(64, 1)

    def forward(
        self,
        task_embedding: torch.Tensor,
        numeric_features: torch.Tensor,
        model_ids: torch.Tensor,
        effort_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        task_vector = self.task_projection(task_embedding)

        model_vector = self.model_embedding(model_ids)
        effort_vector = self.effort_embedding(effort_ids)

        combined = torch.cat(
            [
                task_vector,
                model_vector,
                effort_vector,
                numeric_features,
            ],
            dim=1,
        )

        hidden = self.shared(combined)

        return (
            self.success_head(hidden).squeeze(-1),
            self.cost_head(hidden).squeeze(-1),
            self.latency_head(hidden).squeeze(-1),
        )


def multitask_loss(
    outputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    success_rate: torch.Tensor,
    trials: torch.Tensor,
    cost_target: torch.Tensor,
    latency_target: torch.Tensor,
    cost_mask: torch.Tensor,
    latency_mask: torch.Tensor,
    cost_weight: float = 0.15,
    latency_weight: float = 0.15,
) -> dict[str, torch.Tensor]:

    success_logits, cost_prediction, latency_prediction = outputs

    # Weighted binomial cross-entropy.
    per_example_success = F.binary_cross_entropy_with_logits(
        success_logits,
        success_rate,
        reduction="none",
    )

    success_loss = (per_example_success * trials).sum() / trials.sum().clamp_min(1.0)

    def masked_regression(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        weights = trials * mask

        errors = (prediction - target).square()

        return (errors * weights).sum() / weights.sum().clamp_min(1.0)

    cost_loss = masked_regression(
        cost_prediction,
        cost_target,
        cost_mask,
    )

    latency_loss = masked_regression(
        latency_prediction,
        latency_target,
        latency_mask,
    )

    total = success_loss + cost_weight * cost_loss + latency_weight * latency_loss

    return {
        "total": total,
        "success": success_loss,
        "cost": cost_loss,
        "latency": latency_loss,
    }

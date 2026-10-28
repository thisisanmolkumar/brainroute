import numpy as np
from pydantic import BaseModel, ConfigDict

from backend.benchmarks.ml_dataset import (
    MLExample,
    MLFeatures,
)

NUMERIC_FIELDS = (
    "character_count",
    "word_count",
    "line_count",
    "digit_count",
    "question_mark_count",
    "operator_count",
    "has_code_block",
)


class FeatureEncoder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1

    models: list[str]
    efforts: list[str]

    @classmethod
    def fit(
        cls,
        examples: list[MLExample],
    ) -> "FeatureEncoder":

        if not examples:
            raise ValueError("Cannot fit encoder on empty dataset")

        return cls(
            models=sorted({example.features.model_id for example in examples}),
            efforts=sorted({example.features.effort for example in examples}),
        )

    @property
    def feature_names(self) -> list[str]:
        numeric = list(NUMERIC_FIELDS)

        model_columns = [f"model_{index}" for index in range(len(self.models))]

        effort_columns = [f"effort_{index}" for index in range(len(self.efforts))]

        return numeric + model_columns + effort_columns

    def transform(
        self,
        features: list[MLFeatures],
    ) -> np.ndarray:

        matrix = np.zeros(
            (
                len(features),
                len(self.feature_names),
            ),
            dtype=np.float32,
        )

        model_lookup = {model: index for index, model in enumerate(self.models)}

        effort_lookup = {effort: index for index, effort in enumerate(self.efforts)}

        model_offset = len(NUMERIC_FIELDS)

        effort_offset = model_offset + len(self.models)

        for row_index, item in enumerate(features):
            if item.model_id not in model_lookup:
                raise ValueError(
                    f"Unseen model: {item.model_id}. "
                    "This baseline cannot predict "
                    "untrained model configurations."
                )

            if item.effort not in effort_lookup:
                raise ValueError(f"Unseen effort: {item.effort}")

            # Numerical features
            for column_index, field in enumerate(NUMERIC_FIELDS):
                matrix[row_index, column_index] = float(getattr(item, field))

            # Model one-hot encoding
            model_column = model_offset + model_lookup[item.model_id]

            matrix[row_index, model_column] = 1.0

            # Effort one-hot encoding
            effort_column = effort_offset + effort_lookup[item.effort]

            matrix[row_index, effort_column] = 1.0

        return matrix

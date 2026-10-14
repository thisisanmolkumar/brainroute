from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from backend.core.model_registry import ModelRegistry


@pytest.fixture
def registry():
    return ModelRegistry()


@pytest.fixture
def registry_data(registry):
    with registry.config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_config(tmp_path, data):
    path = tmp_path / "models.yaml"

    path.write_text(
        yaml.safe_dump(data),
        encoding="utf-8",
    )

    return path


def test_load_registry(registry):
    assert len(registry.list_models()) > 0


def test_get_model(registry):
    model = registry.list_models()[0]

    result = registry.get_model(model.id)

    assert result.id == model.id


def test_supported_efforts(registry):
    model = registry.list_models()[0]

    efforts = registry.get_supported_efforts(model.id)

    assert efforts == model.reasoning.supported_efforts


def test_valid_configuration(registry):
    model = registry.list_models()[0]

    result = registry.validate_configuration(
        model.id,
        model.reasoning.default_effort,
    )

    assert result.id == model.id


def test_invalid_model(registry):
    with pytest.raises(ValueError, match="Unknown model"):
        registry.get_model("nonexistent-model")


def test_invalid_effort(registry):
    model = registry.list_models()[0]

    with pytest.raises(ValueError, match="does not support"):
        registry.validate_configuration(
            model.id,
            "ultra",
        )


def test_disabled_model(tmp_path, registry_data):
    data = deepcopy(registry_data)

    model_id = data["models"][0]["id"]
    data["models"][0]["enabled"] = False

    registry = ModelRegistry(write_config(tmp_path, data))

    assert model_id not in [model.id for model in registry.list_models()]

    with pytest.raises(ValueError, match="disabled"):
        registry.validate_configuration(
            model_id,
            data["models"][0]["reasoning"]["default_effort"],
        )


def test_duplicate_models(tmp_path, registry_data):
    data = deepcopy(registry_data)

    data["models"].append(deepcopy(data["models"][0]))

    with pytest.raises(ValidationError, match="Duplicate model IDs"):
        ModelRegistry(write_config(tmp_path, data))


def test_invalid_default_effort(tmp_path, registry_data):
    data = deepcopy(registry_data)

    reasoning = data["models"][0]["reasoning"]
    reasoning["default_effort"] = "none"
    reasoning["supported_efforts"] = ["medium", "high"]

    with pytest.raises(ValidationError, match="default_effort"):
        ModelRegistry(write_config(tmp_path, data))

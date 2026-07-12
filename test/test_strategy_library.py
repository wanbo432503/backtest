from __future__ import annotations

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

from strategy_engine import (
    StrategyDecision,
    StrategyDefinition,
    StrategyParamMeta,
)
from strategy_library import StrategyLibrary


class FakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: int = 5
    position_pct: float = 0.95


class InvalidNumberGridConfig(BaseModel):
    threshold: float = 65


def prepare_fake_frame(data: pd.DataFrame, config: BaseModel) -> pd.DataFrame:
    return data.copy()


def evaluate_fake(context) -> StrategyDecision:
    return StrategyDecision()


FAKE_DEFINITION = StrategyDefinition(
    strategy_id="fake",
    display_name="Fake strategy",
    description="A fake strategy for registry tests.",
    config_model=FakeConfig,
    parameters=(
        StrategyParamMeta(
            name="period",
            label="Period",
            type="int",
            default=5,
            search_values=[5, 8],
            min_value=2,
            max_value=20,
            step=1,
        ),
        StrategyParamMeta(
            name="position_pct",
            label="Position",
            type="float",
            default=0.95,
            search_values=[0.5, 0.95],
            min_value=0.05,
            max_value=1,
            step=0.05,
        ),
    ),
    prepare_frame=prepare_fake_frame,
    evaluate=evaluate_fake,
    min_history_bars=lambda config: int(config.period) + 1,
)


def test_strategy_library_validates_and_completes_parameters():
    library = StrategyLibrary([FAKE_DEFINITION])

    config = library.validate_config("fake", {"period": 8})

    assert config.period == 8
    assert config.position_pct == 0.95
    assert library.get("fake") is FAKE_DEFINITION


def test_strategy_library_serializes_dual_mode_catalog():
    library = StrategyLibrary([FAKE_DEFINITION])

    catalog = library.to_catalog()

    assert catalog == [
        {
            "name": "fake",
            "display_name": "Fake strategy",
            "description": "A fake strategy for registry tests.",
            "class_name": "evaluate_fake",
            "engine": "unified",
            "supported_modes": ["single_stock", "signal_portfolio"],
            "parameters": [
                parameter.model_dump(mode="json")
                for parameter in FAKE_DEFINITION.parameters
            ],
        }
    ]


def test_strategy_library_rejects_duplicate_strategy_ids():
    with pytest.raises(ValueError, match="duplicate strategy id: fake"):
        StrategyLibrary([FAKE_DEFINITION, FAKE_DEFINITION])


def test_strategy_library_rejects_unknown_strategy_and_parameter():
    library = StrategyLibrary([FAKE_DEFINITION])

    with pytest.raises(ValueError, match="strategy 'missing' does not exist"):
        library.validate_config("missing", {})
    with pytest.raises(ValueError, match="extra_forbidden"):
        library.validate_config("fake", {"unknown": 1})


def test_strategy_library_rejects_metadata_that_differs_from_config_model():
    invalid = StrategyDefinition(
        strategy_id="invalid",
        display_name="Invalid",
        description="Invalid metadata.",
        config_model=FakeConfig,
        parameters=FAKE_DEFINITION.parameters[:1],
        prepare_frame=prepare_fake_frame,
        evaluate=evaluate_fake,
        min_history_bars=lambda config: 1,
    )

    with pytest.raises(ValueError, match="parameter metadata does not match config fields"):
        StrategyLibrary([invalid])


def test_strategy_library_rejects_number_step_grid_that_excludes_integer_default():
    invalid = StrategyDefinition(
        strategy_id="invalid_number_grid",
        display_name="Invalid number grid",
        description="The HTML number input cannot accept its own default.",
        config_model=InvalidNumberGridConfig,
        parameters=(
            StrategyParamMeta(
                name="threshold",
                label="Threshold",
                type="float",
                default=65,
                min_value=0.1,
                max_value=100,
                step=1,
            ),
        ),
        prepare_frame=prepare_fake_frame,
        evaluate=evaluate_fake,
        min_history_bars=lambda config: 1,
    )

    with pytest.raises(ValueError, match="number input step grid"):
        StrategyLibrary([invalid])


def test_real_strategy_float_parameters_accept_defaults_and_whole_numbers():
    from strategy_library import get_strategy_library

    for definition in get_strategy_library().list():
        for parameter in definition.parameters:
            if parameter.type != "float" or parameter.step is None:
                continue
            base = parameter.min_value or 0
            default_steps = (float(parameter.default) - base) / parameter.step
            assert default_steps == pytest.approx(round(default_steps)), (
                definition.strategy_id,
                parameter.name,
            )
            first_integer = max(0, int(base) + (0 if float(base).is_integer() else 1))
            if parameter.max_value is not None and first_integer > parameter.max_value:
                continue
            integer_steps = (first_integer - base) / parameter.step
            assert integer_steps == pytest.approx(round(integer_steps)), (
                definition.strategy_id,
                parameter.name,
            )

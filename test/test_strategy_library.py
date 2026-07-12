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

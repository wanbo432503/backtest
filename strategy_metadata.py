from pydantic import BaseModel, Field

from strategy_engine import ParamType, ParamValue, StrategyParamMeta
from strategy_library import get_strategy_library


class StrategyMeta(BaseModel):
    name: str
    label: str
    parameters: list[StrategyParamMeta] = Field(default_factory=list)


def get_strategy_metadata(strategy_name: str) -> StrategyMeta:
    try:
        definition = get_strategy_library().get(strategy_name)
    except ValueError:
        return StrategyMeta(name=strategy_name, label=strategy_name)
    return StrategyMeta(
        name=definition.strategy_id,
        label=definition.display_name,
        parameters=list(definition.parameters),
    )


def get_strategy_parameters(strategy_name: str) -> list[dict]:
    return [
        parameter.model_dump(mode="json")
        for parameter in get_strategy_metadata(strategy_name).parameters
    ]

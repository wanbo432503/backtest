from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ValidationError

from strategy_engine import StrategyDefinition


class StrategyLibrary:
    def __init__(self, definitions: Iterable[StrategyDefinition] = ()) -> None:
        self._definitions: dict[str, StrategyDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: StrategyDefinition) -> None:
        strategy_id = definition.strategy_id.strip()
        if not strategy_id:
            raise ValueError("strategy id must not be empty")
        if strategy_id in self._definitions:
            raise ValueError(f"duplicate strategy id: {strategy_id}")
        self._validate_metadata(definition)
        self._definitions[strategy_id] = definition

    def get(self, strategy_id: str) -> StrategyDefinition:
        definition = self._definitions.get(strategy_id)
        if definition is None:
            raise ValueError(f"strategy '{strategy_id}' does not exist")
        return definition

    def list(self) -> list[StrategyDefinition]:
        return list(self._definitions.values())

    def validate_config(
        self,
        strategy_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> BaseModel:
        definition = self.get(strategy_id)
        try:
            return definition.config_model.model_validate(parameters or {})
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    def to_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.strategy_id,
                "display_name": definition.display_name,
                "description": definition.description,
                "class_name": definition.evaluate.__name__,
                "engine": "unified",
                "supported_modes": list(definition.supported_modes),
                "parameters": [
                    parameter.model_dump(mode="json")
                    for parameter in definition.parameters
                ],
            }
            for definition in self.list()
        ]

    @staticmethod
    def _validate_metadata(definition: StrategyDefinition) -> None:
        config_fields = set(definition.config_model.model_fields)
        metadata_fields = {parameter.name for parameter in definition.parameters}
        if config_fields != metadata_fields:
            raise ValueError(
                "parameter metadata does not match config fields for "
                f"strategy '{definition.strategy_id}'"
            )
        if len(metadata_fields) != len(definition.parameters):
            raise ValueError(
                f"duplicate parameter metadata for strategy '{definition.strategy_id}'"
            )
        for parameter in definition.parameters:
            field_info = definition.config_model.model_fields[parameter.name]
            if field_info.default != parameter.default:
                raise ValueError(
                    "parameter metadata default does not match config default for "
                    f"strategy '{definition.strategy_id}': {parameter.name}"
                )

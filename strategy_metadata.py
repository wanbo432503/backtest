from typing import Literal

from pydantic import BaseModel, Field


ParamType = Literal["int", "float", "str", "bool"]
ParamValue = int | float | str | bool


class StrategyParamMeta(BaseModel):
    name: str
    label: str
    type: ParamType
    default: ParamValue
    search_values: list[ParamValue] = Field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    description: str = ""


class StrategyMeta(BaseModel):
    name: str
    label: str
    parameters: list[StrategyParamMeta] = Field(default_factory=list)


STRATEGY_METADATA: dict[str, StrategyMeta] = {
    "sma_cross": StrategyMeta(
        name="sma_cross",
        label="双均线交叉策略",
        parameters=[
            StrategyParamMeta(
                name="n1",
                label="短期均线",
                type="int",
                default=10,
                search_values=[5, 10, 15, 20],
                min_value=2,
                max_value=60,
                step=1,
                description="短周期均线窗口，数值越小越敏感。",
            ),
            StrategyParamMeta(
                name="n2",
                label="长期均线",
                type="int",
                default=30,
                search_values=[20, 30, 45, 60],
                min_value=5,
                max_value=180,
                step=1,
                description="长周期均线窗口，应大于短期均线。",
            ),
        ],
    ),
    "rsi": StrategyMeta(
        name="rsi",
        label="RSI策略",
        parameters=[
            StrategyParamMeta(
                name="rsi_period",
                label="RSI周期",
                type="int",
                default=14,
                search_values=[6, 14, 21],
                min_value=2,
                max_value=60,
                step=1,
            ),
            StrategyParamMeta(
                name="rsi_lower",
                label="买入阈值",
                type="int",
                default=30,
                search_values=[25, 30, 35],
                min_value=1,
                max_value=50,
                step=1,
            ),
            StrategyParamMeta(
                name="rsi_upper",
                label="卖出阈值",
                type="int",
                default=70,
                search_values=[60, 70, 80],
                min_value=50,
                max_value=99,
                step=1,
            ),
        ],
    ),
    "rsi_risk_control": StrategyMeta(
        name="rsi_risk_control",
        label="RSI风控策略",
        parameters=[
            StrategyParamMeta(
                name="rsi_period",
                label="RSI周期",
                type="int",
                default=14,
                search_values=[6, 14, 21],
                min_value=2,
                max_value=60,
                step=1,
            ),
            StrategyParamMeta(
                name="rsi_buy",
                label="买入阈值",
                type="int",
                default=30,
                search_values=[25, 30, 35],
                min_value=1,
                max_value=50,
                step=1,
            ),
            StrategyParamMeta(
                name="rsi_sell",
                label="卖出阈值",
                type="int",
                default=70,
                search_values=[60, 70, 80],
                min_value=50,
                max_value=99,
                step=1,
            ),
            StrategyParamMeta(
                name="trend_ma",
                label="趋势均线",
                type="int",
                default=60,
                search_values=[30, 60, 120],
                min_value=5,
                max_value=250,
                step=1,
            ),
            StrategyParamMeta(
                name="stop_loss_pct",
                label="止损比例",
                type="float",
                default=5,
                search_values=[3, 5, 8],
                min_value=0,
                max_value=30,
                step=0.5,
            ),
            StrategyParamMeta(
                name="take_profit_pct",
                label="止盈比例",
                type="float",
                default=12,
                search_values=[8, 12, 20],
                min_value=0,
                max_value=80,
                step=0.5,
            ),
            StrategyParamMeta(
                name="max_holding_bars",
                label="最大持仓周期",
                type="int",
                default=120,
                search_values=[40, 80, 120],
                min_value=1,
                max_value=500,
                step=1,
            ),
            StrategyParamMeta(
                name="position_pct",
                label="仓位比例",
                type="float",
                default=0.95,
                search_values=[0.5, 0.8, 0.95],
                min_value=0.05,
                max_value=1,
                step=0.05,
            ),
            StrategyParamMeta(
                name="cooldown_bars",
                label="冷却周期",
                type="int",
                default=3,
                search_values=[0, 3, 5],
                min_value=0,
                max_value=60,
                step=1,
            ),
        ],
    ),
}


def get_strategy_metadata(strategy_name: str) -> StrategyMeta:
    return STRATEGY_METADATA.get(
        strategy_name,
        StrategyMeta(name=strategy_name, label=strategy_name, parameters=[]),
    )


def get_strategy_parameters(strategy_name: str) -> list[dict]:
    return [
        param.model_dump()
        for param in get_strategy_metadata(strategy_name).parameters
    ]

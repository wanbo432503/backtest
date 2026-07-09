from __future__ import annotations

from portfolio_selection_strategy_models import (
    PortfolioSelectionStrategyDefinition,
    StrategyFactorSpec,
)
from portfolio_factor_optimization_models import FactorSearchSpace, SelectionStrategySearchSpace


def list_selection_strategies() -> list[PortfolioSelectionStrategyDefinition]:
    return list(_STRATEGIES)


def get_selection_strategy(strategy_id: str) -> PortfolioSelectionStrategyDefinition:
    normalized_id = strategy_id.strip()
    for strategy in _STRATEGIES:
        if strategy.strategy_id == normalized_id:
            return strategy
    raise ValueError(f"unknown portfolio selection strategy: {strategy_id}")


def build_factor_search_space_for_strategy(strategy_id: str) -> SelectionStrategySearchSpace:
    strategy = get_selection_strategy(strategy_id)
    factor_lookbacks: dict[str, list[int]] = {}
    factor_weights: dict[str, list[float]] = {}

    for factor in strategy.factors:
        if factor.lookback_candidates:
            factor_lookbacks[factor.key] = list(factor.lookback_candidates)
        elif factor.default_lookback is not None:
            factor_lookbacks[factor.key] = [factor.default_lookback]

        factor_weights[factor.key] = (
            list(factor.weight_candidates)
            if factor.weight_candidates
            else [factor.default_weight]
        )

    return SelectionStrategySearchSpace(
        strategy_id=strategy.strategy_id,
        factor_lookbacks=factor_lookbacks,
        factor_weights=factor_weights,
        top_n=list(strategy.top_n_candidates),
        score_threshold=list(strategy.score_threshold_candidates),
        legacy_factor_search_space=_build_legacy_factor_search_space(
            strategy,
            factor_lookbacks=factor_lookbacks,
            factor_weights=factor_weights,
        ),
    )


def _factor(
    key: str,
    label: str,
    direction: str,
    default_weight: float,
    *,
    default_lookback: int | None = None,
    lookback_candidates: list[int] | None = None,
    weight_candidates: list[float] | None = None,
    required: bool = True,
) -> StrategyFactorSpec:
    return StrategyFactorSpec(
        key=key,
        label=label,
        direction=direction,
        default_weight=default_weight,
        default_lookback=default_lookback,
        lookback_candidates=lookback_candidates or [],
        weight_candidates=weight_candidates or [],
        required=required,
    )


def _build_legacy_factor_search_space(
    strategy: PortfolioSelectionStrategyDefinition,
    *,
    factor_lookbacks: dict[str, list[int]],
    factor_weights: dict[str, list[float]],
) -> FactorSearchSpace:
    default_space = FactorSearchSpace()

    momentum_lookback = factor_lookbacks.get("momentum_return", default_space.momentum_lookback)
    volatility_lookback = (
        factor_lookbacks.get("realized_volatility")
        or factor_lookbacks.get("downside_volatility")
        or factor_lookbacks.get("max_drawdown_window")
        or default_space.volatility_lookback
    )
    liquidity_lookback = factor_lookbacks.get("liquidity_turnover", default_space.liquidity_lookback)

    return FactorSearchSpace(
        momentum_lookback=momentum_lookback,
        volatility_lookback=volatility_lookback,
        liquidity_lookback=liquidity_lookback,
        momentum_weight=factor_weights.get("momentum_return", default_space.momentum_weight),
        volatility_weight=_legacy_volatility_weights(
            strategy,
            factor_weights.get("realized_volatility")
            or factor_weights.get("downside_volatility")
            or factor_weights.get("max_drawdown_window")
            or default_space.volatility_weight,
        ),
        liquidity_weight=factor_weights.get("liquidity_turnover", default_space.liquidity_weight),
        trend_weight=factor_weights.get("ma_trend", default_space.trend_weight),
        top_n=list(strategy.top_n_candidates),
        score_threshold=list(strategy.score_threshold_candidates),
    )


def _legacy_volatility_weights(
    strategy: PortfolioSelectionStrategyDefinition,
    weights: list[float],
) -> list[float]:
    lower_better_factor_keys = {
        factor.key
        for factor in strategy.factors
        if factor.direction == "lower_better"
    }
    if lower_better_factor_keys.intersection(
        {"realized_volatility", "downside_volatility", "max_drawdown_window"}
    ):
        return sorted({-abs(float(weight)) for weight in weights})
    return list(weights)


_STRATEGIES: tuple[PortfolioSelectionStrategyDefinition, ...] = (
    PortfolioSelectionStrategyDefinition(
        strategy_id="steady_low_vol_momentum",
        name="稳健低波动动量策略",
        description="选择中期动量向上、波动和下行波动更温和、流动性足够的股票。",
        suitable_for="偏耐心、希望收益曲线稳步抬升且不追逐剧烈波动的组合轮动。",
        caveats=[
            "强牛市中可能跑输高弹性突破策略。",
            "低波动不代表无回撤，仍需观察验证集回撤和换手。",
        ],
        default_rebalance_frequency="monthly",
        default_top_n=5,
        top_n_candidates=[2, 3, 5, 10],
        factors=[
            _factor(
                "momentum_return",
                "中期动量收益",
                "higher_better",
                0.35,
                default_lookback=60,
                lookback_candidates=[40, 60, 90, 120],
                weight_candidates=[0.2, 0.35, 0.5],
            ),
            _factor(
                "realized_volatility",
                "实现波动率",
                "lower_better",
                0.25,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.15, 0.25, 0.35],
            ),
            _factor(
                "downside_volatility",
                "下行波动率",
                "lower_better",
                0.20,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.1, 0.2, 0.3],
            ),
            _factor(
                "liquidity_turnover",
                "流动性成交额",
                "higher_better",
                0.10,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.05, 0.1, 0.2],
            ),
            _factor(
                "ma_trend",
                "均线趋势确认",
                "higher_better",
                0.10,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.05, 0.1, 0.2],
            ),
        ],
    ),
    PortfolioSelectionStrategyDefinition(
        strategy_id="strong_trend_breakout",
        name="强趋势突破策略",
        description="选择突破近期价格区间、成交量同步放大、趋势确认较强的股票。",
        suitable_for="愿意承受更高波动，捕捉趋势启动和阶段性强势行情的组合轮动。",
        caveats=[
            "突破失败时回撤可能更快，需要关注验证集回撤和高波动风险标记。",
            "成交量确认不足时不应单独依赖价格突破。",
        ],
        default_rebalance_frequency="weekly",
        default_top_n=5,
        top_n_candidates=[2, 3, 5, 10],
        factors=[
            _factor(
                "breakout_strength",
                "突破强度",
                "higher_better",
                0.35,
                default_lookback=60,
                lookback_candidates=[40, 60, 90, 120],
                weight_candidates=[0.2, 0.35, 0.5],
            ),
            _factor(
                "momentum_return",
                "中期动量收益",
                "higher_better",
                0.25,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.15, 0.25, 0.4],
            ),
            _factor(
                "volume_expansion",
                "成交量放大",
                "higher_better",
                0.20,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.1, 0.2, 0.3],
            ),
            _factor(
                "ma_trend",
                "均线趋势确认",
                "higher_better",
                0.15,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.05, 0.15, 0.25],
            ),
            _factor(
                "realized_volatility",
                "实现波动率",
                "lower_better",
                0.05,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.0, 0.05, 0.15],
            ),
        ],
    ),
    PortfolioSelectionStrategyDefinition(
        strategy_id="high_liquidity_trend",
        name="高流动性趋势策略",
        description="优先选择成交额和成交稳定性较好的趋势股票，降低流动性和执行层面的不确定性。",
        suitable_for="希望候选股票更容易跟踪、虚拟盘或手动交易更可执行的组合。",
        caveats=[
            "过度强调流动性可能错过小市值高弹性机会。",
            "仍需结合滑点、涨跌停和实际成交约束观察结果。",
        ],
        default_rebalance_frequency="monthly",
        default_top_n=5,
        top_n_candidates=[2, 3, 5, 10, 20],
        factors=[
            _factor(
                "liquidity_turnover",
                "流动性成交额",
                "higher_better",
                0.35,
                default_lookback=20,
                lookback_candidates=[10, 20, 40, 60],
                weight_candidates=[0.2, 0.35, 0.5],
            ),
            _factor(
                "volume_stability",
                "成交稳定性",
                "higher_better",
                0.20,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.1, 0.2, 0.3],
            ),
            _factor(
                "ma_trend",
                "均线趋势确认",
                "higher_better",
                0.20,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.1, 0.2, 0.3],
            ),
            _factor(
                "momentum_return",
                "中期动量收益",
                "higher_better",
                0.15,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.05, 0.15, 0.25],
            ),
            _factor(
                "realized_volatility",
                "实现波动率",
                "lower_better",
                0.10,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.0, 0.1, 0.2],
            ),
        ],
    ),
    PortfolioSelectionStrategyDefinition(
        strategy_id="drawdown_control_rotation",
        name="回撤控制型轮动策略",
        description="选择仍有动量但近期回撤、下行波动更可控，并具备一定修复能力的股票。",
        suitable_for="重视验证集最大回撤和收益曲线平滑度，宁愿慢一点也不要剧烈波动的组合。",
        caveats=[
            "强趋势初期可能因为刚经历波动而被过滤。",
            "应重点观察验证集收益曲线、回撤、换手和过拟合风险。",
        ],
        default_rebalance_frequency="monthly",
        default_top_n=5,
        top_n_candidates=[2, 3, 5, 10],
        factors=[
            _factor(
                "momentum_return",
                "中期动量收益",
                "higher_better",
                0.25,
                default_lookback=60,
                lookback_candidates=[40, 60, 90, 120],
                weight_candidates=[0.1, 0.25, 0.4],
            ),
            _factor(
                "max_drawdown_window",
                "近期最大回撤",
                "lower_better",
                0.30,
                default_lookback=60,
                lookback_candidates=[40, 60, 90, 120],
                weight_candidates=[0.15, 0.3, 0.45],
            ),
            _factor(
                "downside_volatility",
                "下行波动率",
                "lower_better",
                0.20,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.1, 0.2, 0.3],
            ),
            _factor(
                "recovery_strength",
                "回撤修复强度",
                "higher_better",
                0.15,
                default_lookback=60,
                lookback_candidates=[40, 60, 90],
                weight_candidates=[0.05, 0.15, 0.25],
            ),
            _factor(
                "liquidity_turnover",
                "流动性成交额",
                "higher_better",
                0.10,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.05, 0.1, 0.2],
            ),
        ],
    ),
    PortfolioSelectionStrategyDefinition(
        strategy_id="a_share_full_financial_multifactor",
        name="A股完整财务多因子策略",
        description="选择估值相对便宜、盈利质量和现金流更稳、分红更可持续、波动不过热的 A 股组合。",
        suitable_for="能提供 AkShare 财务/估值/分红数据，希望月度低频持有一篮子质量价值股的组合回测。",
        caveats=[
            "财务数据按公告日或可得日期做快照，接口缺失时该股票的财务因子会降级。",
            "当前组合回测 Top N 上限为 20，完整策略的 50 股版本需要后续扩大持仓约束。",
            "低估值和低波动可能在强题材行情中阶段性跑输。",
        ],
        default_rebalance_frequency="monthly",
        default_top_n=5,
        top_n_candidates=[5, 10, 20],
        factors=[
            _factor(
                "pe_inverse",
                "低PE估值",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "pb_inverse",
                "低PB估值",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "ps_inverse",
                "低PS估值",
                "higher_better",
                0.03,
                weight_candidates=[0.01, 0.03, 0.05],
            ),
            _factor(
                "pcf_inverse",
                "低市现率",
                "higher_better",
                0.03,
                weight_candidates=[0.01, 0.03, 0.05],
            ),
            _factor(
                "fcf_yield",
                "自由现金流收益率",
                "higher_better",
                0.05,
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "roe",
                "ROE",
                "higher_better",
                0.07,
                weight_candidates=[0.04, 0.07, 0.10],
            ),
            _factor(
                "roa",
                "ROA",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "gross_margin",
                "毛利率",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "net_margin",
                "净利率",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "debt_to_assets",
                "资产负债率",
                "lower_better",
                0.05,
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "operating_cashflow_to_profit",
                "经营现金流/净利润",
                "higher_better",
                0.05,
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "dividend_yield",
                "股息率",
                "higher_better",
                0.05,
                weight_candidates=[0.02, 0.05, 0.08],
            ),
            _factor(
                "dividend_stability",
                "连续分红稳定性",
                "higher_better",
                0.04,
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "dividend_coverage",
                "现金流覆盖分红",
                "higher_better",
                0.03,
                weight_candidates=[0.01, 0.03, 0.05],
            ),
            _factor(
                "realized_volatility",
                "120日波动率",
                "lower_better",
                0.05,
                default_lookback=120,
                lookback_candidates=[60, 120, 180],
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "downside_volatility",
                "下行波动率",
                "lower_better",
                0.05,
                default_lookback=60,
                lookback_candidates=[40, 60, 120],
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "max_drawdown_window",
                "阶段最大回撤",
                "lower_better",
                0.05,
                default_lookback=120,
                lookback_candidates=[60, 120, 180],
                weight_candidates=[0.03, 0.05, 0.08],
            ),
            _factor(
                "seasoned_momentum",
                "剔除近月动量",
                "higher_better",
                0.06,
                default_lookback=180,
                lookback_candidates=[120, 180, 240],
                weight_candidates=[0.03, 0.06, 0.09],
            ),
            _factor(
                "ma_trend",
                "均线趋势确认",
                "higher_better",
                0.03,
                default_lookback=120,
                lookback_candidates=[60, 120, 180],
                weight_candidates=[0.01, 0.03, 0.05],
            ),
            _factor(
                "recent_overheat_return",
                "近期过热涨幅",
                "lower_better",
                0.04,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.02, 0.04, 0.06],
            ),
            _factor(
                "liquidity_turnover",
                "流动性成交额",
                "higher_better",
                0.03,
                default_lookback=20,
                lookback_candidates=[10, 20, 40],
                weight_candidates=[0.01, 0.03, 0.05],
            ),
        ],
    ),
)

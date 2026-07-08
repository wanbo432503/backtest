from fastapi.testclient import TestClient

import main
from strategy_metadata import get_strategy_metadata


RETAINED_STRATEGIES = {
    "macd_volume_divergence_risk_control",
    "ma_breakout_atr_risk_control",
    "rsi_risk_control",
    "ma_trend_risk_control",
    "volume_breakout_risk_control",
}


def test_reserved_rsi_risk_control_metadata_exists_before_strategy_file():
    metadata = get_strategy_metadata("rsi_risk_control")
    params = {param.name for param in metadata.parameters}

    assert "stop_loss_pct" in params
    assert "take_profit_pct" in params
    assert "cooldown_bars" in params


def test_macd_volume_divergence_metadata_exposes_optimizable_parameters():
    metadata = get_strategy_metadata("macd_volume_divergence_risk_control")
    params = {param.name: param for param in metadata.parameters}

    assert metadata.label == "MACD放量背离风控策略"
    assert set(params) == {
        "fast_period",
        "slow_period",
        "signal_period",
        "volume_lookback",
        "volume_multiplier",
        "divergence_lookback",
        "zero_axis_threshold",
        "trend_ma",
        "histogram_fade_bars",
        "continuation_volume_multiplier",
        "continuation_pullback_pct",
        "stop_loss_pct",
        "take_profit_pct",
        "trailing_stop_pct",
        "max_holding_bars",
        "position_pct",
    }
    for param in params.values():
        assert param.search_values
        assert param.default in param.search_values
    assert params["trend_ma"].search_values == [30, 60, 90]
    assert params["histogram_fade_bars"].search_values == [3, 4, 5]
    assert params["trailing_stop_pct"].search_values == [8, 10, 12]
    assert params["continuation_volume_multiplier"].search_values == [1.0, 1.2, 1.5]
    assert params["continuation_pullback_pct"].search_values == [6, 8, 12]


def test_ma_breakout_atr_metadata_exposes_trend_breakout_parameters():
    metadata = get_strategy_metadata("ma_breakout_atr_risk_control")
    params = {param.name: param for param in metadata.parameters}

    assert metadata.label == "均线突破ATR风控策略"
    assert set(params) == {
        "short_ma",
        "medium_ma",
        "long_ma",
        "breakout_lookback",
        "volume_lookback",
        "volume_multiplier",
        "bootstrap_bars",
        "atr_period",
        "atr_stop_multiplier",
        "max_holding_bars",
        "target_atr_risk_pct",
        "min_position_pct",
        "max_position_pct",
    }
    for param in params.values():
        assert param.search_values
        assert param.default in param.search_values
    assert params["short_ma"].default == 20
    assert params["medium_ma"].default == 60
    assert params["long_ma"].default == 120
    assert params["breakout_lookback"].default == 40
    assert params["volume_multiplier"].default == 1.5
    assert params["bootstrap_bars"].default == 120
    assert params["bootstrap_bars"].search_values == [0, 60, 120]
    assert params["atr_stop_multiplier"].default == 2.5


def test_strategies_endpoint_includes_parameter_metadata():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert set(strategies) == RETAINED_STRATEGIES
    assert "parameters" in strategies["macd_volume_divergence_risk_control"]
    assert strategies["macd_volume_divergence_risk_control"]["parameters"]

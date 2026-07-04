from fastapi.testclient import TestClient

import main
from strategy_metadata import get_strategy_metadata


def test_sma_cross_metadata_defines_optimizable_parameters():
    metadata = get_strategy_metadata("sma_cross")

    assert metadata.name == "sma_cross"
    assert [param.name for param in metadata.parameters] == ["n1", "n2"]
    for param in metadata.parameters:
        assert param.label
        assert param.type == "int"
        assert param.default in param.search_values


def test_rsi_metadata_defines_threshold_parameters():
    metadata = get_strategy_metadata("rsi")
    params = {param.name: param for param in metadata.parameters}

    assert set(params) == {"rsi_period", "rsi_lower", "rsi_upper"}
    assert params["rsi_lower"].search_values == [25, 30, 35]
    assert params["rsi_upper"].search_values == [60, 70, 80]


def test_reserved_rsi_risk_control_metadata_exists_before_strategy_file():
    metadata = get_strategy_metadata("rsi_risk_control")
    params = {param.name for param in metadata.parameters}

    assert "stop_loss_pct" in params
    assert "take_profit_pct" in params
    assert "cooldown_bars" in params


def test_strategies_endpoint_includes_parameter_metadata():
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert "parameters" in strategies["sma_cross"]
    assert strategies["sma_cross"]["parameters"][0]["name"] == "n1"
    assert strategies["sma_cross"]["parameters"][0]["search_values"]

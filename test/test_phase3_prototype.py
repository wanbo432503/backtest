from test.fixtures.portfolio_ohlcv import (
    build_demo_portfolio_request,
    build_invalid_universe_symbols,
    build_ohlcv_frame,
    build_portfolio_ohlcv_fixture,
)


def test_phase3_demo_request_contract():
    request = build_demo_portfolio_request()

    assert request["universe"]["symbols"] == ["SH603019", "SZ002241"]
    assert request["rebalance"]["frequency"] == "monthly"
    assert request["selection"]["top_n"] == 1
    assert request["data_provider"] == "auto"
    assert request["start_date"] < request["end_date"]


def test_phase3_invalid_universe_contract():
    symbols = build_invalid_universe_symbols()

    assert "SZ300750" in symbols
    assert "SH688001" in symbols
    assert len(symbols) == 5


def test_phase3_portfolio_ohlcv_fixture_has_stable_shape_and_late_momentum():
    data_by_symbol = build_portfolio_ohlcv_fixture(["SH603019", "SZ002241"])

    assert set(data_by_symbol) == {"SH603019", "SZ002241"}
    for data in data_by_symbol.values():
        assert len(data) >= 180
        assert list(data.columns) == ["Open", "High", "Low", "Close", "Volume"]

    late_momentum = data_by_symbol["SZ002241"]["Close"]
    early_return = late_momentum.iloc[90] / late_momentum.iloc[30] - 1
    late_return = late_momentum.iloc[-1] / late_momentum.iloc[-61] - 1

    assert late_return > early_return


def test_phase3_ohlcv_fixture_can_create_limit_days():
    data = build_ohlcv_frame(
        start_date="2025-01-01",
        periods=200,
        base_price=20,
        limit_up_days=[10],
        limit_down_days=[20],
    )

    assert data["Close"].iloc[10] / data["Close"].iloc[9] - 1 >= 0.10
    assert data["Close"].iloc[20] / data["Close"].iloc[19] - 1 <= -0.10

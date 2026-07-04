from fastapi.testclient import TestClient

import main
from strategies.volume_breakout_risk_control import should_enter_breakout


def test_breakout_with_volume_enters():
    assert should_enter_breakout(
        close=12.5,
        previous_highest_close=12,
        volume=2500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=11.8,
        limit_up_down_filter=True,
    )


def test_breakout_without_enough_volume_does_not_enter():
    assert not should_enter_breakout(
        close=12.5,
        previous_highest_close=12,
        volume=1500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=11.8,
        limit_up_down_filter=True,
    )


def test_limit_up_filter_blocks_entry():
    assert not should_enter_breakout(
        close=11,
        previous_highest_close=10.5,
        volume=2500000,
        average_volume=1000000,
        volume_multiplier=2,
        previous_close=10,
        limit_up_down_filter=True,
    )


def test_volume_breakout_strategy_appears_in_strategy_list():
    main.load_strategy_modules()
    client = TestClient(main.app)

    response = client.get("/strategies")

    assert response.status_code == 200
    strategies = {item["name"]: item for item in response.json()}
    assert strategies["volume_breakout_risk_control"]["display_name"] == "放量突破风控策略"
    assert strategies["volume_breakout_risk_control"]["parameters"]

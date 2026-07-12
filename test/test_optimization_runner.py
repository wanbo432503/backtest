import time
from threading import Lock

from backtest_runner import BacktestResult
from optimization_models import OptimizationConfig, OptimizationRequest, StrategyParamConfig
from optimization_runner import (
    expand_search_space,
    run_optimization,
    run_train_validate,
    score_backtest_result,
)
from optimization_models import AShareTradingConfig
from strategy_library import get_strategy_library


def test_expand_search_space_builds_cartesian_product():
    combos = expand_search_space({"rsi_period": [6, 14], "rsi_buy": [25, 30]}, max_combinations=10)

    assert combos == [
        {"rsi_period": 6, "rsi_buy": 25},
        {"rsi_period": 6, "rsi_buy": 30},
        {"rsi_period": 14, "rsi_buy": 25},
        {"rsi_period": 14, "rsi_buy": 30},
    ]


def test_expand_search_space_truncates_at_max_combinations():
    combos = expand_search_space({"a": [1, 2, 3], "b": [10, 20, 30]}, max_combinations=4)

    assert len(combos) == 4


def test_expand_search_space_samples_across_all_parameters_when_truncated():
    combos = expand_search_space(
        {
            "short_ma": [10, 20, 30],
            "medium_ma": [40, 60, 90],
            "long_ma": [90, 120, 200],
            "breakout_lookback": [20, 40, 60],
            "volume_lookback": [10, 20, 30],
            "volume_multiplier": [1.2, 1.5, 2.0],
            "atr_period": [10, 14, 20],
            "atr_stop_multiplier": [2.0, 2.5, 3.0],
        },
        max_combinations=30,
    )

    assert len(combos) == 30
    assert {combo["short_ma"] for combo in combos} == {10, 20, 30}
    assert {combo["medium_ma"] for combo in combos} == {40, 60, 90}
    assert {combo["long_ma"] for combo in combos} == {90, 120, 200}
    assert {combo["breakout_lookback"] for combo in combos} == {20, 40, 60}


def test_score_backtest_result_reads_core_score():
    result = BacktestResult(
        plot_html="",
        stats={},
        metrics={"score": 3.5},
        symbol="SH603019",
        interval="1d",
        data_provider="test",
        data_warnings=[],
    )

    assert score_backtest_result(result) == 3.5


def test_train_validate_passes_requested_a_share_rules_to_both_runs(monkeypatch):
    calls = []

    def fake_run_single_backtest(**kwargs):
        calls.append(kwargs)
        return _fake_result(kwargs["symbol"], score=1, trades=8)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)
    trading = AShareTradingConfig(lot_size=200, slippage_pct=0.15)
    request = OptimizationRequest(
        start_date="2025-01-01",
        end_date="2026-01-01",
        a_share_config=trading,
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
        ),
    )

    run_train_validate(
        "SH603019",
        request.optimization_config.strategies[0],
        {},
        request,
        get_strategy_library(),
    )

    assert len(calls) == 2
    assert all(call["trading_config"] == trading for call in calls)


def test_run_optimization_marks_low_trade_results_filtered(monkeypatch):
    def fake_run_single_backtest(**kwargs):
        return _fake_result(kwargs["symbol"], score=4, trades=1)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
            min_trades=5,
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert result.top_results[0]["risk_flags"] == ["too_few_trades"]
    assert result.top_results[0]["recommended"] is False


def test_run_optimization_sorts_by_validate_score(monkeypatch):
    scores = {
        ("SH603019", 6): 1,
        ("SH603019", 14): 5,
    }

    def fake_run_single_backtest(**kwargs):
        period = kwargs["strategy_params"]["rsi_period"]
        return _fake_result(kwargs["symbol"], score=scores[(kwargs["symbol"], period)], trades=8)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[
                StrategyParamConfig(
                    strategy_name="rsi_risk_control",
                    search_space={"rsi_period": [6, 14]},
                )
            ],
            top_n=2,
            min_trades=5,
            train_start_date="2025-07-03",
            train_end_date="2025-12-31",
            validate_start_date="2026-01-01",
            validate_end_date="2026-07-04",
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert result.top_results[0]["params"] == {"rsi_period": 14}
    assert result.top_results[0]["validate_score"] == 5
    assert result.top_results[1]["params"] == {"rsi_period": 6}


def test_run_optimization_uses_train_score_as_validate_tiebreaker(monkeypatch):
    scores = {
        6: (2, 10),
        14: (2, 20),
    }

    def fake_run_single_backtest(**kwargs):
        period = kwargs["strategy_params"]["rsi_period"]
        is_validate = kwargs["start_date"] == "2026-01-01"
        validate_score, train_score = scores[period]
        return _fake_result(
            kwargs["symbol"],
            score=validate_score if is_validate else train_score,
            trades=8,
        )

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[
                StrategyParamConfig(
                    strategy_name="rsi_risk_control",
                    search_space={"rsi_period": [6, 14]},
                )
            ],
            top_n=2,
            min_trades=5,
            train_start_date="2025-07-03",
            train_end_date="2025-12-31",
            validate_start_date="2026-01-01",
            validate_end_date="2026-07-04",
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert result.top_results[0]["params"] == {"rsi_period": 14}
    assert result.top_results[0]["validate_score"] == 2
    assert result.top_results[0]["train_score"] == 20
    assert result.top_results[1]["params"] == {"rsi_period": 6}


def test_run_optimization_ranks_tradeable_results_before_no_trade_results(monkeypatch):
    scores = {
        6: (0, 0),
        14: (-1, 8),
    }

    def fake_run_single_backtest(**kwargs):
        period = kwargs["strategy_params"]["rsi_period"]
        score, trades = scores[period]
        return _fake_result(kwargs["symbol"], score=score, trades=trades)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[
                StrategyParamConfig(
                    strategy_name="rsi_risk_control",
                    search_space={"rsi_period": [6, 14]},
                )
            ],
            top_n=2,
            min_trades=5,
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert result.top_results[0]["params"] == {"rsi_period": 14}
    assert result.top_results[0]["validate_score"] == -1
    assert result.top_results[1]["params"] == {"rsi_period": 6}
    assert "too_few_trades" in result.top_results[1]["risk_flags"]


def test_run_optimization_runs_trials_in_parallel_batches_and_sorts_top_results(monkeypatch):
    active_trials = 0
    max_active_trials = 0
    lock = Lock()

    def fake_run_train_validate(**kwargs):
        nonlocal active_trials, max_active_trials
        with lock:
            active_trials += 1
            max_active_trials = max(max_active_trials, active_trials)
        time.sleep(0.03)
        with lock:
            active_trials -= 1

        period = kwargs["params"]["rsi_period"]
        return {
            "symbol": kwargs["symbol"],
            "strategy_name": kwargs["strategy_config"].strategy_name,
            "params": kwargs["params"],
            "train_metrics": {"score": period, "trades": 8},
            "validate_metrics": {"score": period, "trades": 8},
            "train_score": period,
            "validate_score": period,
            "validate_stats": {},
            "data_provider": "test",
            "data_warnings": [],
        }

    monkeypatch.setattr("optimization_runner.run_train_validate", fake_run_train_validate)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[
                StrategyParamConfig(
                    strategy_name="rsi_risk_control",
                    search_space={"rsi_period": [6, 14, 20, 30]},
                )
            ],
            top_n=2,
            min_trades=5,
            max_workers=2,
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert max_active_trials == 2
    assert [row["params"]["rsi_period"] for row in result.top_results] == [30, 20]
    assert [row["rank"] for row in result.top_results] == [1, 2]


def test_run_optimization_emits_trial_progress(monkeypatch):
    def fake_run_single_backtest(**kwargs):
        return _fake_result(kwargs["symbol"], score=3, trades=8)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)
    events = []
    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[
                StrategyParamConfig(
                    strategy_name="rsi_risk_control",
                    search_space={"rsi_period": [6, 14]},
                )
            ],
            top_n=2,
            min_trades=5,
        ),
    )

    run_optimization(
        request,
        strategy_library=get_strategy_library(),
        progress_callback=events.append,
    )

    optimizing_events = [event for event in events if event["phase"] == "optimizing"]
    assert optimizing_events[0]["total_trials"] == 2
    assert optimizing_events[0]["completed_trials"] == 0
    assert optimizing_events[-1]["completed_trials"] == 2
    assert optimizing_events[-1]["current_symbol"] == "SH603019"
    assert optimizing_events[-1]["current_strategy"] == "rsi_risk_control"
    assert events[-1]["phase"] == "completed"
    assert events[-1]["total_trials"] == 2


def test_run_optimization_flags_negative_validation_score(monkeypatch):
    def fake_run_single_backtest(**kwargs):
        is_validate = kwargs["start_date"] == "2026-01-01"
        return _fake_result(kwargs["symbol"], score=-1 if is_validate else 9, trades=8)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
            train_start_date="2025-07-03",
            train_end_date="2025-12-31",
            validate_start_date="2026-01-01",
            validate_end_date="2026-07-04",
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert "validation_score_negative" in result.top_results[0]["risk_flags"]
    assert result.top_results[0]["recommended"] is False


def test_run_optimization_flags_possible_overfit(monkeypatch):
    def fake_run_single_backtest(**kwargs):
        is_validate = kwargs["start_date"] == "2026-01-01"
        return _fake_result(kwargs["symbol"], score=2 if is_validate else 12, trades=8)

    monkeypatch.setattr("optimization_runner.run_single_backtest", fake_run_single_backtest)

    request = OptimizationRequest(
        start_date="2025-07-03",
        end_date="2026-07-04",
        optimization_config=OptimizationConfig(
            symbols=["SH603019"],
            strategies=[StrategyParamConfig(strategy_name="rsi_risk_control")],
            train_start_date="2025-07-03",
            train_end_date="2025-12-31",
            validate_start_date="2026-01-01",
            validate_end_date="2026-07-04",
        ),
    )

    result = run_optimization(request, strategy_library=get_strategy_library())

    assert "possible_overfit" in result.top_results[0]["risk_flags"]
    assert result.top_results[0]["recommended"] is False


def _fake_result(symbol: str, score: float, trades: int) -> BacktestResult:
    return BacktestResult(
        plot_html="",
        stats={"综合评分": f"{score:.2f}"},
        metrics={
            "score": score,
            "trades": trades,
            "is_rankable": trades >= 5,
            "risk_notes": [],
        },
        symbol=symbol,
        interval="1d",
        data_provider="test",
        data_warnings=[],
    )

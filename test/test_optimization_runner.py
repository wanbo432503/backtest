from backtest_runner import BacktestResult
from optimization_models import OptimizationConfig, OptimizationRequest, StrategyParamConfig
from optimization_runner import (
    expand_search_space,
    run_optimization,
    score_backtest_result,
)


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

    result = run_optimization(request, strategy_registry={"rsi_risk_control": object})

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

    result = run_optimization(request, strategy_registry={"rsi_risk_control": object})

    assert result.top_results[0]["params"] == {"rsi_period": 14}
    assert result.top_results[0]["validate_score"] == 5
    assert result.top_results[1]["params"] == {"rsi_period": 6}


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
        strategy_registry={"rsi_risk_control": object},
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

    result = run_optimization(request, strategy_registry={"rsi_risk_control": object})

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

    result = run_optimization(request, strategy_registry={"rsi_risk_control": object})

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

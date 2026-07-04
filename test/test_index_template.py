from pathlib import Path
import re


def test_index_template_only_shows_a_share_symbol_examples():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "股票/加密货币代码" not in template
    assert "AAPL" not in template
    assert "MSFT" not in template
    assert "BTC-USD" not in template
    assert "ETH-USD" not in template
    assert "A股代码" in template
    assert "setSymbol('SH603019')\">中科曙光</span>" in template
    assert "setSymbol('SZ002241')\">歌尔股份</span>" in template
    assert len(re.findall(r'class="symbol-suggestion"', template)) == 2


def test_index_template_reads_strategy_parameter_metadata():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "window.strategyMetadata" in template
    assert "strategy.parameters || []" in template
    assert "可优化参数" in template


def test_index_template_contains_optimization_controls():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "股票池" in template
    assert 'id="poolSH603019"' in template
    assert 'id="poolSZ002241"' in template
    assert "中科曙光" in template
    assert "歌尔股份" in template
    assert "启用参数优化" in template
    assert "score" in template
    assert "collectOptimizationRequest" in template
    assert "开始优化" in template


def test_index_template_contains_risk_and_a_share_rule_controls():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "策略参数" in template
    assert 'id="strategyParamContainer"' in template
    assert "风控设置" in template
    assert 'id="stopLossPct"' in template
    assert 'id="takeProfitPct"' in template
    assert 'id="positionPct"' in template
    assert "A 股交易规则" in template
    assert "T+1" in template
    assert "涨跌停过滤" in template
    assert 'id="lotSize"' in template
    assert 'id="slippagePct"' in template
    assert "collectRiskConfig" in template
    assert "collectAShareConfig" in template
    assert "strategy_params" in template


def test_index_template_renders_optimization_result_table_actions():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "优化结果" in template
    assert "参数摘要" in template
    assert "训练 score" in template
    assert "应用参数" in template
    assert "回测该参数" in template
    assert "applyOptimizationResult" in template
    assert "backtestOptimizationResult" in template
    assert "setOptimizationResultsCollapsed" in template
    assert "risk_flags" in template
    assert "badge bg-warning" in template
    assert "可能过拟合" in template


def test_index_template_keeps_optimization_results_collapsible_after_backtest():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "optimizationResultsPanel" in template
    assert "optimizationResultsBody" in template
    assert "optimizationResultToggleIcon" in template
    assert "optimization-results-collapsed" in template
    assert "function setOptimizationResultsCollapsed" in template
    assert "setOptimizationResultsCollapsed(true)" in template
    assert "renderBacktestStats(result)" in template


def test_index_template_renames_validation_start_label():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "验证区间开始" in template
    assert "验证开始" not in template


def test_index_template_has_collapsible_trade_settings_and_insight_panel():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "<summary><i class=\"fas fa-briefcase\"></i> 交易设置</summary>" in template
    assert 'id="tradeSettingsPanel"' in template
    assert 'id="rightInsightColumn"' in template
    assert 'id="toggleInsightPanelButton"' in template
    assert 'onclick="toggleInsightPanel()"' in template
    assert "right-insights-collapsed" in template
    assert "function toggleInsightPanel()" in template

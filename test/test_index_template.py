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

    assert "优化标的" in template
    assert "使用上方当前 A 股代码" in template
    assert "股票池" not in template
    assert 'id="poolSH603019"' not in template
    assert 'id="poolSZ002241"' not in template
    assert 'id="thirdPoolSymbol"' not in template
    assert "optimization-symbol" not in template
    assert "启用参数优化" in template
    assert "score" in template
    assert "collectOptimizationRequest" in template
    assert "collectOptimizationSymbol" in template
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


def test_index_template_aligns_a_share_rule_number_inputs_in_three_rows():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'class="trade-rule-number-grid"' in template
    grid_match = re.search(
        r'<div class="trade-rule-number-grid">(.*?)</div>\s*</details>',
        template,
        flags=re.S,
    )
    assert grid_match is not None
    grid_html = grid_match.group(1)
    for control_id in [
        "slippagePct",
        "lotSize",
        "buyCommissionPct",
        "sellCommissionPct",
        "stampTaxPct",
        "minCommission",
    ]:
        assert f'id="{control_id}"' in grid_html


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
    assert ".optimization-results-collapsed .optimization-results-body" in template
    assert ".optimization-results-collapsed #optimizationResultsBody" not in template
    assert "function setOptimizationResultsCollapsed" in template
    assert "setOptimizationResultsCollapsed(true)" in template
    assert "renderBacktestStats(result)" in template


def test_index_template_persists_current_and_historical_optimization_results():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="optimizationLibraryContainer"' in template
    assert 'id="optimizationCurrentHost"' in template
    assert 'id="optimizationHistoryHost"' in template
    assert "当前优化列表" in template
    assert "历史优化列表" in template
    assert "backtest.optimizationResults.v1" in template
    assert "function saveOptimizationResultsToStorage" in template
    assert "function deleteOptimizationResult" in template
    assert "function applyOptimizationResultById" in template
    assert "function backtestOptimizationResultById" in template
    assert "function isOptimizationResultForSelectedStrategy" in template
    assert "renderOptimizationLibrary();" in template
    assert "删除" in template


def test_index_template_delegates_optimization_result_actions_without_broken_inline_js():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'data-optimization-action="apply"' in template
    assert 'data-optimization-action="backtest"' in template
    assert 'data-optimization-action="delete"' in template
    assert 'data-optimization-action="toggle"' in template
    assert "function handleOptimizationLibraryClick" in template
    assert "addEventListener('click', handleOptimizationLibraryClick)" in template
    assert 'onclick="deleteOptimizationResult(' not in template
    assert 'onclick="backtestOptimizationResultById(' not in template
    assert 'onclick="applyOptimizationResultById(' not in template


def test_index_template_renames_validation_start_label():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "验证区间开始" in template
    assert "验证开始" not in template


def test_index_template_has_collapsible_trade_settings_and_analysis_panel():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "<summary><i class=\"fas fa-briefcase\"></i> 交易设置</summary>" in template
    assert 'id="tradeSettingsPanel"' in template
    assert 'id="rightAnalysisColumn"' in template
    assert 'id="toggleAnalysisPanelButton"' in template
    assert 'onclick="toggleAnalysisPanel()"' in template
    assert "right-analysis-collapsed" in template
    assert "function toggleAnalysisPanel()" in template


def test_index_template_contains_tradingagents_analysis_tab_shell():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "智能分析" in template
    assert 'id="tradingAgentsTabs"' in template
    assert 'id="taAnalysisTab"' in template
    assert 'id="taSettingsTab"' in template
    assert 'id="taSymbol"' in template
    assert 'id="taAnalysisDate"' in template
    assert 'id="taAnalystMarket"' in template
    assert 'id="taAnalystNews"' in template
    assert 'id="taAnalystFundamentals"' in template
    assert 'id="taAnalystSocial"' in template
    assert 'id="taRunButton"' in template
    assert 'id="taClearButton"' in template
    assert 'id="taStatus"' in template
    assert 'id="taReportTabs"' in template
    assert 'id="taReportContent"' in template
    assert "function getSelectedTradingAgentsAnalysts()" in template
    assert "function setTradingAgentsStatus" in template
    assert "function renderTradingAgentsReports" in template

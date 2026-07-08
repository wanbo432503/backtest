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
    assert "optimization-symbol" not in template
    assert "启用参数优化" in template
    assert "score" in template
    assert "collectOptimizationRequest" in template
    assert "collectOptimizationSymbol" in template
    assert "开始优化" in template


def test_index_template_contains_phase3_portfolio_workbench_controls():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "组合选股回测" in template
    assert '<i class="fas fa-layer-group"></i> 组合选股回测' in template
    assert 'id="portfolioBacktestForm"' in template
    assert "自动扫描 60/00 股票池" in template
    assert 'id="portfolioUniverseMode"' in template
    assert 'id="portfolioSymbols"' in template
    assert "SH603019" in template
    assert "SZ002241" in template
    assert "高级诊断" in template
    assert "手动候选池" in template
    assert "最终持仓最多 20 只" in template
    assert "仅支持 60/00 开头" in template
    assert 'id="portfolioMaxScanSymbols"' in template
    assert 'id="portfolioOhlcvBatchSize"' in template
    assert 'id="portfolioOhlcvRequestDelay"' in template
    assert 'id="portfolioOhlcvBatchDelay"' in template
    assert 'id="portfolioMinAvgTurnover"' in template
    assert 'id="portfolioMinAvgVolume"' in template
    assert 'id="portfolioMinPrice"' in template
    assert 'id="portfolioMaxPrice"' in template
    assert 'id="portfolioTopN"' in template
    assert 'id="portfolioTopN" min="1" max="20"' in template
    assert 'id="portfolioRebalanceFrequency"' in template
    assert 'id="portfolioMomentumWeight"' in template
    assert 'id="portfolioVolatilityWeight"' in template
    assert 'id="portfolioLiquidityWeight"' in template
    assert 'id="portfolioTrendWeight"' in template
    assert 'id="portfolioVolatilityLookback"' in template
    assert 'id="portfolioLiquidityLookback"' in template
    assert 'id="portfolioScoreThreshold"' in template
    assert 'id="singleStockDiagnosticPanel"' in template
    assert "单股诊断" in template
    assert "加入股票池" in template
    assert "function addSymbolToPortfolio" in template
    assert "function isPortfolioManualMode" in template
    assert "function collectPortfolioSymbols()" in template
    assert "function validatePortfolioUniverse" in template
    assert "function collectPortfolioRequest()" in template
    assert "mode: getPortfolioUniverseMode()" in template
    assert "fetch('/portfolio-backtest/jobs'" in template
    assert "function pollPortfolioBacktestJob" in template
    assert "function renderPortfolioProgress" in template


def test_index_template_renders_phase3_portfolio_results():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="portfolioResultPanel"' in template
    assert 'id="portfolioProgressPanel"' in template
    assert 'id="portfolioProgressBar"' in template
    assert 'id="portfolioProgressText"' in template
    assert 'id="portfolioEquityChart"' in template
    assert 'id="portfolioPositionsTable"' in template
    assert 'id="portfolioRebalanceTable"' in template
    assert 'id="portfolioCandidateTable"' in template
    assert "function renderPortfolioResult(result)" in template
    assert "function scrollPortfolioPanelIntoView" in template
    assert "scrollPortfolioPanelIntoView('portfolioProgressPanel')" in template
    assert "scrollPortfolioPanelIntoView('portfolioResultPanel')" in template
    assert "function renderPortfolioSummary" in template
    assert "function renderPortfolioEquityCurve" in template
    assert "function renderPortfolioTable" in template
    assert "candidate_rankings" in template
    assert "rebalance_events" in template
    assert "scan_diagnostics" in template
    assert 'id="portfolioScanDiagnostics"' in template
    assert "function renderPortfolioScanDiagnostics" in template
    assert 'id="portfolioSummaryButton"' in template
    assert 'id="portfolioSummaryPanel"' in template
    assert "function collectPortfolioSummaryPayload" in template
    assert "function runPortfolioSummary" in template
    assert "function renderPortfolioSummaryExplanation" in template
    assert "fetchJson('/tradingagents/portfolio-summary'" in template
    assert "risk_flags" in template
    assert "syncTradingAgentsSymbol" in template


def test_index_template_contains_portfolio_factor_optimization_controls():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="portfolioFactorOptimizationPanel"' in template
    assert "因子优化" in template
    assert "参数候选" in template
    assert "虚拟盘参考" in template
    assert 'id="portfolioOptimizationMaxTrials"' in template
    assert 'id="portfolioOptimizationMaxWorkers"' in template
    assert 'id="portfolioOptimizationTrainRatio"' in template
    assert 'id="portfolioOptimizationExecutorBackend"' in template
    assert 'id="portfolioOptimizationIncludeTopN"' in template
    assert 'id="portfolioOptimizationTopNCandidates"' in template
    assert 'id="portfolioOptimizationMomentumLookbacks"' in template
    assert 'id="portfolioOptimizationVolatilityLookbacks"' in template
    assert 'id="portfolioOptimizationLiquidityLookbacks"' in template
    assert 'id="portfolioOptimizationMomentumWeights"' in template
    assert 'id="portfolioOptimizationVolatilityWeights"' in template
    assert 'id="portfolioOptimizationLiquidityWeights"' in template
    assert 'id="portfolioOptimizationTrendWeights"' in template
    assert 'id="portfolioOptimizationScoreThresholds"' in template
    assert 'id="startPortfolioFactorOptimizationButton"' in template
    assert "function parsePortfolioOptimizationNumberList" in template
    assert "function collectPortfolioFactorOptimizationRequest()" in template
    assert "function createPortfolioFactorOptimizationJob" in template
    assert "function pollPortfolioFactorOptimizationJob" in template
    assert "fetchJson('/portfolio-factor-optimization/jobs'" in template
    assert "portfolio-factor-optimization/jobs/${encodeURIComponent(jobId)}" in template


def test_index_template_contains_portfolio_selection_strategy_controls():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="portfolioSelectionStrategy"' in template
    assert 'id="portfolioSelectionStrategyDescription"' in template
    assert 'id="portfolioSelectionStrategyCaveats"' in template
    assert 'id="portfolioApplyStrategyDefaultsButton"' in template
    assert 'id="portfolioStrategyFactorSummary"' in template
    assert "loadPortfolioSelectionStrategies" in template
    assert "renderPortfolioSelectionStrategyOptions" in template
    assert "applyPortfolioSelectionStrategyDefaults" in template
    assert "collectPortfolioSelectionStrategyConfig" in template
    assert "selection_strategy: collectPortfolioSelectionStrategyConfig()" in template
    assert "portfolio-selection-strategies" in template
    assert "稳健低波动动量策略" in template
    assert "自定义因子组合" in template


def test_index_template_renders_portfolio_factor_optimization_results_and_apply_flow():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="portfolioFactorOptimizationProgressPanel"' in template
    assert 'id="portfolioFactorOptimizationProgressBar"' in template
    assert 'id="portfolioFactorOptimizationProgressText"' in template
    assert 'id="portfolioFactorOptimizationResults"' in template
    assert 'id="portfolioFactorOptimizationResultBody"' in template
    assert 'id="portfolioFactorOptimizationApplyStatus"' in template
    assert "function renderPortfolioFactorOptimizationProgress" in template
    assert "function renderPortfolioFactorOptimizationResults" in template
    assert "function applyPortfolioFactorOptimizationResult" in template
    assert "portfolioMomentumLookback').value = factorConfig.momentum_lookback" in template
    assert "portfolioVolatilityLookback').value = factorConfig.volatility_lookback" in template
    assert "portfolioLiquidityLookback').value = factorConfig.liquidity_lookback" in template
    assert "portfolioMomentumWeight').value = factorConfig.momentum_weight" in template
    assert "portfolioVolatilityWeight').value = factorConfig.volatility_weight" in template
    assert "portfolioLiquidityWeight').value = factorConfig.liquidity_weight" in template
    assert "portfolioTrendWeight').value = factorConfig.trend_weight" in template
    assert "portfolioTopN').value = selectionConfig.top_n" in template
    assert "不会自动下单" in template
    assert "应用参数" in template
    assert "验证年化" in template
    assert "验证波动" in template
    assert "趋势R²" in template


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


def test_index_template_contains_tradingagents_settings_tab_shell():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert 'id="taProvider"' in template
    assert 'id="taBackendUrl"' in template
    assert 'id="taApiKey"' in template
    assert 'id="taToggleApiKeyButton"' in template
    assert 'id="taClearApiKey"' in template
    assert 'id="taDeepModel"' in template
    assert 'id="taQuickModel"' in template
    assert 'id="taOutputLanguage"' in template
    assert 'id="taDebateRounds"' in template
    assert 'id="taRiskRounds"' in template
    assert 'id="taCheckpointEnabled"' in template
    assert 'id="taTemperature"' in template
    assert 'id="taReasoningEffort"' in template
    assert 'id="taSaveConfigButton"' in template
    assert 'id="taReloadConfigButton"' in template
    assert 'id="taTestConfigButton"' in template
    assert 'id="taConfigStatus"' in template
    assert "function populateTradingAgentsConfig(config)" in template
    assert "function setTradingAgentsApiKeyVisibility(visible)" in template
    assert "async function toggleTradingAgentsApiKeyVisibility()" in template
    assert "function collectTradingAgentsConfigPayload()" in template
    assert "function setTradingAgentsConfigStatus" in template


def test_index_template_connects_tradingagents_panel_to_api():
    template = Path("templates/index.html").read_text(encoding="utf-8")

    assert "async function fetchJson(url, options = {})" in template
    assert "async function loadTradingAgentsConfig()" in template
    assert "async function saveTradingAgentsConfig()" in template
    assert "async function testTradingAgentsConfig()" in template
    assert "async function runTradingAgentsAnalysis()" in template
    assert "fetchJson('/tradingagents/config')" in template
    assert "fetchJson('/tradingagents/config/api-key')" in template
    assert "fetchJson('/tradingagents/config'," in template
    assert "fetchJson('/tradingagents/config/test'" in template
    assert "fetchJson('/tradingagents/analysis'" in template
    assert "fetchJson('/tradingagents/portfolio-summary'" in template
    assert "loadTradingAgentsConfig();" in template
    assert "addEventListener('click', toggleTradingAgentsApiKeyVisibility)" in template
    assert "addEventListener('click', saveTradingAgentsConfig)" in template
    assert "addEventListener('click', loadTradingAgentsConfig)" in template
    assert "addEventListener('click', testTradingAgentsConfig)" in template
    assert "addEventListener('click', runTradingAgentsAnalysis)" in template
    assert "addEventListener('input', event => syncTradingAgentsSymbol(event.target.value))" in template
    assert "当前后端服务未加载 TradingAgents API，请重启 backtest 服务后刷新页面" in template

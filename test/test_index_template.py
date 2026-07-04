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

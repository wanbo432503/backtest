#!/usr/bin/env python3
"""
股票搜索多市场功能示例
演示如何在Python代码中使用股票搜索API支持美股、A股和港股
"""

from stock_search import (
    search_stocks,
    search_us_stocks,
    search_cn_stocks,
    search_hk_stocks,
    get_stock_info,
    MARKET_US,
    MARKET_CN,
    MARKET_HK
)

def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)

def print_result(result: dict):
    """格式化打印股票搜索结果"""
    print(f"  📊 {result['symbol']:15s} | {result['name'][:30]:30s} | {result.get('market_name', result.get('market', 'N/A')):6s}")

# ============================================================
# 示例 1: 基础搜索 - 美股
# ============================================================

print_separator("示例 1: 美股搜索")

print('\n🔍 搜索美股中的"苹果":')
us_results = search_us_stocks("苹果")
print(f"找到 {len(us_results)} 个结果:")
for result in us_results[:3]:
    print_result(result)

print('\n🔍 搜索美股中的"Microsoft":')
us_results = search_us_stocks("Microsoft")
print(f"找到 {len(us_results)} 个结果:")
for result in us_results[:3]:
    print_result(result)

# ============================================================
# 示例 2: A股搜索
# ============================================================

print_separator("示例 2: A股搜索")

print('\n🔍 搜索A股中的"茅台":')
cn_results = search_cn_stocks("茅台")
print(f"找到 {len(cn_results)} 个结果:")
for result in cn_results[:3]:
    print_result(result)

print('\n🔍 搜索A股中的"平安":')
cn_results = search_cn_stocks("平安")
print(f"找到 {len(cn_results)} 个结果:")
for result in cn_results[:3]:
    print_result(result)

print('\n🔍 搜索A股中的"工商银行":')
cn_results = search_cn_stocks("工商银行")
print(f"找到 {len(cn_results)} 个结果:")
for result in cn_results[:3]:
    print_result(result)

# ============================================================
# 示例 3: 港股搜索
# ============================================================

print_separator("示例 3: 港股搜索")

print('\n🔍 搜索港股中的"腾讯":')
hk_results = search_hk_stocks("腾讯")
print(f"找到 {len(hk_results)} 个结果:")
for result in hk_results[:3]:
    print_result(result)

print('\n🔍 搜索港股中的"阿里巴巴":')
hk_results = search_hk_stocks("阿里巴巴")
print(f"找到 {len(hk_results)} 个结果:")
for result in hk_results[:3]:
    print_result(result)

print('\n🔍 搜索港股中的"小米":')
hk_results = search_hk_stocks("小米")
print(f"找到 {len(hk_results)} 个结果:")
for result in hk_results[:3]:
    print_result(result)

# ============================================================
# 示例 4: 跨市场搜索
# ============================================================

print_separator("示例 4: 跨市场搜索（同时搜索所有市场）")

print('\n🔍 在所有市场中搜索"阿里巴巴":')
results = search_stocks("阿里巴巴")
print(f"找到 {len(results)} 个结果:")
for result in results:
    market_map = {MARKET_US: "美股", MARKET_CN: "A股", MARKET_HK: "港股"}
    result['market_name'] = market_map.get(result['market'], result['market'])
    print_result(result)

print('\n🔍 在所有市场中搜索"腾讯":')
results = search_stocks("腾讯")
print(f"找到 {len(results)} 个结果:")
for result in results:
    market_map = {MARKET_US: "美股", MARKET_CN: "A股", MARKET_HK: "港股"}
    result['market_name'] = market_map.get(result['market'], result['market'])
    print_result(result)

print('\n🔍 在所有市场中搜索"百度":')
results = search_stocks("百度")
print(f"找到 {len(results)} 个结果:")
for result in results:
    market_map = {MARKET_US: "美股", MARKET_CN: "A股", MARKET_HK: "港股"}
    result['market_name'] = market_map.get(result['market'], result['market'])
    print_result(result)

# ============================================================
# 示例 5: 按市场代码搜索
# ============================================================

print_separator("示例 5: 按股票代码搜索")

print('\n🔍 按代码搜索美股 "AAPL":')
results = search_stocks("AAPL", market=MARKET_US)
print(f"找到 {len(results)} 个结果:")
for result in results[:3]:
    print_result(result)

print('\n🔍 按代码搜索A股 "SH600519":')
results = search_stocks("SH600519", market=MARKET_CN)
print(f"找到 {len(results)} 个结果:")
for result in results[:3]:
    print_result(result)

print('\n🔍 按代码搜索港股 "0700.HK":')
results = search_stocks("0700.HK", market=MARKET_HK)
print(f"找到 {len(results)} 个结果:")
for result in results[:3]:
    print_result(result)

# ============================================================
# 示例 6: 获取股票详细信息
# ============================================================

print_separator("示例 6: 获取股票详细信息")

print("\n📈 美股: Apple Inc. (AAPL)")
info = get_stock_info("AAPL", market=MARKET_US)
print(f"  公司名称: {info.get('name', 'N/A')}")
print(f"  市场: {info.get('country', 'N/A')}")
print(f"  行业: {info.get('sector', 'N/A')}")
print(f"  货币: {info.get('currency', 'N/A')}")
print(f"  市值: {info.get('market_cap', 'N/A')}")
print(f"  市盈率: {info.get('pe_ratio', 'N/A')}")

print("\n📈 港股: Tencent (0700.HK)")
info = get_stock_info("0700.HK", market=MARKET_HK)
print(f"  公司名称: {info.get('name', 'N/A')}")
print(f"  市场: {info.get('country', 'N/A')}")
print(f"  行业: {info.get('sector', 'N/A')}")
print(f"  货币: {info.get('currency', 'N/A')}")
print(f"  市值: {info.get('market_cap', 'N/A')}")
print(f"  市盈率: {info.get('pe_ratio', 'N/A')}")

# ============================================================
# 示例 7: 实际应用 - 回测前的股票选择流程
# ============================================================

print_separator("示例 7: 回测前的股票选择流程")

def select_stock_for_backtest(stock_name: str):
    """
    为回测选择股票的流程
    """
    print(f"\n🎯 用户输入: {stock_name}")
    
    # 第1步: 搜索股票
    results = search_stocks(stock_name)
    
    if not results:
        print(f"❌ 未找到相关股票")
        return None
    
    print(f"✅ 找到 {len(results)} 个结果:")
    
    # 按市场分组
    by_market = {}
    for i, result in enumerate(results, 1):
        market = result.get('market', 'Unknown')
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(result)
        
        market_map = {MARKET_US: "美股", MARKET_CN: "A股", MARKET_HK: "港股"}
        market_name = market_map.get(market, market)
        print(f"   {i}. [{market_name}] {result['symbol']:15s} - {result['name']}")
    
    # 如果只有一个结果，直接选择
    if len(results) == 1:
        selected = results[0]
    else:
        # 这里实际应用中可以让用户选择
        # 为了演示，我们选择第一个结果
        selected = results[0]
    
    print(f"\n✨ 选择的股票: {selected['symbol']} ({selected['name']})")
    
    # 第2步: 获取股票信息
    info = get_stock_info(selected['symbol'], market=selected['market'])
    print(f"\n📊 股票信息:")
    print(f"   代码: {info.get('symbol', 'N/A')}")
    print(f"   名称: {info.get('name', 'N/A')}")
    print(f"   市场: {info.get('country', 'N/A')}")
    print(f"   货币: {info.get('currency', 'N/A')}")
    print(f"   行业: {info.get('sector', 'N/A')}")
    
    # 第3步: 返回股票代码用于回测
    return selected['symbol']

# 测试股票选择流程
print('\n--- 场景 1: 搜索"茅台" ---')
symbol = select_stock_for_backtest("茅台")
if symbol:
    print(f"\n💡 建议: 使用 '{symbol}' 进行回测")

print('\n--- 场景 2: 搜索"腾讯" ---')
symbol = select_stock_for_backtest("腾讯")
if symbol:
    print(f"\n💡 建议: 使用 '{symbol}' 进行回测")

print('\n--- 场景 3: 搜索"苹果" ---')
symbol = select_stock_for_backtest("苹果")
if symbol:
    print(f"\n💡 建议: 使用 '{symbol}' 进行回测")

# ============================================================
# 总结
# ============================================================

print_separator("总结")
print("""
✅ 新增功能总结:

1. 市场支持:
   - 美股 (US): 使用 search_us_stocks() 或指定 market='US'
   - A股 (CN): 使用 search_cn_stocks() 或指定 market='CN'
   - 港股 (HK): 使用 search_hk_stocks() 或指定 market='HK'
   - 跨市场: 使用 search_stocks() 搜索所有市场

2. 支持的搜索方式:
   - 中文公司名: "苹果", "茅台", "腾讯"
   - 英文公司名: "apple", "moutai", "tencent"
   - 股票代码: "AAPL", "SH600519", "0700.HK"

3. 主要函数:
   - search_stocks(query, market=None): 通用搜索
   - search_us_stocks(query): 美股搜索
   - search_cn_stocks(query): A股搜索
   - search_hk_stocks(query): 港股搜索
   - get_stock_info(symbol, market=None): 获取详细信息

4. API 端点:
   - GET /search-stocks-multi-market: 多市场搜索
   - GET /search-us-stocks: 美股搜索
   - GET /search-cn-stocks: A股搜索
   - GET /search-hk-stocks: 港股搜索
""")

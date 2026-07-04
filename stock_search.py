#!/usr/bin/env python3
"""
股票代码搜索模块 - 通过公司名字查找股票代码
支持多个市场和数据源：
1. 美股 (US Stocks) - 通过 yfinance
2. A股 (Chinese Stocks) - 通过符号转换
3. 港股 (Hong Kong Stocks) - 通过符号转换
4. 本地 CSV 数据库（可选）
5. 常见股票映射表
"""

import yfinance as yf
import pandas as pd
import os
import json
import requests
from typing import List, Dict, Tuple, Optional

# 尝试导入模糊匹配库
try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False
    # 简单的相似度计算函数
    def simple_similarity(s1: str, s2: str) -> int:
        """简单的字符串相似度计算"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 100
        if s1 in s2 or s2 in s1:
            return 80
        # 计算共同字符
        common = sum(1 for c in s1 if c in s2)
        return int((common / max(len(s1), len(s2))) * 100)

# 市场类型常量
MARKET_US = "US"
MARKET_CN = "CN"  # A股
MARKET_HK = "HK"  # 港股

EASTMONEY_SUGGEST_URL = "https://searchapi.eastmoney.com/api/suggest/get"
EASTMONEY_SUGGEST_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"

# 美股常见股票映射表
COMMON_US_STOCKS = {
    # 科技公司
    "苹果": "AAPL",
    "apple": "AAPL",
    "微软": "MSFT",
    "microsoft": "MSFT",
    "谷歌": "GOOGL",
    "google": "GOOGL",
    "特斯拉": "TSLA",
    "tesla": "TSLA",
    "亚马逊": "AMZN",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "脸书": "META",
    "nvidia": "NVDA",
    "英伟达": "NVDA",
    "英特尔": "INTC",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "思科": "CSCO",
    "cisco": "CSCO",
    "oracle": "ORCL",
    "甲骨文": "ORCL",
    "salesforce": "CRM",
    "德州仪器": "TXN",
    "ti": "TXN",
    "zoom": "ZM",
    "苹果电脑": "AAPL",
    "windows": "MSFT",
    
    # 金融公司
    "摩根大通": "JPM",
    "jpmorgan": "JPM",
    "高盛": "GS",
    "美国银行": "BAC",
    "bofa": "BAC",
    "富国银行": "WFC",
    "wells": "WFC",
    "花旗": "C",
    "citigroup": "C",
    "visa": "V",
    "万事达": "MA",
    "mastercard": "MA",
    "美运通": "AXP",
    "amex": "AXP",
    "沃伦巴菲特": "BRK.B",
    "伯克希尔": "BRK.B",
    
    # 消费品公司
    "可口可乐": "KO",
    "coca cola": "KO",
    "百事": "PEP",
    "pepsi": "PEP",
    "麦当劳": "MCD",
    "mcdonalds": "MCD",
    "星巴克": "SBUX",
    "starbucks": "SBUX",
    "耐克": "NKE",
    "nike": "NKE",
    "阿迪达斯": "ADDYY",
    "adidas": "ADDYY",
    "迪士尼": "DIS",
    "disney": "DIS",
    "netflix": "NFLX",
    "奈飞": "NFLX",
    "沃尔玛": "WMT",
    "walmart": "WMT",
    "亚沃": "AZO",
    "autozone": "AZO",
    
    # 健康医疗
    "约翰逊": "JNJ",
    "强生": "JNJ",
    "johnson": "JNJ",
    "pfizer": "PFE",
    "辉瑞": "PFE",
    "merck": "MRK",
    "默克": "MRK",
    "修正药业": "ABBV",
    "艾伯维": "ABBV",
    "amgen": "AMGN",
    "安进": "AMGN",
    
    # 能源公司
    "埃克森美孚": "XOM",
    "exxon": "XOM",
    "雪佛龙": "CVX",
    "chevron": "CVX",
    "壳牌": "RDS.A",
    "shell": "RDS.A",
    
    # 房地产/建筑
    "美国家居": "HD",
    "home depot": "HD",
    "劳氏": "LOW",
    "lowes": "LOW",
}

# A股常见股票映射表 (Shanghai Stock Exchange - SH & Shenzhen Stock Exchange - SZ)
COMMON_CN_STOCKS = {
    # 科技公司
    "阿里巴巴": "600690.SS",
    "alibaba": "600690.SS",
    "腾讯": "0700.HK",  # 腾讯在香港上市
    "tencent": "0700.HK",
    "华为": "SZ002502",  # 华为未上市，使用代理公司
    "huawei": "SZ002502",
    "大疆": "SZ300023",
    "dji": "SZ300023",
    "中兴通讯": "SZ000063",
    "zte": "SZ000063",
    "中芯国际": "SH688981",
    "smic": "SH688981",
    "百度": "BIDU",  # 在美上市
    "baidu": "BIDU",
    
    # 金融公司
    "中国平安": "SZ000001",
    "pingan": "SZ000001",
    "工商银行": "SH601398",
    "icbc": "SH601398",
    "建设银行": "SH601939",
    "ccb": "SH601939",
    "农业银行": "SH601288",
    "abc": "SH601288",
    "中国银行": "SH601988",
    "boc": "SH601988",
    "招商银行": "SH600036",
    "cmc": "SH600036",
    "浦发银行": "SH600000",
    "spdb": "SH600000",
    
    # 消费品公司
    "贵州茅台": "SH600519",
    "moutai": "SH600519",
    "五粮液": "SZ000858",
    "wuliangye": "SZ000858",
    "伊利股份": "SH600887",
    "yili": "SH600887",
    "美的集团": "SZ000333",
    "midea": "SZ000333",
    "格力电器": "SZ000651",
    "gree": "SZ000651",
    "万科": "SZ000002",
    "vanke": "SZ000002",
    
    # 汽车
    "比亚迪": "SZ001898",
    "byd": "SZ001898",
    "长城汽车": "SH601633",
    "gwm": "SH601633",
    "吉利汽车": "0175.HK",  # 在香港上市
    "geely": "0175.HK",
    
    # 能源
    "中国石油": "SH601857",
    "cnpc": "SH601857",
    "中国石化": "SH600028",
    "sinopec": "SH600028",
    "中国神华": "SH601088",
    "csha": "SH601088",
}

# 港股常见股票映射表 (Hong Kong Stock Exchange)
COMMON_HK_STOCKS = {
    # 科技公司
    "腾讯": "0700.HK",
    "tencent": "0700.HK",
    "阿里巴巴": "9988.HK",
    "alibaba": "9988.HK",
    "百度": "9888.HK",
    "baidu": "9888.HK",
    "小米": "1810.HK",
    "xiaomi": "1810.HK",
    "网易": "9999.HK",
    "netease": "9999.HK",
    "京东": "9618.HK",
    "jd": "9618.HK",
    "比亚迪": "1211.HK",
    "byd": "1211.HK",
    
    # 金融公司
    "中国平安": "2318.HK",
    "pingan": "2318.HK",
    "工商银行": "1398.HK",
    "icbc": "1398.HK",
    "建设银行": "0939.HK",
    "ccb": "0939.HK",
    "农业银行": "1288.HK",
    "abc": "1288.HK",
    "中国银行": "3988.HK",
    "boc": "3988.HK",
    "招商银行": "3968.HK",
    "cmc": "3968.HK",
    "汇丰控股": "0005.HK",
    "hsbc": "0005.HK",
    
    # 消费品公司
    "贵州茅台": "6001.HK",  # 等待上市
    "五粮液": "6002.HK",    # 等待上市
    "美团": "3690.HK",
    "meituan": "3690.HK",
    "快手": "1024.HK",
    "kuaishou": "1024.HK",
    "虎牙": "1207.HK",
    "huya": "1207.HK",
    
    # 地产公司
    "恒大": "3333.HK",
    "evergrande": "3333.HK",
    "碧桂园": "2007.HK",
    "country garden": "2007.HK",
    "中国海外": "0688.HK",
    "cohl": "0688.HK",
}

# 合并所有股票映射表（用于向后兼容）
COMMON_STOCKS = {
    **COMMON_US_STOCKS,
    **COMMON_CN_STOCKS,
    **COMMON_HK_STOCKS,
}


def _is_likely_symbol(value: str) -> bool:
    """Return True when the query looks like a ticker instead of a company name."""
    normalized = value.strip().upper()
    if not normalized:
        return False
    if not normalized.isascii():
        return False
    if normalized.endswith(".HK"):
        return normalized[:-3].isdigit()
    if normalized.startswith(("SH", "SZ")):
        return normalized[2:].isdigit()
    return normalized.replace(".", "").replace("-", "").isalnum()


def _normalize_a_share_symbol(code: str, market_type: str = "") -> Optional[str]:
    """Convert Eastmoney A-share codes into the app's SH/SZ-prefixed format."""
    code = str(code or "").strip().upper()
    market_type = str(market_type or "").strip()
    if not code or not code.isdigit():
        return None
    if market_type == "1" or code.startswith("6"):
        return f"SH{code}"
    if market_type == "0" or code.startswith(("0", "2", "3")):
        return f"SZ{code}"
    return None


def _fetch_eastmoney_a_share_suggestions(query: str, limit: int = 8) -> List[Dict]:
    """Search A-share names through Eastmoney's free suggestion endpoint."""
    try:
        response = requests.get(
            EASTMONEY_SUGGEST_URL,
            params={
                "input": query,
                "type": "14",
                "token": EASTMONEY_SUGGEST_TOKEN,
                "count": str(limit),
            },
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    rows = payload.get("QuotationCodeTable", {}).get("Data") or []
    results = []
    for row in rows[:limit]:
        if row.get("Classify") != "AStock":
            continue
        symbol = _normalize_a_share_symbol(row.get("Code"), row.get("MarketType"))
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "name": row.get("Name") or symbol,
            "market": MARKET_CN,
            "sector": "N/A",
            "country": "China",
            "type": "remote_name_match",
        })
    return results

class StockSearcher:
    """股票搜索引擎 - 支持美股、A股、港股"""
    
    def __init__(self):
        """初始化搜索引擎"""
        self.stock_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """加载本地缓存（如果存在）"""
        cache_file = "stock_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.stock_cache = json.load(f)
            except:
                self.stock_cache = {}
    
    def _save_cache(self):
        """保存缓存到本地"""
        try:
            with open("stock_cache.json", 'w', encoding='utf-8') as f:
                json.dump(self.stock_cache, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def _detect_market(self, symbol: str) -> str:
        """
        检测股票代码属于哪个市场
        
        Args:
            symbol: 股票代码
            
        Returns:
            市场标识 (US, CN, HK)
        """
        symbol = symbol.upper()
        
        # 港股标识: .HK 结尾
        if symbol.endswith('.HK'):
            return MARKET_HK
        
        # A股标识: SH 开头（上海）或 SZ 开头（深圳）
        if symbol.startswith('SH') or symbol.startswith('SZ'):
            return MARKET_CN
        
        # 美股（默认）
        return MARKET_US
    
    def _get_market_stocks_dict(self, market: str) -> Dict[str, str]:
        """获取指定市场的股票映射表"""
        if market == MARKET_CN:
            return COMMON_CN_STOCKS
        elif market == MARKET_HK:
            return COMMON_HK_STOCKS
        else:
            return COMMON_US_STOCKS
    
    def _normalize_symbol(self, symbol: str, market: Optional[str] = None) -> Tuple[str, str]:
        """
        规范化股票代码
        
        Args:
            symbol: 原始股票代码
            market: 指定市场（可选）
            
        Returns:
            (规范化的代码, 市场标识)
        """
        symbol = symbol.strip().upper()
        
        if market is None:
            market = self._detect_market(symbol)
        
        # A股标准化：添加后缀
        if market == MARKET_CN:
            if not (symbol.startswith('SH') or symbol.startswith('SZ')):
                # 如果是纯数字，根据首位数字判断交易所
                if symbol.isdigit():
                    if symbol.startswith('6'):
                        symbol = 'SH' + symbol
                    else:
                        symbol = 'SZ' + symbol
        
        return symbol, market
    
    def search_by_name(self, query: str, market: Optional[str] = None) -> List[Dict]:
        """
        通过公司名字搜索股票（支持多个市场）
        
        Args:
            query: 公司名字或代码（可以是中文或英文）
            market: 指定市场搜索 (US, CN, HK)，如果为None则搜索所有市场
        
        Returns:
            搜索结果列表，包含股票代码、公司名字、市场、交易所等信息
        """
        query = query.strip()
        results = []
        
        # 如果指定了市场，只在该市场搜索；否则在所有市场搜索
        markets = [market] if market else [MARKET_US, MARKET_CN, MARKET_HK]
        
        for target_market in markets:
            # 先尝试直接匹配（如果已经是股票代码）
            if _is_likely_symbol(query):
                try:
                    normalized_symbol, detected_market = self._normalize_symbol(query, target_market)
                    
                    if detected_market == target_market:
                        ticker = yf.Ticker(normalized_symbol)
                        info = ticker.info
                        if info and 'longName' in info:
                            results.append({
                                'symbol': normalized_symbol,
                                'name': info.get('longName', normalized_symbol),
                                'market': target_market,
                                'sector': info.get('sector', 'N/A'),
                                'country': self._get_country_by_market(target_market),
                                'type': 'direct_match'
                            })
                            if market:  # 如果指定了市场，找到了就直接返回
                                return results
                except:
                    pass
            
            # 检查该市场的常见股票映射表
            stocks_dict = self._get_market_stocks_dict(target_market)
            query_lower = query.lower()
            
            for name, symbol in stocks_dict.items():
                if name.lower() in query_lower or query_lower in name.lower():
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        results.append({
                            'symbol': symbol,
                            'name': info.get('longName', symbol),
                            'market': target_market,
                            'sector': info.get('sector', 'N/A'),
                            'country': self._get_country_by_market(target_market),
                            'type': 'common_match'
                        })
                    except:
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'market': target_market,
                            'sector': 'N/A',
                            'country': self._get_country_by_market(target_market),
                            'type': 'common_match'
                        })

            if target_market == MARKET_CN:
                results.extend(_fetch_eastmoney_a_share_suggestions(query))
        
        if results:
            # 去重
            unique_results = []
            seen_symbols = set()
            for result in results:
                if result['symbol'] not in seen_symbols:
                    unique_results.append(result)
                    seen_symbols.add(result['symbol'])
            return unique_results
        
        # 4. 使用模糊匹配（仅在没有明确指定市场或搜索所有市场时）
        if not market:
            best_matches = []
            all_names = list(COMMON_STOCKS.keys())
            
            for name in all_names:
                if HAS_FUZZYWUZZY:
                    score = fuzz.token_set_ratio(query_lower, name.lower())
                else:
                    score = simple_similarity(query_lower, name)
                
                if score > 60:  # 只返回相似度大于 60% 的结果
                    best_matches.append((name, score))
            
            # 按相似度排序
            best_matches.sort(key=lambda x: x[1], reverse=True)
            
            for name, score in best_matches[:5]:  # 最多返回 5 个结果
                # 找到该名称所属的市场
                symbol = COMMON_STOCKS[name]
                detected_market = self._detect_market(symbol)
                
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    results.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'market': detected_market,
                        'sector': info.get('sector', 'N/A'),
                        'country': self._get_country_by_market(detected_market),
                        'type': 'fuzzy_match',
                        'confidence': int(score)
                    })
                except:
                    results.append({
                        'symbol': symbol,
                        'name': name,
                        'market': detected_market,
                        'sector': 'N/A',
                        'country': self._get_country_by_market(detected_market),
                        'type': 'fuzzy_match',
                        'confidence': int(score)
                    })
        
        return results
    
    def _get_country_by_market(self, market: str) -> str:
        """根据市场获取国家代码"""
        market_to_country = {
            MARKET_US: "USA",
            MARKET_CN: "China",
            MARKET_HK: "Hong Kong",
        }
        return market_to_country.get(market, "Unknown")
    
    def get_stock_info(self, symbol: str, market: Optional[str] = None) -> Dict:
        """
        获取股票详细信息
        
        Args:
            symbol: 股票代码
            market: 指定市场（可选）
        
        Returns:
            股票信息字典
        """
        try:
            if market is None:
                market = self._detect_market(symbol)
            
            normalized_symbol, _ = self._normalize_symbol(symbol, market)
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.info
            
            return {
                'symbol': normalized_symbol,
                'name': info.get('longName', normalized_symbol),
                'market': market,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': self._get_country_by_market(market),
                'currency': info.get('currency', self._get_currency_by_market(market)),
                'description': info.get('longBusinessSummary', 'N/A')[:200],  # 限制长度
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'name': 'Unknown',
                'market': market or 'Unknown',
                'error': f'Failed to fetch stock info: {str(e)}'
            }
    
    def _get_currency_by_market(self, market: str) -> str:
        """根据市场获取货币"""
        currency_map = {
            MARKET_US: "USD",
            MARKET_CN: "CNY",
            MARKET_HK: "HKD",
        }
        return currency_map.get(market, "Unknown")

def search_stocks(query: str, market: Optional[str] = None) -> List[Dict]:
    """便利函数 - 搜索股票（支持多个市场）"""
    searcher = StockSearcher()
    return searcher.search_by_name(query, market)

def get_stock_info(symbol: str, market: Optional[str] = None) -> Dict:
    """便利函数 - 获取股票信息"""
    searcher = StockSearcher()
    return searcher.get_stock_info(symbol, market)

def search_us_stocks(query: str) -> List[Dict]:
    """便利函数 - 搜索美股"""
    return search_stocks(query, MARKET_US)

def search_cn_stocks(query: str) -> List[Dict]:
    """便利函数 - 搜索A股"""
    return search_stocks(query, MARKET_CN)

def search_hk_stocks(query: str) -> List[Dict]:
    """便利函数 - 搜索港股"""
    return search_stocks(query, MARKET_HK)

if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("股票搜索测试 - 支持美股、A股、港股")
    print("=" * 60)
    
    # 测试美股
    print("\n【美股搜索】")
    us_queries = ["苹果", "微软", "特斯拉", "nvidia"]
    for query in us_queries:
        print(f"\n搜索: {query}")
        results = search_us_stocks(query)
        for result in results[:2]:  # 只显示前2个结果
            market_name = "美股" if result.get('market') == MARKET_US else result.get('market', 'N/A')
            print(f"  ✅ {result['symbol']:10s} - {result['name'][:30]:30s} [{market_name}]")
    
    # 测试A股
    print("\n\n【A股搜索】")
    cn_queries = ["阿里巴巴", "贵州茅台", "中国平安", "工商银行"]
    for query in cn_queries:
        print(f"\n搜索: {query}")
        results = search_cn_stocks(query)
        for result in results[:2]:  # 只显示前2个结果
            market_name = "A股" if result.get('market') == MARKET_CN else result.get('market', 'N/A')
            print(f"  ✅ {result['symbol']:10s} - {result['name'][:30]:30s} [{market_name}]")
    
    # 测试港股
    print("\n\n【港股搜索】")
    hk_queries = ["腾讯", "阿里巴巴", "小米", "美团"]
    for query in hk_queries:
        print(f"\n搜索: {query}")
        results = search_hk_stocks(query)
        for result in results[:2]:  # 只显示前2个结果
            market_name = "港股" if result.get('market') == MARKET_HK else result.get('market', 'N/A')
            print(f"  ✅ {result['symbol']:10s} - {result['name'][:30]:30s} [{market_name}]")
    
    # 测试跨市场搜索（搜索所有市场）
    print("\n\n【跨市场搜索 - 同时搜索多个市场】")
    cross_queries = ["阿里巴巴", "腾讯"]
    for query in cross_queries:
        print(f"\n搜索: {query} (所有市场)")
        results = search_stocks(query)
        for result in results[:3]:  # 显示前3个结果
            market_map = {MARKET_US: "美股", MARKET_CN: "A股", MARKET_HK: "港股"}
            market_name = market_map.get(result.get('market'), result.get('market', 'N/A'))
            print(f"  ✅ {result['symbol']:10s} - {result['name'][:30]:30s} [{market_name}]")

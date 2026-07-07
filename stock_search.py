#!/usr/bin/env python3
"""
股票代码搜索模块 - 通过公司名字查找股票代码
当前主路径仅支持 A 股：
1. A 股代码直接识别并规范化
2. A 股中文公司名通过本地映射和东方财富建议接口查询
3. 美股、港股和加密货币不作为回测对象支持
"""

import yfinance as yf
import pandas as pd
import os
import json
import requests
from typing import List, Dict, Tuple, Optional

from tradable_universe import validate_tradable_symbol

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


def _is_a_share_code(value: str) -> bool:
    normalized = value.strip().upper()
    if normalized.startswith(("SH", "SZ", "BJ")) and normalized[2:].isdigit() and len(normalized[2:]) == 6:
        return True
    if normalized.endswith((".SH", ".SZ", ".BJ")):
        code = normalized.split(".")[0]
        return code.isdigit() and len(code) == 6
    return normalized.isdigit() and len(normalized) == 6


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
    """股票搜索引擎 - 仅支持 A 股主路径"""
    
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
            if symbol.endswith((".SH", ".SZ", ".BJ")):
                code, suffix = symbol.split(".")
                symbol = f"{suffix}{code}"
            elif not symbol.startswith(('SH', 'SZ', 'BJ')):
                # 如果是纯数字，根据首位数字判断交易所
                if symbol.isdigit():
                    if symbol.startswith('6'):
                        symbol = 'SH' + symbol
                    elif symbol.startswith('8'):
                        symbol = 'BJ' + symbol
                    else:
                        symbol = 'SZ' + symbol
        
        return symbol, market
    
    def search_by_name(
        self,
        query: str,
        market: Optional[str] = None,
        portfolio_mode: bool = False,
    ) -> List[Dict]:
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
        
        if market and market != MARKET_CN:
            return []

        markets = [MARKET_CN]
        
        for target_market in markets:
            if _is_a_share_code(query):
                normalized_symbol, _ = self._normalize_symbol(query, target_market)
                results.append({
                    'symbol': normalized_symbol,
                    'name': normalized_symbol,
                    'market': target_market,
                    'sector': 'N/A',
                    'country': self._get_country_by_market(target_market),
                    'type': 'direct_match'
                })
            
            # 检查该市场的常见股票映射表
            stocks_dict = self._get_market_stocks_dict(target_market)
            query_lower = query.lower()
            
            for name, symbol in stocks_dict.items():
                if name.lower() in query_lower or query_lower in name.lower():
                    if not _is_a_share_code(symbol):
                        continue
                    normalized_symbol, _ = self._normalize_symbol(symbol, target_market)
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        results.append({
                            'symbol': normalized_symbol,
                            'name': info.get('longName', normalized_symbol),
                            'market': target_market,
                            'sector': info.get('sector', 'N/A'),
                            'country': self._get_country_by_market(target_market),
                            'type': 'common_match'
                        })
                    except:
                        results.append({
                            'symbol': normalized_symbol,
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
            return _annotate_portfolio_tradability(unique_results) if portfolio_mode else unique_results
        
        # 4. 使用模糊匹配（仅限 A 股本地映射表）
        if not market or market == MARKET_CN:
            best_matches = []
            all_names = list(COMMON_CN_STOCKS.keys())
            
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
                if not _is_a_share_code(symbol):
                    continue
                normalized_symbol, _ = self._normalize_symbol(symbol, MARKET_CN)
                detected_market = self._detect_market(symbol)
                
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    results.append({
                        'symbol': normalized_symbol,
                        'name': info.get('longName', normalized_symbol),
                        'market': MARKET_CN,
                        'sector': info.get('sector', 'N/A'),
                        'country': self._get_country_by_market(MARKET_CN),
                        'type': 'fuzzy_match',
                        'confidence': int(score)
                    })
                except:
                    results.append({
                        'symbol': normalized_symbol,
                        'name': name,
                        'market': MARKET_CN,
                        'sector': 'N/A',
                        'country': self._get_country_by_market(MARKET_CN),
                        'type': 'fuzzy_match',
                        'confidence': int(score)
                    })
        
        return _annotate_portfolio_tradability(results) if portfolio_mode else results
    
    def _get_country_by_market(self, market: str) -> str:
        """根据市场获取国家代码"""
        market_to_country = {
            MARKET_US: "USA",
            MARKET_CN: "China",
            MARKET_HK: "Hong Kong",
        }
        return market_to_country.get(market, "Unknown")
    
def _annotate_portfolio_tradability(results: List[Dict]) -> List[Dict]:
    annotated = []
    for result in results:
        validation = validate_tradable_symbol(result.get("symbol", ""))
        annotated.append({
            **result,
            "tradable": validation.ok,
            "tradable_reason": validation.reason,
        })
    return annotated


def search_stocks(
    query: str,
    market: Optional[str] = None,
    portfolio_mode: bool = False,
) -> List[Dict]:
    """便利函数 - 搜索 A 股"""
    searcher = StockSearcher()
    return searcher.search_by_name(query, market, portfolio_mode=portfolio_mode)

def search_us_stocks(query: str) -> List[Dict]:
    """兼容旧接口：当前不再支持美股搜索。"""
    return search_stocks(query, MARKET_US)

def search_cn_stocks(query: str) -> List[Dict]:
    """便利函数 - 搜索A股"""
    return search_stocks(query, MARKET_CN)

def search_hk_stocks(query: str) -> List[Dict]:
    """兼容旧接口：当前不再支持港股搜索。"""
    return search_stocks(query, MARKET_HK)

if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("股票搜索测试 - 仅支持 A 股")
    print("=" * 60)

    print("\n\n【A股搜索】")
    cn_queries = ["阿里巴巴", "贵州茅台", "中国平安", "工商银行"]
    for query in cn_queries:
        print(f"\n搜索: {query}")
        results = search_cn_stocks(query)
        for result in results[:2]:  # 只显示前2个结果
            market_name = "A股" if result.get('market') == MARKET_CN else result.get('market', 'N/A')
            print(f"  ✅ {result['symbol']:10s} - {result['name'][:30]:30s} [{market_name}]")

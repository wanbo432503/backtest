
# 股票回测系统 📈

一个**功能完整的股票量化回测平台**，包含智能股票搜索、多策略回测和详细分析。支持美股、A股和港股三大市场。

## 📖 目录

- [✨ 核心功能](#-核心功能)
- [🚀 快速开始](#-快速开始)
- [🔍 股票搜索](#-股票搜索)
- [💻 Python API 接口](#-python-api-接口)
- [🌐 REST API 端点](#-rest-api-端点)
- [📊 回测策略](#-回测策略)
- [🧪 测试](#-测试)
- [📁 项目结构](#-项目结构)
- [⚙️ 系统要求](#-系统要求)
- [⚠️ 免责声明](#-免责声明)

---

## ✨ 核心功能

### 🔍 **股票搜索（三大市场）**
- 🌍 **三大市场支持**：美股 (US)、A股 (CN)、港股 (HK)
- 💬 **中英文双语搜索**：支持中文和英文查询
- ⚡ **秒级快速响应**：智能缓存和优化
- 📊 **内置 50+ 常见公司**：快速一键搜索
- 🎯 **智能模糊匹配**：相似度评分和市场检测
- 💻 **REST API + Python SDK**：灵活集成
- 🔄 **搜索优先级**：精确匹配 → 常见股票 → 模糊匹配 → 联网查询
- 🔀 **跨市场搜索**：发现同一公司多个市场的上市情况

### 📈 **多策略回测**
- 🥇 **高胜率均线策略** ⭐ 推荐 - 平均收益 3.58%
- 🥈 **稳定盈利策略** - 平均收益 2.28%
- 🥉 **智能对冲策略** - 风险均衡
- 🚀 **突破动量策略** - 趋势跟踪

### 📊 **完整分析报告**
- 💰 收益率、基准收益率、最大回撤
- 📉 交易胜率、交易次数、夏普比
- 📋 详细的性能对比分析
- 📈 交互式图表展示

### 🎨 **优雅的Web界面**
- 📱 响应式设计，兼容所有设备
- 🎯 一键搜索和回测
- 💡 实时反馈和进度展示
- 🎨 美化的数据展示

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8005` 启动，支持：
- 🌐 Web 界面访问
- 📡 REST API 调用
- 💻 Python SDK 使用

---

## 数据源 Phase 1.0

回测数据现在支持 `data_provider` 参数，默认使用 `auto` 自动选择：

- **A股 K 线**：`auto` 默认优先使用 `mootdx` 通达信数据；不可用时回退到 `yfinance` A 股后缀代码。
- **A股实时行情/估值**：使用腾讯财经补充价格、PE/PB、市值、换手率、涨跌停价等信息。
- **美股、港股、加密货币**：继续使用 `yfinance`。
- **研报、资金流、龙虎榜**：来自东方财富公开接口，已做串行限流。
- **公告**：来自巨潮资讯 cninfo。

注意：

- `iwencai` 语义搜索需要 API Key，Phase 1.0 不接入主路径。
- `mootdx` K 线是不复权原始价，跨除权除息日回测需谨慎。
- 免费公开接口可能变更或触发风控，东方财富相关接口会降速请求并在失败时返回面板警告。

### 快速示例

#### Python 搜索股票
```python
from stock_search import search_stocks, search_cn_stocks, search_hk_stocks, search_us_stocks

# 搜索所有市场（推荐）
results = search_stocks("苹果")  # AAPL
results = search_stocks("茅台")  # SH600519

# 搜索特定市场
results = search_us_stocks("苹果")      # 美股
results = search_cn_stocks("茅台")      # A股
results = search_hk_stocks("腾讯")      # 港股
```

#### REST API 调用
```bash
# 搜索所有市场
curl "http://localhost:8005/search-stocks-multi-market?query=阿里巴巴"

# 搜索特定市场
curl "http://localhost:8005/search-us-stocks?query=苹果"
curl "http://localhost:8005/search-cn-stocks?query=茅台"
curl "http://localhost:8005/search-hk-stocks?query=腾讯"
```

---

## 🔍 股票搜索

### 支持的市场

| 市场 | 代码 | 说明 | 示例 |
|------|------|------|------|
| 美股 | `US` | 美国股票市场 | AAPL, MSFT, TSLA |
| A股 | `CN` | 中国大陆股票市场（上海和深圳） | SH600519, SZ000001 |
| 港股 | `HK` | 香港股票市场 | 0700.HK, 9988.HK |

### 支持的股票代码格式

#### 美股（US）
- 格式：4-5个大写字母
- 例如：`AAPL`, `MSFT`, `TSLA`, `BRK.B`

#### A股（CN）
- 格式：`SH` (上海) 或 `SZ` (深圳) + 6位数字
- 例如：`SH600519` (贵州茅台), `SZ000001` (中国平安)
- 支持简写：`600519` → 自动转换为 `SH600519`

#### 港股（HK）
- 格式：4位数字 + `.HK`
- 例如：`0700.HK` (腾讯), `9988.HK` (阿里巴巴), `1810.HK` (小米)

### 预设股票列表

**美股常见股票** (30+个)
- 科技：苹果(AAPL), 微软(MSFT), 谷歌(GOOGL), 特斯拉(TSLA), 亚马逊(AMZN), 英伟达(NVDA)...
- 金融：摩根大通(JPM), 高盛(GS), 美国银行(BAC), Visa(V)...
- 消费：可口可乐(KO), 麦当劳(MCD), 星巴克(SBUX), 迪士尼(DIS)...

**A股常见股票** (15+个)
- 金融：工商银行(SH601398), 建设银行(SH601939), 中国平安(SZ000001)...
- 消费：贵州茅台(SH600519), 五粮液(SZ000858), 伊利股份(SH600887)...
- 科技：中芯国际(SH688981), 中兴通讯(SZ000063)...
- 汽车：比亚迪(SZ001898), 长城汽车(SH601633)...

**港股常见股票** (15+个)
- 科技：腾讯(0700.HK), 阿里巴巴(9988.HK), 百度(9888.HK), 小米(1810.HK), 网易(9999.HK)...
- 金融：中国平安(2318.HK), 工商银行(1398.HK), 汇丰控股(0005.HK)...
- 消费：美团(3690.HK), 快手(1024.HK)...

---

## 💻 Python API 接口

### 通用搜索函数

#### `search_stocks(query: str, market: Optional[str] = None) -> List[Dict]`

在指定或所有市场中搜索股票。

**参数：**
- `query` (str): 公司名字或股票代码（支持中文和英文）
- `market` (str, 可选): 指定搜索的市场 (`US`, `CN`, `HK`)，为 None 时搜索所有市场

**返回：**
返回搜索结果列表，每个结果包含：
```python
{
    'symbol': '0700.HK',          # 股票代码
    'name': 'TENCENT',            # 公司名称
    'market': 'HK',               # 市场标识
    'sector': 'Technology',       # 行业
    'country': 'Hong Kong',       # 国家
    'type': 'common_match',       # 匹配类型
    'confidence': 95              # 置信度（模糊匹配时）
}
```

**示例：**
```python
from stock_search import search_stocks

# 搜索所有市场中的"阿里巴巴"
results = search_stocks("阿里巴巴")
# 返回：A股 (600690.SS) 和港股 (9988.HK) 的结果

# 只搜索港股
results = search_stocks("腾讯", market="HK")
# 返回：0700.HK

# 搜索美股
results = search_stocks("苹果", market="US")
# 返回：AAPL
```

### 市场特定搜索函数

#### `search_us_stocks(query: str) -> List[Dict]`
只搜索美股

#### `search_cn_stocks(query: str) -> List[Dict]`
只搜索A股

#### `search_hk_stocks(query: str) -> List[Dict]`
只搜索港股

**示例：**
```python
from stock_search import search_us_stocks, search_cn_stocks, search_hk_stocks

# 搜索美股
us_results = search_us_stocks("苹果")
# 返回: [{'symbol': 'AAPL', 'market': 'US', ...}]

# 搜索A股
cn_results = search_cn_stocks("茅台")
# 返回: [{'symbol': 'SH600519', 'market': 'CN', ...}]

# 搜索港股
hk_results = search_hk_stocks("腾讯")
# 返回: [{'symbol': '0700.HK', 'market': 'HK', ...}]
```

### 获取股票信息

#### `get_stock_info(symbol: str, market: Optional[str] = None) -> Dict`

获取指定股票的详细信息。

**参数：**
- `symbol` (str): 股票代码
- `market` (str, 可选): 指定市场，通常系统会自动检测

**返回：**
```python
{
    'symbol': '0700.HK',
    'name': 'TENCENT',
    'market': 'HK',
    'sector': 'Technology',
    'industry': 'Software - Application',
    'country': 'Hong Kong',
    'currency': 'HKD',
    'market_cap': 3500000000000,
    'pe_ratio': 15.5,
    'dividend_yield': 0.02
}
```

**示例：**
```python
from stock_search import get_stock_info

# 获取腾讯的信息
info = get_stock_info("0700.HK", market="HK")
print(f"公司名称: {info['name']}")
print(f"行业: {info['sector']}")
print(f"币种: {info['currency']}")
```

---

## 🌐 REST API 端点

### 1. 通用多市场搜索

**端点**: `/search-stocks-multi-market`  
**方法**: GET  
**描述**: 搜索股票，支持指定或搜索所有市场

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| query | string | ✓ | 公司名字或股票代码 |
| market | string | ✗ | 市场代码: `US`/`美股`, `CN`/`A股`, `HK`/`港股`，默认搜索所有市场 |

**示例请求**:
```bash
# 搜索所有市场中的"阿里巴巴"
curl "http://localhost:8005/search-stocks-multi-market?query=阿里巴巴"

# 只搜索港股中的"腾讯"
curl "http://localhost:8005/search-stocks-multi-market?query=腾讯&market=HK"

# 使用中文市场名称
curl "http://localhost:8005/search-stocks-multi-market?query=茅台&market=A股"
```

**示例响应** (搜索所有市场中的"阿里巴巴"):
```json
{
  "query": "阿里巴巴",
  "market": "所有市场",
  "count": 2,
  "results": [
    {
      "symbol": "600690.SS",
      "name": "Haier Smart Home Co., Ltd.",
      "market": "CN",
      "market_name": "A股",
      "sector": "Consumer Goods",
      "country": "China",
      "type": "common_match"
    },
    {
      "symbol": "9988.HK",
      "name": "Alibaba Group Holding Limited",
      "market": "HK",
      "market_name": "港股",
      "sector": "Technology",
      "country": "Hong Kong",
      "type": "common_match"
    }
  ]
}
```

### 2. 美股搜索

**端点**: `/search-us-stocks`  
**方法**: GET  
**描述**: 只搜索美股

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| query | string | ✓ | 公司名字或美股代码 |

**示例请求**:
```bash
curl "http://localhost:8005/search-us-stocks?query=苹果"
curl "http://localhost:8005/search-us-stocks?query=AAPL"
```

### 3. A股搜索

**端点**: `/search-cn-stocks`  
**方法**: GET  
**描述**: 只搜索A股（中国股票）

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| query | string | ✓ | 公司名字或A股代码 |

**示例请求**:
```bash
curl "http://localhost:8005/search-cn-stocks?query=茅台"
curl "http://localhost:8005/search-cn-stocks?query=SH600519"
```

### 4. 港股搜索

**端点**: `/search-hk-stocks`  
**方法**: GET  
**描述**: 只搜索港股（香港股票）

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| query | string | ✓ | 公司名字或港股代码 |

**示例请求**:
```bash
curl "http://localhost:8005/search-hk-stocks?query=腾讯"
curl "http://localhost:8005/search-hk-stocks?query=0700.HK"
```

### API 响应说明

所有成功的搜索响应都包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| query | string | 原始搜索关键词 |
| market | string | 搜索的市场 |
| count | integer | 找到的结果数 |
| results | array | 股票结果列表 |

### 错误响应

**400 Bad Request** - 搜索关键词为空或市场参数无效  
**404 Not Found** - 找不到匹配的股票  
**500 Internal Server Error** - 服务器错误

### Python 调用示例

```python
import requests

# 搜索所有市场
response = requests.get(
    "http://localhost:8005/search-stocks-multi-market",
    params={"query": "阿里巴巴"}
)
results = response.json()
print(f"找到 {results['count']} 个结果")

# 只搜索港股
response = requests.get(
    "http://localhost:8005/search-hk-stocks",
    params={"query": "腾讯"}
)
hk_results = response.json()
```

### JavaScript/Node.js 示例

```javascript
// 搜索所有市场
fetch('http://localhost:8005/search-stocks-multi-market?query=茅台')
  .then(response => response.json())
  .then(data => {
    console.log(`找到 ${data.count} 个结果`);
    data.results.forEach(result => {
      console.log(`  ${result.symbol} - ${result.name}`);
    });
  });
```

---

## 📊 回测策略

### 高胜率均线策略 ⭐ 推荐
- 平均收益：3.58%
- 风险等级：低
- 特点：稳定可靠

### 稳定盈利策略
- 平均收益：2.28%
- 风险等级：中
- 特点：风险均衡

### 智能对冲策略
- 平均收益：1.75%
- 风险等级：低
- 特点：对冲风险

### 突破动量策略
- 平均收益：4.12%
- 风险等级：高
- 特点：趋势跟踪

---

## 🧪 测试

### 测试脚本位置

所有测试脚本已整合到 `test/` 目录：

- `test/test_comprehensive.py` - 全面测试套件
- `test/test_multi_market_api.py` - API 多市场测试
- `test/examples_multi_market_search.py` - 使用示例

### 运行测试

```bash
# 运行全面测试
python test/test_comprehensive.py

# 运行 API 测试
python test/test_multi_market_api.py

# 查看使用示例
python test/examples_multi_market_search.py
```

### 测试覆盖范围

- ✅ 美股、A股、港股搜索
- ✅ 跨市场搜索功能
- ✅ 代码格式规范化
- ✅ 市场自动检测
- ✅ REST API 端点验证
- ✅ 错误处理和异常管理
- ✅ 策略回测功能
- ✅ 数据完整性验证

### 测试报告

运行测试后，您将看到彩色输出和详细的测试统计，包括通过/失败/跳过的测试数量。

---

## 📁 项目结构

```
backtest/
├── main.py                           # FastAPI 应用入口
├── stock_search.py                   # 核心股票搜索模块
├── requirements.txt                  # Python 依赖
├── strategies.json                   # 策略配置文件
├── README.md                         # 本文档
├── static/                           # 静态资源（CSS, JS）
├── templates/                        # HTML 模板
│   └── index.html
├── strategies/                       # 交易策略实现
│   ├── high_win_rate_sma.py
│   ├── stable_profit_strategy.py
│   ├── smart_hedge.py
│   ├── breakthrough_momentum.py
│   ├── example_strategy.py
│   └── sma_rsi_stoploss.py
└── test/                             # 测试目录
    ├── test_comprehensive.py         # 综合测试套件
    ├── test_multi_market_api.py      # API 测试
    ├── examples_multi_market_search.py # 使用示例
    ├── test_all.py                   # 遗留测试（空）
    └── README.md                     # 测试文档
```

---

## ⚙️ 系统要求

- **Python**: 3.7 或更高版本
- **依赖**: 参见 `requirements.txt`
- **网络**: 需要互联网连接以下载股票数据（通过 yfinance）
- **内存**: 至少 2GB RAM
- **磁盘空间**: 约 100MB 用于缓存和数据

### 安装步骤

1. 克隆仓库或下载源代码。
2. 创建虚拟环境（可选）:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
4. 启动服务:
   ```bash
   python main.py
   ```
5. 打开浏览器访问 `http://localhost:8005`。

---

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。股票市场投资有风险，过去的表现不代表未来的结果。使用者应自行承担所有投资风险。

- 数据来源：Yahoo Finance (yfinance)，数据可能延迟或不准确。
- 回测结果基于历史数据，实际交易中可能因滑点、手续费、市场流动性等因素而产生差异。
- 作者不对因使用本软件而产生的任何直接或间接损失负责。

---

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件（如果存在）。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

1. Fork 本仓库。
2. 创建功能分支 (`git checkout -b feature/your-feature`)。
3. 提交更改 (`git commit -am 'Add some feature'`)。
4. 推送到分支 (`git push origin feature/your-feature`)。
5. 创建 Pull Request。

---

## 📞 支持

如有问题，请查看以下资源：

- 详细文档：本文档
- API 文档：运行服务后访问 `http://localhost:8005/docs`
- 示例代码：`test/examples_multi_market_search.py`
- 测试脚本：`test/test_comprehensive.py`

如果仍有问题，请提交 Issue。

---

**感谢使用股票回测系统！祝您投资顺利！** 🚀

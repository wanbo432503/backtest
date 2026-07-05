# 测试文档 🧪

本目录包含股票回测系统的所有测试脚本。

## 📊 统一测试脚本

### `test_comprehensive.py` ⭐ 推荐

统一的测试脚本，整合了所有功能测试，一次性测试系统的所有关键功能。

**功能覆盖：**
- ✅ 股票搜索功能（中文、英文、模糊匹配）
- ✅ 股票信息获取
- ✅ 策略回测（高胜率均线、稳定盈利）
- ✅ 自定义策略（布林带）
- ✅ API 接口（需要服务运行）
- ✅ 数据完整性验证

**运行方式：**

```bash
# 运行所有测试
python test/test_comprehensive.py

# 运行特定类型的测试
python test/test_comprehensive.py --search       # 股票搜索测试
python test/test_comprehensive.py --backtest     # 回测策略测试
python test/test_comprehensive.py --api          # API 接口测试
python test/test_comprehensive.py --bb           # 布林带策略测试

# 显示详细输出
python test/test_comprehensive.py --verbose
python test/test_comprehensive.py --search --verbose
```

**输出示例：**

```
======================================================================
  📊 股票回测系统 - 统一测试套件
======================================================================

======================================================================
  🔍 测试 1: 股票搜索功能
======================================================================

✅ 搜索 '苹果' (中文公司名)
✅ 搜索 '微软' (中文公司名)
✅ 搜索 'Google' (英文公司名)
...

──────────────────────────────────────────────────────────────────────
  📊 测试总结
──────────────────────────────────────────────────────────────────────

  总测试数:  28
  通过:      28
  失败:      0
  跳过:      0

🎉 所有测试通过！
```

---

## 测试详解

### 1️⃣ 股票搜索功能测试

**测试项目：**
- 中文公司名搜索（如"苹果"→ AAPL）
- 英文公司名搜索（如"Apple"→ AAPL）
- 股票代码搜索（如"TSLA"→ Tesla）
- 模糊匹配（如"特"→ TSLA）
- 股票详细信息获取

**测试数据：**
- 苹果(AAPL)、微软(MSFT)、谷歌(GOOGL)、特斯拉(TSLA)、强生(JNJ)

### 2️⃣ 策略回测功能测试

**测试策略：**
- 高胜率均线策略（三重 SMA）
- 稳定盈利策略（SMA + RSI）

**验证项目：**
- 回测结果完整性
- 关键指标可用性（收益率、最大回撤、夏普比等）
- 交易记录生成

### 3️⃣ 自定义策略测试

**测试策略：**
- 布林带交易策略（Bollinger Bands）

**验证项目：**
- 指标计算正确性
- 交易信号生成
- 回测统计

### 4️⃣ API 接口测试

**测试端点：**
- `GET /strategies` - 获取策略列表
- `GET /search-stocks?query=苹果` - 搜索股票

**前置条件：**
需要运行 FastAPI 服务：
```bash
python main.py
```

### 5️⃣ 数据完整性验证

**验证项目：**
- 策略文件存在性
- 配置文件完整性
- 内置股票映射表

---

## 快速开始

### 基础测试

运行最常用的测试：

```bash
# 在项目根目录运行
cd /Users/wanbo/Documents/workspace/backtest
python test/test_comprehensive.py
```

### 完整测试（包含 API）

```bash
# 终端 1: 启动服务
python main.py

# 终端 2: 运行完整测试
python test/test_comprehensive.py --verbose
```

### 定向测试

测试特定功能：

```bash
# 只测试搜索功能
python test/test_comprehensive.py --search

# 只测试回测功能
python test/test_comprehensive.py --backtest

# 只测试布林带策略
python test/test_comprehensive.py --bb
```

---

## 测试结果解读

### 成功输出

```
✅ 搜索 '苹果' (中文公司名)
```
表示该测试通过，系统功能正常。

### 失败输出

```
❌ 搜索 '苹果' (中文公司名) - 未找到任何结果
```
表示测试失败，需要检查相关功能或数据。

### 警告输出

```
⚠️  API 接口测试 - 服务未运行
```
表示某项可选测试被跳过（通常是因为依赖条件不满足）。

---

## 常见问题

**Q: 运行测试时出现"ModuleNotFoundError"？**

A: 确保安装了所有依赖：
```bash
pip install -r requirements.txt
```

**Q: API 测试显示"连接失败"？**

A: API 测试需要 FastAPI 服务运行。在另一个终端执行：
```bash
python main.py
```

**Q: 搜索测试失败？**

A: 检查网络连接（yfinance 需要联网）。可以尝试：
```bash
python test/test_comprehensive.py --search --verbose
```
查看详细错误信息。

**Q: 回测测试很慢？**

A: 这是正常的。首次运行需要下载数据并计算指标。后续运行会更快。

---

## 添加新测试

要添加新的测试，编辑 `test_comprehensive.py`：

1. 创建测试函数：
```python
def test_new_feature(results: TestResults, verbose: bool = False):
    """测试新功能"""
    print_header("🆕 测试: 新功能")
    
    test_name = "某项测试"
    try:
        # 测试逻辑
        result = do_something()
        
        if result:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, "失败原因")
    
    except Exception as e:
        results.add_fail(test_name, str(e))
```

2. 在 `run_all_tests()` 函数中调用：
```python
def run_all_tests(verbose: bool = False):
    results = TestResults()
    # ...
    test_new_feature(results, verbose)  # 添加这一行
    # ...
```

---

## 性能指标

| 项目 | 数值 |
|------|------|
| 搜索响应时间 | < 500ms |
| 回测速度 | ~ 1 秒/年 |
| 总测试耗时 | ~ 30-60 秒 |
| 总测试数 | 28+ |

---

## 相关文件

- `../main.py` - FastAPI 应用
- `../stock_search.py` - 搜索模块
- `../strategies/` - 策略实现
- `../requirements.txt` - 项目依赖

---

## 更新日志

### v1.0
- 创建统一测试脚本
- 整合所有功能测试
- 支持分模块测试
- 彩色输出和详细报告

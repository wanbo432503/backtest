# Phase 3.0 TradingAgents 智能分析面板实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` or equivalent task-by-task execution discipline to implement this plan. Follow TDD, keep commits small, mask secrets in all API responses/logs, and do not bundle unrelated refactors.

**Goal:** 删除当前右侧“标的信息”面板及其无效数据后端，改为在右侧区域集成 TradingAgents 分析能力，提供“分析”和“设置”两个 subtab。

**Architecture:** 保留 backtest 作为 FastAPI + Bootstrap 单页应用，新增一个薄的 TradingAgents Adapter 作为隔离层，通过 `/tradingagents/*` API 读取配置、启动分析、返回报告。TradingAgents 代码仍位于 `/Users/wanbo/knowledge/knowledge/repo/TradingAgents`，Phase 3.0 不复制其源码，不重写多智能体流程，只把它包装成 backtest 可调用的服务能力。

**Tech Stack:** FastAPI, Pydantic, Bootstrap UI, TradingAgents, python-dotenv/env 文件管理, pytest.

---

## 1. 范围和非目标

### 1.1 本期目标

- 完全移除右侧旧“标的信息”UI，包括研报、资金、龙虎榜、公告、quote strip、收起/展开按钮中旧文案和加载逻辑。
- 删除旧信息面板相关后端：
  - `/market-insights/{symbol}`
  - `market_insights.py`
  - 仅服务旧面板的 `/stock-info/{symbol}` 和 `stock_search.get_stock_info(...)` 辅助逻辑。
- 右侧区域改为 TradingAgents 面板，包含两个 subtab：
  - `分析`: 用户输入区 + 分析运行状态 + 报告输出区。
  - `设置`: OpenAI-compatible 模型配置区，字段对齐 TradingAgents 当前 `.env`。
- 分析 tab 支持从当前回测标的自动带入 A 股代码，并转换为 TradingAgents 可识别格式，例如 `SH603019 -> 603019.SS`、`SZ002241 -> 002241.SZ`。
- 设置 tab 能读取、展示、保存 TradingAgents OpenAI-compatible 相关配置；API key 只显示是否已设置，永不回传明文。
- 后端返回结构化报告字段，前端按分区展示：
  - Market / Sentiment / News / Fundamentals analyst reports
  - Research team decision
  - Trader plan
  - Risk debate
  - Portfolio manager final decision

### 1.2 非目标

- 不把 TradingAgents 改造成 backtest 内部策略或回测引擎。
- 不引入多股票组合轮动、权重优化或投资组合推荐。
- 不把旧研报/资金/龙虎榜/公告数据源迁移到 TradingAgents。
- 不在前端显示或保存用户密钥明文。
- 不支持美股、港股、加密货币作为 backtest 标的；TradingAgents 分析入口本期也默认服务当前 A 股研究流程。
- 不实现 WebSocket 实时流式报告；MVP 用请求级状态和最终报告即可，后续再扩展 SSE/WebSocket。

---

## 2. 当前代码入口

### 2.1 backtest 当前入口

- `main.py`
  - 当前 FastAPI 应用。
  - `GET /market-insights/{symbol}` 是右侧旧信息面板入口。
  - `GET /stock-info/{symbol}` 只服务旧股票详情能力，需确认并删除无用调用。
- `market_insights.py`
  - 当前旧面板数据聚合逻辑，包含研报、资金、龙虎榜、公告。
- `templates/index.html`
  - 当前 UI 主页面。
  - 旧右侧列：`#rightInsightColumn`。
  - 旧加载函数：`loadMarketInsights(symbol)`、`renderInsightList(...)`、`renderQuote(...)`。
  - 旧 CSS：`.insight-section`、`.insight-item`、`.insight-empty`、`.quote-strip`、`.insight-header`、`.insight-restore-tab` 等。
- `test/test_market_insights_api.py`
  - 旧 `/market-insights` API 测试，删除或替换为 TradingAgents API 测试。

### 2.2 TradingAgents 当前入口

- TradingAgents repo: `/Users/wanbo/knowledge/knowledge/repo/TradingAgents`
- `tradingagents/default_config.py`
  - 支持 env override：
    - `TRADINGAGENTS_LLM_PROVIDER`
    - `TRADINGAGENTS_DEEP_THINK_LLM`
    - `TRADINGAGENTS_QUICK_THINK_LLM`
    - `TRADINGAGENTS_LLM_BACKEND_URL`
    - `TRADINGAGENTS_OUTPUT_LANGUAGE`
    - `TRADINGAGENTS_MAX_DEBATE_ROUNDS`
    - `TRADINGAGENTS_MAX_RISK_ROUNDS`
    - `TRADINGAGENTS_CHECKPOINT_ENABLED`
    - `TRADINGAGENTS_TEMPERATURE`
    - `TRADINGAGENTS_OPENAI_REASONING_EFFORT`
- `tradingagents/llm_clients/openai_client.py`
  - Provider `openai_compatible` 要求 `backend_url`，API key 可选但支持 `OPENAI_COMPATIBLE_API_KEY`。
- `cli/main.py`
  - CLI 的 `run_analysis(...)` 会交互式收集配置，不适合直接从 Web 调用。
  - 其中可复用的运行思路是构建 `TradingAgentsGraph`，stream chunks，合并 `final_state`，最后读取报告字段。

---

## 3. 产品设计

### 3.1 右侧面板结构

右侧列保留三栏布局中的 `col-xl-3 col-lg-12` 宽度，但从“标的信息”改为“智能分析”。

Tab 结构：

- `分析`
  - 标的输入：默认同步左侧 `A股代码` 输入框。
  - 分析日期：默认今天，可手动选择。
  - Analyst 选择：默认 `market`、`news`、`fundamentals`，可勾选 `sentiment`。
  - 研究轮数：默认读取设置中的 `TRADINGAGENTS_MAX_DEBATE_ROUNDS` / `TRADINGAGENTS_MAX_RISK_ROUNDS`。
  - 操作按钮：开始分析、清空结果。
  - 状态区：idle/running/succeeded/failed，展示耗时和错误。
  - 报告输出区：用 Bootstrap tabs 或 accordion 分区展示 Markdown 文本。
- `设置`
  - Provider 固定优先支持 `openai_compatible`。
  - Backend URL: `TRADINGAGENTS_LLM_BACKEND_URL`。
  - API Key: `OPENAI_COMPATIBLE_API_KEY`，只显示“已设置/未设置”。
  - Deep model: `TRADINGAGENTS_DEEP_THINK_LLM`。
  - Quick model: `TRADINGAGENTS_QUICK_THINK_LLM`。
  - Output language: `TRADINGAGENTS_OUTPUT_LANGUAGE`，默认建议 `Chinese`。
  - Debate rounds / Risk rounds。
  - Checkpoint enabled。
  - Temperature / OpenAI reasoning effort。
  - 按钮：保存设置、重新加载、测试配置。

### 3.2 用户流程

1. 用户在左侧选择或输入 A 股代码，例如 `SZ002241`。
2. 右侧 `分析` tab 自动填充同一标的。
3. 用户点击“开始分析”。
4. 前端调用 `POST /tradingagents/analysis`。
5. 后端校验 A 股代码，转换为 TradingAgents ticker，加载设置，运行 TradingAgents。
6. 前端收到报告后，按分区展示；失败时显示可行动错误，例如缺少模型、缺少 backend URL、依赖未安装、API 调用失败。

---

## 4. 后端设计

### 4.1 新增文件

- Create: `tradingagents_models.py`
  - Pydantic 请求/响应模型。
- Create: `tradingagents_config.py`
  - `.env` 读取、白名单更新、密钥遮罩。
- Create: `tradingagents_adapter.py`
  - TradingAgents repo path 校验、A 股 ticker 转换、分析执行、报告字段规范化。
- Test:
  - `test/test_tradingagents_config.py`
  - `test/test_tradingagents_adapter.py`
  - `test/test_tradingagents_api.py`

### 4.2 配置白名单

只允许 UI 修改以下键：

```text
TRADINGAGENTS_LLM_PROVIDER
TRADINGAGENTS_LLM_BACKEND_URL
OPENAI_COMPATIBLE_API_KEY
TRADINGAGENTS_DEEP_THINK_LLM
TRADINGAGENTS_QUICK_THINK_LLM
TRADINGAGENTS_OUTPUT_LANGUAGE
TRADINGAGENTS_MAX_DEBATE_ROUNDS
TRADINGAGENTS_MAX_RISK_ROUNDS
TRADINGAGENTS_CHECKPOINT_ENABLED
TRADINGAGENTS_TEMPERATURE
TRADINGAGENTS_OPENAI_REASONING_EFFORT
```

默认 `.env` 路径：

```text
/Users/wanbo/knowledge/knowledge/repo/TradingAgents/.env
```

实现要求：

- 读取时返回 `api_key_set: bool`，不返回 `OPENAI_COMPATIBLE_API_KEY` 明文。
- 保存时保留未知键、注释和原有顺序。
- 空 API key 表示“不修改”；如需清空，使用显式 `clear_api_key: true`。
- 写入 `.env` 使用临时文件 + atomic replace，避免半写入。

### 4.3 API 设计

#### `GET /tradingagents/config`

返回遮罩后的设置。

```json
{
  "repo_path": "/Users/wanbo/knowledge/knowledge/repo/TradingAgents",
  "env_path": "/Users/wanbo/knowledge/knowledge/repo/TradingAgents/.env",
  "config": {
    "provider": "openai_compatible",
    "backend_url": "http://localhost:1234/v1",
    "deep_model": "model-name",
    "quick_model": "model-name",
    "output_language": "Chinese",
    "max_debate_rounds": 1,
    "max_risk_rounds": 1,
    "checkpoint_enabled": false,
    "temperature": null,
    "openai_reasoning_effort": null,
    "api_key_set": true
  }
}
```

#### `PUT /tradingagents/config`

保存设置。校验：

- `provider` Phase 3.0 只允许 `openai_compatible`。
- `backend_url` 必填，必须以 `http://` 或 `https://` 开头。
- rounds 必须是 `0-5` 范围内整数，默认建议 `1`。
- temperature 可为空；非空时必须是 `0-2` 数值。

#### `POST /tradingagents/config/test`

做轻量配置测试：

- repo path 是否存在。
- `.env` 是否可读写。
- provider/backend URL/model/API key 状态是否满足运行条件。
- 不在 MVP 中强制发起真实 LLM 调用，避免保存设置时产生费用。

#### `POST /tradingagents/analysis`

请求：

```json
{
  "symbol": "SZ002241",
  "analysis_date": "2026-07-05",
  "analysts": ["market", "news", "fundamentals"],
  "max_debate_rounds": 1,
  "max_risk_rounds": 1
}
```

响应：

```json
{
  "status": "succeeded",
  "symbol": "SZ002241",
  "tradingagents_ticker": "002241.SZ",
  "analysis_date": "2026-07-05",
  "elapsed_seconds": 123.4,
  "reports": {
    "market_report": "...",
    "sentiment_report": null,
    "news_report": "...",
    "fundamentals_report": "...",
    "research_decision": "...",
    "trader_plan": "...",
    "risk_discussion": "...",
    "portfolio_decision": "..."
  },
  "warnings": []
}
```

错误响应必须避免泄露密钥和完整请求头。

### 4.4 Adapter 策略

MVP 推荐直接 import TradingAgents，但用 adapter 隔离：

1. 将 TradingAgents repo path 临时加入 `sys.path`。
2. 使用 `python-dotenv` 或自定义解析加载 `.env` 到子运行环境。
3. 构造 `TradingAgentsGraph`，选择 analysts，创建初始 state。
4. 遍历 `graph.graph.stream(...)`，合并 chunks 为 `final_state`。
5. 从 `final_state` 提取报告。
6. 清理临时 `sys.path` 修改，避免污染 backtest 其他模块。

如果直接 import 发生依赖冲突，再降级为子进程方案：

- Create: `scripts/run_tradingagents_analysis.py`
- FastAPI 用 `subprocess.run(...)` 调用 TradingAgents repo 内 Python。
- stdout 输出 JSON，stderr 写入后端日志。

---

## 5. 前端设计

### 5.1 删除旧 UI

从 `templates/index.html` 删除：

- `#restoreInsightPanelButton`
- `#rightInsightColumn` 内旧“标的信息”卡片内容。
- `#insightStatus`
- `#insightWarnings`
- `#quotePanel`
- `#reportsList`
- `#fundFlowList`
- `#dragonTigerList`
- `#announcementsList`
- `loadMarketInsights(...)`
- `renderInsightList(...)`
- `renderQuote(...)`
- `toggleInsightPanel(...)` 中旧命名；如果仍保留折叠行为，改名为 `toggleAnalysisPanel(...)`。
- `setSymbol(...)` 中的 `loadMarketInsights(symbol)` 调用，改为同步分析表单 symbol。
- `/backtest` 提交成功后的 `loadMarketInsights(formData.symbol)` 调用。

### 5.2 新增 UI

右侧新增：

- Header: `智能分析`
- Bootstrap nav tabs:
  - `分析`
  - `设置`
- 分析 tab 元素：
  - `#taSymbol`
  - `#taAnalysisDate`
  - `#taAnalysts`
  - `#taRunButton`
  - `#taClearButton`
  - `#taStatus`
  - `#taReportTabs`
  - `#taReportContent`
- 设置 tab 元素：
  - `#taBackendUrl`
  - `#taApiKey`
  - `#taDeepModel`
  - `#taQuickModel`
  - `#taOutputLanguage`
  - `#taDebateRounds`
  - `#taRiskRounds`
  - `#taCheckpointEnabled`
  - `#taTemperature`
  - `#taReasoningEffort`
  - `#taSaveConfigButton`
  - `#taReloadConfigButton`
  - `#taTestConfigButton`

### 5.3 前端行为

- DOMContentLoaded 时调用 `loadTradingAgentsConfig()`。
- `setSymbol(symbol)` 只负责同步左侧 symbol、关闭搜索弹窗、更新 `#taSymbol`。
- 点击“开始分析”时：
  - 禁用按钮，显示 running 状态。
  - 调用 `/tradingagents/analysis`。
  - 成功后渲染报告 tabs。
  - 失败后显示错误 alert，并恢复按钮。
- Markdown MVP 可以先用纯文本 `<pre>` 或简单换行渲染；如引入 Markdown 渲染库，需确认 CSP/依赖来源并测试。

---

## 6. 测试计划

### 6.1 单元测试

- `test/test_tradingagents_config.py`
  - 读取 `.env` 时密钥被遮罩。
  - 保存配置保留未知键和注释。
  - 空 API key 不覆盖已有值。
  - `clear_api_key` 能显式清空。
- `test/test_tradingagents_adapter.py`
  - `SH603019 -> 603019.SS`。
  - `SZ002241 -> 002241.SZ`。
  - `BJxxxxxx -> xxxxxx.BJ` 如 TradingAgents 支持，否则返回明确错误。
  - 非 A 股代码被拒绝。
  - mock TradingAgentsGraph 后能把 `final_state` 转成响应 reports。
- `test/test_tradingagents_api.py`
  - `GET /tradingagents/config` 不泄露 API key。
  - `PUT /tradingagents/config` 校验 backend URL。
  - `POST /tradingagents/analysis` 对 invalid symbol 返回 400。
  - mock adapter 下 successful analysis 返回结构化 reports。

### 6.2 删除旧功能测试

- 删除或重写 `test/test_market_insights_api.py`。
- 确认 `/market-insights/{symbol}` 返回 404。
- 确认前端不再出现旧文案：
  - `标的信息`
  - `研报`
  - `资金`
  - `龙虎榜`
  - `公告`

### 6.3 手动验证

1. 启动 FastAPI。
2. 打开首页。
3. 搜索或输入 `SZ002241`。
4. 确认右侧分析 tab 自动填充 `SZ002241`。
5. 打开设置 tab，确认密钥只显示已设置状态。
6. 保存 openai-compatible 配置。
7. 点击测试配置，确认返回 pass/warnings。
8. mock 或真实运行一次分析。
9. 确认报告分区可读，失败时错误不泄露密钥。

---

## 7. 详细子任务和 Todo List

### 7.1 全局 Todo List

按顺序执行，不跳任务；每个任务完成后运行本任务 focused tests，再提交。

- [x] T0: 建立 Phase 3.0 基线检查，确认旧面板入口、测试入口和 TradingAgents repo 可用。
- [x] T1: 删除旧市场信息后端 `/market-insights` 与旧股票详情 endpoint。
- [x] T2: 删除旧右侧信息面板 UI 和所有旧 JS/CSS 调用点。
- [x] T3: 新增 TradingAgents 请求/响应 Pydantic 模型。
- [x] T4: 新增 TradingAgents `.env` 配置读取、遮罩、保存和校验。
- [x] T5: 新增 A 股 symbol 转换和 TradingAgents report extraction adapter。
- [x] T6: 新增 TradingAgents 分析执行 adapter，先 mock 可测，再接真实 import。
- [x] T7: 新增 FastAPI `/tradingagents/*` endpoints。
- [x] T8: 新增右侧 `智能分析` 面板的 `分析` subtab。
- [x] T9: 新增右侧 `智能分析` 面板的 `设置` subtab。
- [x] T10: 接通前端和后端 API，完成状态、错误和报告渲染。
- [ ] T11: 做旧字符串清理、focused tests、核心回归和手动验收。

### T0: 基线检查和保护栏

**Goal:** 在改代码前确认当前工作树、旧入口和 TradingAgents repo 状态，避免误删仍被使用的逻辑。

**Files:**

- Read: `main.py`
- Read: `templates/index.html`
- Read: `test/test_market_insights_api.py`
- Read: `/Users/wanbo/knowledge/knowledge/repo/TradingAgents/tradingagents/default_config.py`
- Read: `/Users/wanbo/knowledge/knowledge/repo/TradingAgents/cli/main.py`

**Todo:**

- [x] 运行 `git status --short`，确认是否有用户未提交改动。
- [x] 用 CodeGraph 或 `rg` 列出旧入口引用：

```bash
rg -n "market-insights|stock-info|loadMarketInsights|renderInsightList|renderQuote|quotePanel|dragonTiger|fundFlow|标的信息|龙虎榜" main.py templates test static
```

- [x] 记录哪些测试依赖 `market_insights.py`。
- [x] 确认 TradingAgents repo 路径存在：

```bash
test -d /Users/wanbo/knowledge/knowledge/repo/TradingAgents && test -f /Users/wanbo/knowledge/knowledge/repo/TradingAgents/.env
```

- [x] 只做检查，不改文件。

**Expected:** 工作树状态明确，旧入口清单明确，TradingAgents `.env` 存在。

### T1: 删除旧市场信息后端

**Goal:** 后端不再提供旧右侧“标的信息”数据源。

**Files:**

- Modify: `main.py`
- Delete: `market_insights.py`
- Modify/Delete: `test/test_market_insights_api.py`
- Possibly modify: `stock_search.py`
- Possibly modify: `test/test_comprehensive.py`

**Todo:**

- [x] 先改测试：将旧 `test/test_market_insights_api.py` 替换为 `test/test_removed_market_insights_api.py`，断言 `/market-insights/SZ002241` 返回 404。
- [x] 运行新测试，确认当前代码 FAIL，因为 endpoint 仍存在：

```bash
pytest test/test_removed_market_insights_api.py -v
```

- [x] 从 `main.py` 删除 `from market_insights import get_market_insights`。
- [x] 从 `main.py` 删除 `market_insights_endpoint(...)`。
- [x] 从 `main.py` 删除 `get_stock_info_endpoint(...)`，除非基线检查发现其他前端仍需要它。
- [x] 如果 `/stock-info` 删除后 `stock_search.get_stock_info(...)` 无调用，删除 `StockSearcher.get_stock_info(...)` 和 module-level `get_stock_info(...)`。
- [x] 删除 `market_insights.py`。
- [x] 运行：

```bash
pytest test/test_removed_market_insights_api.py -v
pytest test/test_comprehensive.py -v
```

- [x] 搜索确认后端旧入口无残留：

```bash
rg -n "market_insights|get_market_insights|/market-insights|/stock-info|get_stock_info_endpoint" .
```

- [x] Commit:

```bash
git add main.py stock_search.py test/test_removed_market_insights_api.py test/test_market_insights_api.py market_insights.py
git commit -m "refactor: remove obsolete market insights backend"
```

**Expected:** `/market-insights/*` 不再是 FastAPI route；删除旧模块后测试通过。

### T2: 删除旧右侧信息面板 UI

**Goal:** 前端完全移除旧“标的信息”区域，不再请求旧 API。

**Files:**

- Modify: `templates/index.html`

**Todo:**

- [x] 删除 `.insight-restore-tab` 旧样式，或改名为 `.analysis-restore-tab`。
- [x] 删除旧 `.insight-section`、`.insight-item`、`.insight-empty`、`.quote-strip`、`.insight-header` 中只服务旧面板的样式。
- [x] 删除旧 restore button：

```html
<button id="restoreInsightPanelButton" ...>
```

- [x] 用空的 `智能分析` card 替换 `#rightInsightColumn` 旧内容，先只保留 header 和占位 body。
- [x] 从 `setSymbol(symbol)` 删除 `loadMarketInsights(symbol)`，改为调用后续会实现的 `syncTradingAgentsSymbol(symbol)`；此时可先放 no-op 函数。
- [x] 删除 `renderInsightList(...)`。
- [x] 删除 `renderQuote(...)`。
- [x] 删除 `loadMarketInsights(...)`。
- [x] 将 `toggleInsightPanel(...)` 改名为 `toggleAnalysisPanel(...)`，并同步按钮 id、class、aria-label 文案。
- [x] 删除 `/backtest` 成功后的 `loadMarketInsights(formData.symbol)` 调用。
- [x] 搜索确认旧 UI 字符串无残留，计划文档除外：

```bash
rg -n "标的信息|研报|资金|龙虎榜|公告|loadMarketInsights|renderInsightList|renderQuote|quotePanel|fundFlow|dragonTiger" templates static main.py
```

- [x] Commit:

```bash
git add templates/index.html
git commit -m "refactor: remove obsolete right info panel ui"
```

**Expected:** 首页不再出现旧面板内容；前端不再引用 `/market-insights`。

### T3: 新增 TradingAgents Pydantic 模型

**Goal:** 先定义 API 合同，避免 adapter/API/UI 各写各的字段。

**Files:**

- Create: `tradingagents_models.py`
- Create: `test/test_tradingagents_models.py`

**Todo:**

- [x] 新增 `TradingAgentsConfigView`：
  - `provider: str = "openai_compatible"`
  - `backend_url: str | None`
  - `deep_model: str | None`
  - `quick_model: str | None`
  - `output_language: str = "Chinese"`
  - `max_debate_rounds: int = 1`
  - `max_risk_rounds: int = 1`
  - `checkpoint_enabled: bool = False`
  - `temperature: float | None = None`
  - `openai_reasoning_effort: str | None = None`
  - `api_key_set: bool = False`
- [x] 新增 `TradingAgentsConfigUpdate`，包含 `api_key: str | None = None` 和 `clear_api_key: bool = False`。
- [x] 新增 `TradingAgentsConfigResponse`，包含 `repo_path`、`env_path`、`config`。
- [x] 新增 `TradingAgentsConfigTestResponse`，包含 `ok: bool`、`checks: list[dict]`、`warnings: list[str]`。
- [x] 新增 `TradingAgentsAnalysisRequest`：
  - `symbol`
  - `analysis_date`
  - `analysts`
  - `max_debate_rounds`
  - `max_risk_rounds`
- [x] 新增 `TradingAgentsReports`：
  - `market_report`
  - `sentiment_report`
  - `news_report`
  - `fundamentals_report`
  - `research_decision`
  - `trader_plan`
  - `risk_discussion`
  - `portfolio_decision`
- [x] 新增 `TradingAgentsAnalysisResponse`。
- [x] 写模型测试：
  - invalid provider 被拒绝。
  - invalid backend URL 被拒绝。
  - rounds 超出范围被拒绝。
  - `analysts=[]` 被拒绝。
  - 非白名单 analyst 被拒绝。
- [x] 运行：

```bash
pytest test/test_tradingagents_models.py -v
```

- [x] Commit:

```bash
git add tradingagents_models.py test/test_tradingagents_models.py
git commit -m "feat: add tradingagents api models"
```

**Expected:** API 字段和基础校验稳定。

### T4: 新增 `.env` 配置管理

**Goal:** 安全读取和保存 TradingAgents OpenAI-compatible 配置，不泄露密钥。

**Files:**

- Create: `tradingagents_config.py`
- Create: `test/test_tradingagents_config.py`
- Use: `tradingagents_models.py`

**Todo:**

- [x] 定义常量：
  - `TRADINGAGENTS_REPO_PATH`
  - `TRADINGAGENTS_ENV_PATH`
  - `ALLOWED_ENV_KEYS`
  - `SECRET_ENV_KEYS = {"OPENAI_COMPATIBLE_API_KEY"}`
- [x] 实现 `parse_env_file(path: Path) -> tuple[list[str], dict[str, str]]`，保留原始行。
- [x] 实现 `get_config_view(env_path: Path = TRADINGAGENTS_ENV_PATH) -> TradingAgentsConfigResponse`。
- [x] 实现 `update_config(update: TradingAgentsConfigUpdate, env_path: Path = ...) -> TradingAgentsConfigResponse`。
- [x] 实现 `test_config(env_path: Path = ...) -> TradingAgentsConfigTestResponse`。
- [x] 写测试：读取时 `api_key_set=True` 但 response 不含 key 明文。
- [x] 写测试：保存 `backend_url` 后保留未知键和注释。
- [x] 写测试：`api_key=None` 不覆盖已有 key。
- [x] 写测试：`api_key=""` 不覆盖已有 key。
- [x] 写测试：`clear_api_key=True` 清空 key。
- [x] 写测试：写入使用临时文件和 replace，可通过 monkeypatch `Path.replace` 或检查结果文件完整性。
- [x] 运行：

```bash
pytest test/test_tradingagents_config.py -v
```

- [x] Commit:

```bash
git add tradingagents_config.py test/test_tradingagents_config.py
git commit -m "feat: manage tradingagents env config"
```

**Expected:** 配置可安全读取/保存，API key 永不出现在返回对象。

### T5: 新增 symbol 转换和报告提取 adapter

**Goal:** 把 backtest 的 A 股代码与 TradingAgents 输出结构集中适配。

**Files:**

- Create: `tradingagents_adapter.py`
- Create: `test/test_tradingagents_adapter.py`
- Use: `tradingagents_models.py`

**Todo:**

- [x] 实现 `normalize_a_share_symbol(symbol: str) -> str`，输出 backtest 标准 `SH603019` / `SZ002241` / `BJxxxxxx`。
- [x] 实现 `to_tradingagents_ticker(symbol: str) -> str`：
  - `SH603019 -> 603019.SS`
  - `603019.SH -> 603019.SS`
  - `SZ002241 -> 002241.SZ`
  - `002241.SZ -> 002241.SZ`
  - `BJxxxxxx -> xxxxxx.BJ`，如后续验证 TradingAgents 不支持北交所，则改为明确 400。
- [x] 实现 `validate_analysts(analysts: list[str]) -> list[str]`，只允许 `market`、`social`、`news`、`fundamentals`。
- [x] 实现 `extract_reports(final_state: dict) -> TradingAgentsReports`。
- [x] 报告提取规则：
  - `market_report` 从 `final_state["market_report"]`。
  - `sentiment_report` 从 `final_state["sentiment_report"]`。
  - `news_report` 从 `final_state["news_report"]`。
  - `fundamentals_report` 从 `final_state["fundamentals_report"]`。
  - `research_decision` 优先 `investment_debate_state.judge_decision`，否则 `investment_plan`。
  - `trader_plan` 从 `trader_investment_plan`。
  - `risk_discussion` 合并 `risk_debate_state.aggressive_history`、`conservative_history`、`neutral_history`。
  - `portfolio_decision` 优先 `risk_debate_state.judge_decision`，否则 `final_trade_decision`。
- [x] 写测试覆盖 symbol 转换。
- [x] 写测试覆盖 invalid symbols：
  - `AAPL`
  - `0700.HK`
  - `BTC-USD`
  - 空字符串。
- [x] 写测试覆盖 final_state 缺字段时返回 `None`，不抛异常。
- [x] 运行：

```bash
pytest test/test_tradingagents_adapter.py -v
```

- [x] Commit:

```bash
git add tradingagents_adapter.py test/test_tradingagents_adapter.py
git commit -m "feat: adapt a-share symbols and tradingagents reports"
```

**Expected:** 转换和报告提取无需真实 LLM 即可测试。

### T6: 新增 TradingAgents 分析执行 adapter

**Goal:** 提供一个可 mock 的 `run_analysis(...)` 函数，真实运行 TradingAgents 时不污染 backtest 全局状态。

**Files:**

- Modify: `tradingagents_adapter.py`
- Modify: `test/test_tradingagents_adapter.py`

**Todo:**

- [x] 定义 `TradingAgentsAdapterError(Exception)`，错误消息经过 secret sanitization。
- [x] 实现 `sanitize_error_message(message: str, env_values: dict[str, str]) -> str`。
- [x] 实现 context manager `temporary_sys_path(path: Path)`，退出时恢复原值。
- [x] 实现 context manager `temporary_environ(overrides: dict[str, str])`，退出时恢复原值。
- [x] 实现 `load_tradingagents_env(env_path: Path) -> dict[str, str]`，复用 T4 parser。
- [x] 实现 `build_run_config(...)`：
  - provider/backend/model 从 `.env` 和 request override 合并。
  - output language 默认 `Chinese`。
  - debate/risk rounds 使用 request 优先，其次 `.env`。
- [x] 实现 `run_tradingagents_analysis(request: TradingAgentsAnalysisRequest) -> TradingAgentsAnalysisResponse`。
- [x] 真实 import 路径：
  - `from tradingagents.graph.trading_graph import TradingAgentsGraph`
  - `from tradingagents.graph.analyst_execution import build_analyst_execution_plan`
  - 依照 TradingAgents CLI 的 graph stream 合并 chunks。
- [x] 测试中 monkeypatch `TradingAgentsGraph`，不调用真实 LLM。
- [x] 测试分析成功时 response 包含 `elapsed_seconds`、`tradingagents_ticker`、reports。
- [x] 测试 import 失败时抛出 `TradingAgentsAdapterError`，错误不包含 API key。
- [x] 运行：

```bash
pytest test/test_tradingagents_adapter.py -v
```

- [x] Commit:

```bash
git add tradingagents_adapter.py test/test_tradingagents_adapter.py
git commit -m "feat: run tradingagents analysis through adapter"
```

**Expected:** Web 层只调用 adapter，不直接碰 TradingAgents internals。

### T7: 新增 FastAPI endpoints

**Goal:** 暴露配置和分析 API，前端只依赖 `/tradingagents/*`。

**Files:**

- Modify: `main.py`
- Create: `test/test_tradingagents_api.py`
- Use: `tradingagents_config.py`
- Use: `tradingagents_adapter.py`
- Use: `tradingagents_models.py`

**Todo:**

- [x] 写测试：`GET /tradingagents/config` 返回 200，且不含 key 明文。
- [x] 写测试：`PUT /tradingagents/config` 保存 backend URL 和 models。
- [x] 写测试：`POST /tradingagents/config/test` 返回 checks。
- [x] 写测试：`POST /tradingagents/analysis` invalid symbol 返回 400。
- [x] 写测试：`POST /tradingagents/analysis` mock adapter 成功返回 reports。
- [x] 在 `main.py` import 新模型和 helper。
- [x] 新增 `GET /tradingagents/config`。
- [x] 新增 `PUT /tradingagents/config`。
- [x] 新增 `POST /tradingagents/config/test`。
- [x] 新增 `POST /tradingagents/analysis`。
- [x] 分析 endpoint 用 `fastapi.concurrency.run_in_threadpool(...)` 包住同步 adapter。
- [x] 捕获 `TradingAgentsAdapterError` 返回 502 或 500，消息 sanitized。
- [x] 捕获 validation error 返回 400/422。
- [x] 运行：

```bash
pytest test/test_tradingagents_api.py -v
```

- [x] Commit:

```bash
git add main.py test/test_tradingagents_api.py
git commit -m "feat: expose tradingagents web api"
```

**Expected:** 后端 API 层不泄露 secret，不阻塞 event loop。

### T8: 新增右侧 `分析` subtab UI

**Goal:** 右侧区域能输入分析参数、触发分析、展示报告。

**Files:**

- Modify: `templates/index.html`

**Todo:**

- [x] 将右侧 header 改为 `智能分析`，图标可用 `fas fa-brain` 或当前 Font Awesome 可用图标。
- [x] 新增 Bootstrap nav tabs：`分析` 和 `设置`。
- [x] 在分析 tab 中新增：
  - `#taSymbol`
  - `#taAnalysisDate`
  - analyst checkboxes: `market`、`news`、`fundamentals`、`social`
  - `#taRunButton`
  - `#taClearButton`
  - `#taStatus`
  - `#taReportTabs`
  - `#taReportContent`
- [x] 实现 `syncTradingAgentsSymbol(symbol)`。
- [x] DOMContentLoaded 时：
  - 设置 `#taAnalysisDate` 为今天。
  - 从左侧 `#symbol` 同步默认 symbol。
- [x] 实现 `getSelectedTradingAgentsAnalysts()`。
- [x] 实现 `setTradingAgentsStatus(kind, message)`，支持 idle/running/succeeded/failed。
- [x] 实现 `renderTradingAgentsReports(reports)`，空报告不渲染 tab。
- [x] 报告内容先用 escaped text + `<pre class="ta-report">`，不要引入新 Markdown 依赖。
- [x] 点击 clear 时清空报告和状态。
- [x] Commit:

```bash
git add templates/index.html
git commit -m "feat: add tradingagents analysis tab"
```

**Expected:** 不接 API 时 UI 也可静态加载，字段不溢出。

### T9: 新增右侧 `设置` subtab UI

**Goal:** 用户能查看和保存 OpenAI-compatible 模型配置。

**Files:**

- Modify: `templates/index.html`

**Todo:**

- [x] 在设置 tab 中新增 provider readonly/select，MVP 默认 `openai_compatible`。
- [x] 新增 `#taBackendUrl`。
- [x] 新增 `#taApiKey`，placeholder 根据 `api_key_set` 显示 `已设置，留空则不修改` 或 `未设置`。
- [x] 新增 `#taClearApiKey` checkbox。
- [x] 新增 `#taDeepModel`。
- [x] 新增 `#taQuickModel`。
- [x] 新增 `#taOutputLanguage`。
- [x] 新增 `#taDebateRounds`。
- [x] 新增 `#taRiskRounds`。
- [x] 新增 `#taCheckpointEnabled`。
- [x] 新增 `#taTemperature`。
- [x] 新增 `#taReasoningEffort`。
- [x] 新增按钮：
  - `#taSaveConfigButton`
  - `#taReloadConfigButton`
  - `#taTestConfigButton`
- [x] 实现 `populateTradingAgentsConfig(config)`。
- [x] 实现 `collectTradingAgentsConfigPayload()`。
- [x] 实现设置 tab 的状态 alert：`#taConfigStatus`。
- [x] Commit:

```bash
git add templates/index.html
git commit -m "feat: add tradingagents settings tab"
```

**Expected:** 设置表单能表达 `.env` 白名单字段，API key 不回填。

### T10: 接通前端 API

**Goal:** 完成前端和 `/tradingagents/*` API 的交互。

**Files:**

- Modify: `templates/index.html`

**Todo:**

- [x] 实现 `loadTradingAgentsConfig()`：
  - GET `/tradingagents/config`
  - 成功后填充设置表单。
  - 失败后显示 config warning。
- [x] 实现 `saveTradingAgentsConfig()`：
  - PUT `/tradingagents/config`
  - 保存时禁用按钮。
  - 成功后清空 API key input，并刷新配置。
- [x] 实现 `testTradingAgentsConfig()`：
  - POST `/tradingagents/config/test`
  - 渲染 checks/warnings。
- [x] 实现 `runTradingAgentsAnalysis()`：
  - POST `/tradingagents/analysis`
  - 请求体包含 symbol、analysis_date、analysts、rounds。
  - running 时禁用 run button。
  - 成功后渲染 reports。
  - 失败时显示 response detail。
- [x] 统一 `fetchJson(url, options)` helper，处理非 2xx。
- [x] 所有用户可见 HTML 注入点使用 `escapeHtml(...)` 或 `textContent`。
- [x] `setSymbol(symbol)`、搜索结果点击、快速选择按钮都同步 `#taSymbol`。
- [x] 运行服务手动打开页面，检查 console 无 JS error。
- [x] Commit:

```bash
git add templates/index.html
git commit -m "feat: connect tradingagents panel to api"
```

**Expected:** UI 能保存配置、测试配置、发起 mock/真实分析并展示结果。

### T11: 集成验证和清理

**Goal:** 证明 Phase 3.0 改动没有残留旧面板路径，也没有破坏核心回测功能。

**Files:**

- Modify as needed.

**Todo:**

- [ ] 运行 TradingAgents focused tests：

```bash
pytest test/test_tradingagents_models.py test/test_tradingagents_config.py test/test_tradingagents_adapter.py test/test_tradingagents_api.py -v
```

- [ ] 运行删除旧面板测试：

```bash
pytest test/test_removed_market_insights_api.py -v
```

- [ ] 运行核心回归：

```bash
pytest test/test_market_data_sources.py test/test_optimization_runner.py test/test_analytics.py -v
```

- [ ] 搜索旧路径，结果只能出现在历史 docs/plans 中：

```bash
rg -n "market-insights|标的信息|龙虎榜|dragonTiger|fundFlow|quotePanel|loadMarketInsights|renderInsightList|renderQuote" .
```

- [ ] 搜索 secret 泄露风险：

```bash
rg -n "OPENAI_COMPATIBLE_API_KEY|api_key|Authorization" main.py tradingagents_*.py templates/index.html test
```

- [ ] 启动服务：

```bash
uvicorn main:app --reload
```

- [ ] 手动验证：
  - 首页加载无 console error。
  - 输入 `SZ002241` 后分析 tab 同步。
  - 设置 tab 能显示 API key 已设置/未设置但无明文。
  - 保存设置后 `.env` 保留未知键。
  - 无 backend URL 时测试配置给出可行动 warning。
  - invalid symbol `AAPL` 分析返回错误。
  - mock 或真实分析成功后报告分区展示。
- [ ] 最终提交：

```bash
git add .
git commit -m "test: verify tradingagents phase 3 integration"
```

**Expected:** Focused tests 通过，核心回归通过，旧面板调用清零，UI 手动验收通过。

---

## 8. 风险和缓解

- **依赖冲突:** TradingAgents 依赖可能与 backtest 环境冲突。先用 adapter 隔离；如冲突，切换到子进程方案。
- **运行耗时:** 多智能体分析可能很慢。MVP 禁用实时流式，只显示 running 状态；后续可加任务队列/SSE。
- **费用风险:** 设置测试不做真实 LLM 调用；真实分析必须由用户点击触发。
- **密钥泄露:** API 永远不返回 key 明文，错误日志过滤敏感 env。
- **A 股数据兼容:** backtest 内部使用 `SH/SZ` 前缀，TradingAgents 常用 Yahoo suffix。adapter 必须集中处理转换，不让前端散落转换逻辑。
- **外部 repo 可变:** TradingAgents 路径固定但可能更新。所有 import 错误都要显示清楚，并在 tests 中 mock 外部依赖。

---

## 9. 验收标准

- 右侧旧“标的信息”区域完全消失。
- 后端旧 `/market-insights/{symbol}` 和旧信息面板逻辑被删除。
- 右侧出现 `分析` / `设置` 两个 subtab。
- 设置 tab 可读取和保存 OpenAI-compatible 配置，且不泄露 API key。
- 分析 tab 可对当前 A 股标的发起 TradingAgents 分析，并展示结构化报告。
- 非 A 股输入被拒绝，错误提示明确。
- Focused tests 通过。
- 代码中没有残留旧面板调用路径。

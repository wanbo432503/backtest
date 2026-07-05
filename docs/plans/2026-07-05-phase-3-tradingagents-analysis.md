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

## 7. 任务拆分

### Task 1: 删除旧市场信息面板后端

**Files:**

- Modify: `main.py`
- Delete: `market_insights.py`
- Delete or rewrite: `test/test_market_insights_api.py`
- Modify tests that import `stock_search.get_stock_info(...)` only because the old `/stock-info` endpoint existed.

**Steps:**

- 写 failing test，确认 `/market-insights/SZ002241` 不再是受支持 API。
- 删除 `from market_insights import get_market_insights`。
- 删除 `market_insights_endpoint(...)`。
- 删除 `get_stock_info_endpoint(...)`，如果 `stock_search.get_stock_info(...)` 无其他真实用途，也删除该 helper 和相关测试。
- 运行 focused tests。
- Commit: `refactor: remove obsolete market insights backend`

### Task 2: 新增 TradingAgents 配置模型和 env 管理

**Files:**

- Create: `tradingagents_models.py`
- Create: `tradingagents_config.py`
- Create: `test/test_tradingagents_config.py`

**Steps:**

- 先写密钥遮罩、白名单保存、atomic replace 的测试。
- 实现 env parser/writer。
- 实现 Pydantic config model 和 validation。
- 运行 `pytest test/test_tradingagents_config.py -v`。
- Commit: `feat: add tradingagents config management`

### Task 3: 新增 TradingAgents Adapter

**Files:**

- Create: `tradingagents_adapter.py`
- Create: `test/test_tradingagents_adapter.py`

**Steps:**

- 先写 A 股 symbol 转换测试。
- 写 mock graph 的报告提取测试。
- 实现 repo path 校验、env loading、analyst validation、report extraction。
- 实现真实 TradingAgents import 的错误包装，错误信息必须可行动且不泄露 secret。
- 运行 adapter tests。
- Commit: `feat: add tradingagents analysis adapter`

### Task 4: 新增 FastAPI endpoints

**Files:**

- Modify: `main.py`
- Create: `test/test_tradingagents_api.py`

**Steps:**

- 先写 API tests，使用 mock config/adapter。
- 新增：
  - `GET /tradingagents/config`
  - `PUT /tradingagents/config`
  - `POST /tradingagents/config/test`
  - `POST /tradingagents/analysis`
- 长耗时分析用线程池执行，避免阻塞 event loop。
- 运行 `pytest test/test_tradingagents_api.py -v`。
- Commit: `feat: expose tradingagents api`

### Task 5: 替换右侧前端 UI

**Files:**

- Modify: `templates/index.html`

**Steps:**

- 删除旧 insight CSS、HTML、JS。
- 新增 TradingAgents tabs、forms、状态区、报告区。
- 新增 JS：
  - `loadTradingAgentsConfig`
  - `saveTradingAgentsConfig`
  - `testTradingAgentsConfig`
  - `syncTradingAgentsSymbol`
  - `runTradingAgentsAnalysis`
  - `renderTradingAgentsReports`
- 保留布局稳定性，右侧列高度仍跟随页面。
- 手动检查移动端和桌面宽度没有文字溢出。
- Commit: `feat: replace right panel with tradingagents ui`

### Task 6: 集成验证和清理

**Files:**

- Modify as needed.

**Steps:**

- 运行：

```bash
pytest test/test_tradingagents_config.py test/test_tradingagents_adapter.py test/test_tradingagents_api.py -v
```

- 运行现有核心测试：

```bash
pytest test/test_market_data_sources.py test/test_optimization_runner.py -v
```

- 启动本地服务并手动验证 UI。
- 搜索旧字符串：

```bash
rg -n "market-insights|标的信息|龙虎榜|dragonTiger|fundFlow|quotePanel|loadMarketInsights" .
```

- 确认只剩历史计划或无结果。
- Commit: `test: verify tradingagents phase 3 integration`

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

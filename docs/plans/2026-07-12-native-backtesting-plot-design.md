# 单股回测原生 backtesting.py 图表设计

## 目标

单股回测继续使用项目现有统一策略模拟器作为交易、A 股规则、费用和统计的唯一事实来源，但返回的 `plot_html` 改由 `backtesting.py` 的 `Backtest.plot()` 生成。删除当前手工拼装的 Close 折线、买卖三角和 Equity 折线。

## 数据适配

绘图适配器把统一模拟结果转换为 `backtesting.py` 原生绘图结果：

- OHLCV 直接使用本次单股回测行情；
- 净值曲线按行情日期对齐，交给库计算回撤显示字段；
- 买入和卖出成交按轮次配对，生成库需要的 Entry/Exit、价格、数量、盈亏和收益率字段；
- 回测结束仍有持仓时，仅在绘图数据中按最后收盘价补一个可视化退出点，不修改真实交易列表或页面统计；
- 调用 `Backtest.plot(open_browser=False)` 输出原生 Equity、Profit/Loss、OHLC、Trades 和 Volume 图层。

`backtesting.py` 固定为当前已验证的 0.6.5 版本，因为适配器需要使用该版本的原生统计结果结构。生成失败应作为单股回测错误返回，不能静默退回自绘图。

## 验证

测试必须先证明旧实现不包含原生图层，再验证生成 HTML 包含 `Equity`、`Profit / Loss`、`Trades (N)` 和 `Volume`；同时确认 `Backtest.plot()` 被实际调用、原有 API 返回结构与策略统计不变，并运行完整回归。

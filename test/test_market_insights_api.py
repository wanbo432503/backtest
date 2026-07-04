from fastapi.testclient import TestClient
import requests

import main
import market_insights


def test_market_insights_endpoint_returns_sections(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_market_insights",
        lambda symbol: {
            "symbol": symbol,
            "reports": [{"title": "Report A"}],
            "fund_flow": [{"date": "2026-07-03", "main_net_inflow": 123.0}],
            "dragon_tiger": [],
            "announcements": [{"title": "Announcement A"}],
            "warnings": [],
        },
    )

    client = TestClient(main.app)
    response = client.get("/market-insights/600519")

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "600519"
    assert set(["reports", "fund_flow", "dragon_tiger", "announcements", "warnings"]).issubset(body)


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fund_flow_falls_back_to_realtime_snapshot_when_history_endpoint_disconnects(monkeypatch):
    calls = []

    def fake_em_get(url, params=None, headers=None, timeout=15):
        calls.append(url)
        if "fflow/daykline" in url:
            raise requests.exceptions.ProxyError("proxy disconnected")
        return DummyResponse(
            {
                "data": {
                    "diff": [
                        {
                            "f12": "603019",
                            "f14": "中科曙光",
                            "f62": -296569424.0,
                            "f84": 103017552.0,
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(market_insights, "em_get", fake_em_get)

    rows = market_insights.fetch_fund_flow_120d("603019")

    assert any("fflow/daykline" in call for call in calls)
    assert any("ulist.np/get" in call for call in calls)
    assert rows == [
        {
            "date": "最新",
            "main_net_inflow": -296569424.0,
            "small_net_inflow": 103017552.0,
        }
    ]


def test_em_get_retries_direct_when_proxy_disconnects(monkeypatch):
    direct_sessions = []
    response = DummyResponse({"ok": True})

    def broken_proxy_get(*args, **kwargs):
        raise requests.exceptions.ProxyError("proxy disconnected")

    class DirectSession:
        def __init__(self):
            self.trust_env = True
            self.headers = {}
            direct_sessions.append(self)

        def get(self, *args, **kwargs):
            return response

    monkeypatch.setattr(market_insights.EM_SESSION, "get", broken_proxy_get)
    monkeypatch.setattr(market_insights.requests, "Session", DirectSession)

    result = market_insights.em_get("https://push2.eastmoney.com/test")

    assert result is response
    assert direct_sessions[0].trust_env is False


def test_market_insights_warning_uses_short_network_message(monkeypatch):
    monkeypatch.setattr(market_insights, "fetch_tencent_quote", lambda codes: {})
    monkeypatch.setattr(market_insights, "fetch_eastmoney_reports", lambda code: [])
    monkeypatch.setattr(
        market_insights,
        "fetch_fund_flow_120d",
        lambda code: (_ for _ in ()).throw(requests.exceptions.ProxyError("very long proxy detail")),
    )
    monkeypatch.setattr(market_insights, "fetch_dragon_tiger_board", lambda code: [])
    monkeypatch.setattr(market_insights, "fetch_cninfo_announcements", lambda code: [])

    result = market_insights.get_market_insights("SH603019")

    assert result["fund_flow"] == []
    assert result["warnings"] == ["fund_flow 获取失败: 网络连接失败，可能是代理或东方财富接口临时断开"]

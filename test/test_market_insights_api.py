from fastapi.testclient import TestClient

import main


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

from fastapi.testclient import TestClient

import main


def test_multi_market_search_rejects_non_a_share_market():
    client = TestClient(main.app)

    response = client.get("/search-stocks-multi-market", params={"query": "AAPL", "market": "US"})

    assert response.status_code == 400
    assert "当前仅支持 CN/A股" in response.json()["detail"]

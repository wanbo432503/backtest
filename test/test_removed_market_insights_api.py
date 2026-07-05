from fastapi.testclient import TestClient

from main import app


def test_market_insights_endpoint_is_removed():
    client = TestClient(app)

    response = client.get("/market-insights/SZ002241")

    assert response.status_code == 404


def test_stock_info_endpoint_is_removed():
    client = TestClient(app)

    response = client.get("/stock-info/SZ002241")

    assert response.status_code == 404

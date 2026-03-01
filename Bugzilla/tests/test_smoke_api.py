from fastapi.testclient import TestClient

from naplace.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "naplace_http_requests_total" in r.text

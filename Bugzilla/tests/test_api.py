from http import HTTPStatus

from fastapi.testclient import TestClient

from naplace.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert data == {"status": "ok"}


def test_gru_predict_success():
    payload = {
        "texts": [
            {"text": "Crash when opening preferences"},
            {"text": "UI glitch in toolbar"},
        ]
    }

    response = client.post("/predict/gru", json=payload)

    assert response.status_code == HTTPStatus.OK
    data = response.json()

    assert data["model_name"] == "gru"
    assert "predictions" in data
    assert len(data["predictions"]) == len(payload["texts"])

    first_pred = data["predictions"][0]
    assert "input_text" in first_pred
    assert "predicted_label" in first_pred
    # probability potrebbe essere None, ma in genere sarà un float
    assert "probability" in first_pred


def test_lstm_predict_success():
    payload = {
        "texts": [
            {"text": "Error when saving project"},
        ]
    }

    response = client.post("/predict/lstm", json=payload)

    assert response.status_code == HTTPStatus.OK
    data = response.json()

    assert data["model_name"] == "lstm"
    assert len(data["predictions"]) == len(payload["texts"])


def test_predict_with_empty_texts_returns_400():
    payload = {"texts": []}

    response = client.post("/predict/gru", json=payload)

    assert response.status_code == HTTPStatus.BAD_REQUEST

    data = response.json()
    # detail viene dalla HTTPException nel main.py
    assert "detail" in data
    assert data["detail"] == "No texts provided for prediction."

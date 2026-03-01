from locust import HttpUser, task, between
import random

SAMPLE_TEXTS = [
    "Crash when opening settings panel",
    "UI freezes after clicking save button",
    "Unexpected behavior when switching tabs",
    "Application fails to start on Windows",
    "Memory leak observed after long usage",
]

class NaplaceUser(HttpUser):
    wait_time = between(1, 3)

    def _payload(self):
        return {"texts": [{"text": random.choice(SAMPLE_TEXTS)}]}

    @task(2)
    def health(self):
        with self.client.get("/health", name="/health", timeout=10, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"health status={r.status_code} body={r.text[:200]}")

    @task(7)
    def predict_setfit(self):
        with self.client.post("/predict/setfit", name="/predict/setfit", json=self._payload(), timeout=30, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"setfit status={r.status_code} body={r.text[:200]}")

    @task(3)
    def predict_tfidf(self):
        with self.client.post("/predict/tfidf", name="/predict/tfidf", json=self._payload(), timeout=30, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"tfidf status={r.status_code} body={r.text[:200]}")

    @task(2)
    def predict_gru(self):
        with self.client.post("/predict/gru", name="/predict/gru", json=self._payload(), timeout=30, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"gru status={r.status_code} body={r.text[:200]}")

    @task(1)
    def predict_lstm(self):
        with self.client.post("/predict/lstm", name="/predict/lstm", json=self._payload(), timeout=30, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"lstm status={r.status_code} body={r.text[:200]}")

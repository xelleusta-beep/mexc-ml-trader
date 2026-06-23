import time
from typing import Any


class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.status = "idle"
        self.last_update = 0.0
        self.signals: list[dict] = []
        self.metrics: dict[str, Any] = {}
        self.error_count = 0
        self.last_error = None
        self.thinking: list[str] = []
        self.last_analysis_time = 0.0

    async def analyze(self, data: dict) -> dict:
        raise NotImplementedError(f"{self.name} agent must implement analyze()")

    def update_status(self, status: str, error: str = None):
        self.status = status
        self.last_update = time.time()
        if status == "running":
            self.last_analysis_time = time.time()
        if error:
            self.error_count += 1
            self.last_error = error

    def get_status(self) -> dict:
        elapsed = 0.0
        if self.status == "running" and self.last_analysis_time > 0:
            elapsed = time.time() - self.last_analysis_time

        return {
            "name": self.name,
            "status": self.status,
            "last_update": self.last_update,
            "signal_count": len(self.signals),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "thinking": self.thinking,
            "elapsed_seconds": round(elapsed, 1),
        }

    def reset(self):
        self.signals = []
        self.metrics = {}
        self.error_count = 0
        self.last_error = None
        self.thinking = []

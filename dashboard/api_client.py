"""Read-only client for the authoritative inspection API, plus audited control."""

from typing import Any, Dict, Optional

import requests


class InspectionApiClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def get(
        self,
        path: str,
        *,
        accepted_statuses: tuple[int, ...] = (200,),
        **params: Any,
    ) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params or None,
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code not in accepted_statuses:
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {path}")
        return payload

    def post(
        self,
        path: str,
        *,
        accepted_statuses: tuple[int, ...] = (200, 202),
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{path}",
            data=data or None,
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code not in accepted_statuses:
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {path}")
        payload["_http_status"] = response.status_code
        return payload

    def health(self) -> Dict[str, Any]:
        # HTTP 503 is an authoritative DEGRADED readiness payload, not an
        # unreachable API. The dashboard must display it instead of fabricating fallback state.
        return self.get("/health", accepted_statuses=(200, 503))

    def roll_map(self, roll_id: str) -> Dict[str, Any]:
        return self.get(f"/api/roll/{roll_id}/map")

    def active_roll(self) -> Dict[str, Any]:
        return self.get("/api/roll/active")

    def batch_stats(self, batch_id: str) -> Dict[str, Any]:
        return self.get(f"/api/batch/{batch_id}/stats")

    def batch_spc(self, batch_id: str) -> Dict[str, Any]:
        return self.get(f"/api/batch/{batch_id}/spc")

    def industrial_state(self) -> Dict[str, Any]:
        return self.get("/api/industrial/state")

    def operations_snapshot(self, signal_limit: int = 25) -> Dict[str, Any]:
        if not 1 <= signal_limit <= 100:
            raise ValueError("signal_limit must be between 1 and 100")
        return self.get("/api/operations/snapshot", signal_limit=signal_limit)

    def operations_control(
        self,
        action: str,
        operator_id: str,
        confirmation: str,
        reason: str,
        snapshot_id: str = "",
    ) -> Dict[str, Any]:
        return self.post(
            "/api/operations/control",
            data={
                "action": action,
                "operator_id": operator_id,
                "confirmation": confirmation,
                "reason": reason,
                "snapshot_id": snapshot_id,
            },
            accepted_statuses=(200, 202, 409, 503),
        )

    def passport(self) -> Dict[str, Any]:
        return self.get("/api/spc/passport")

    def certificate(self, roll_id: str, batch_id: str) -> Dict[str, Any]:
        return self.get(f"/api/roll/{roll_id}/certificate", batch_id=batch_id)

    def libad_protocol(self) -> Dict[str, Any]:
        return self.get("/api/libad/protocol")

    def libad_demo(self, case_id: int) -> Dict[str, Any]:
        if case_id not in {1, 2, 3, 4}:
            raise ValueError("LIBAD demo case must be 1, 2, 3, or 4")
        return self.get(f"/api/libad/demo/{case_id}")

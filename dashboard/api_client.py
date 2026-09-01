"""Read-only client for the authoritative inspection API, plus audited control."""

import os
import re
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


class InspectionApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 5.0,
        snapshot_timeout_seconds: float = 45.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.snapshot_timeout_seconds = snapshot_timeout_seconds
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def get(
        self,
        path: str,
        *,
        accepted_statuses: tuple[int, ...] = (200,),
        timeout_seconds: Optional[float] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}",
            params=params or None,
            headers=self.headers,
            timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        if response.status_code not in accepted_statuses:
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {path}")
        return payload

    def get_bytes(self, path: str, *, max_bytes: int = 3 * 1024 * 1024) -> bytes:
        response = self.session.get(
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=self.timeout_seconds,
            stream=True,
        )
        try:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type != "image/jpeg":
                raise ValueError(f"Expected image/jpeg from {path}")
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise ValueError("Inspection artifact exceeds the dashboard limit")
            if not payload:
                raise ValueError("Inspection artifact is empty")
            return bytes(payload)
        finally:
            response.close()

    def post(
        self,
        path: str,
        *,
        accepted_statuses: tuple[int, ...] = (200, 202),
        data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        headers = dict(self.headers)
        headers.update(extra_headers or {})
        response = self.session.post(
            f"{self.base_url}{path}",
            data=data or None,
            headers=headers,
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

    def operations_snapshot(
        self,
        signal_limit: int = 25,
        *,
        timeout_seconds: Optional[float] = None,
        scope: str = "full",
    ) -> Dict[str, Any]:
        if not 1 <= signal_limit <= 100:
            raise ValueError("signal_limit must be between 1 and 100")
        if scope not in {"full", "live"}:
            raise ValueError("snapshot scope must be full or live")
        params: Dict[str, Any] = {"signal_limit": signal_limit}
        if scope != "full":
            params["scope"] = scope
        return self.get(
            "/api/operations/snapshot",
            timeout_seconds=(
                self.snapshot_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
            **params,
        )

    def inspection_image(self, run_id: str, view: str = "overlay") -> bytes:
        if not run_id.startswith("RUN_") or len(run_id) != 16:
            raise ValueError("Invalid inspection run identifier")
        if view not in {"raw", "mask", "blend", "heatmap", "input", "overlay"}:
            raise ValueError("Invalid inspection artifact view")
        return self.get_bytes(f"/api/inspections/{run_id}/image?view={view}")

    def dataset_catalog(
        self,
        offset: int = 0,
        limit: int = 24,
        class_flag: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not 0 <= offset <= 10_000:
            raise ValueError("dataset catalog offset must be between 0 and 10000")
        if not 1 <= limit <= 100:
            raise ValueError("dataset catalog limit must be between 1 and 100")
        params: Dict[str, Any] = {"offset": offset, "limit": limit}
        if class_flag:
            params["class_flag"] = class_flag
        return self.get("/api/dataset/catalog", **params)

    def dataset_image(self, filename: str, view: str = "original") -> bytes:
        if os.path.basename(filename) != filename or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", filename or ""
        ):
            raise ValueError("Invalid dataset image basename")
        if view not in {"original", "mask", "blend", "heatmap", "thumb"}:
            raise ValueError("Invalid dataset evidence view")
        suffix = "" if view == "original" else f"?view={view}"
        return self.get_bytes(
            f"/api/dataset/images/{quote(filename, safe='')}{suffix}"
        )

    def operations_control(
        self,
        action: str,
        operator_id: str,
        confirmation: str,
        reason: str,
        idempotency_key: str,
        snapshot_id: str = "",
        operator_token: str = "",
    ) -> Dict[str, Any]:
        if not idempotency_key:
            raise ValueError("idempotency_key is required for control commands")
        return self.post(
            "/api/operations/control",
            data={
                "action": action,
                "operator_id": operator_id,
                "confirmation": confirmation,
                "reason": reason,
                "snapshot_id": snapshot_id,
                "idempotency_key": idempotency_key,
            },
            extra_headers={"x-operator-token": operator_token} if operator_token else None,
            accepted_statuses=(200, 202, 409, 503),
        )

    def passport(self) -> Dict[str, Any]:
        return self.get("/api/spc/passport")

    def certificate(self, roll_id: str, batch_id: str) -> Dict[str, Any]:
        return self.get(f"/api/roll/{roll_id}/certificate", batch_id=batch_id)

    def libad_protocol(self) -> Dict[str, Any]:
        return self.get("/api/libad/protocol")

    def libad_samples(self, offset: int = 0, limit: int = 12) -> Dict[str, Any]:
        if not 0 <= offset <= 10_000:
            raise ValueError("multimodal sample offset must be between 0 and 10000")
        if not 1 <= limit <= 50:
            raise ValueError("multimodal sample limit must be between 1 and 50")
        return self.get("/api/libad/samples", offset=offset, limit=limit)

    def libad_sample_image(self, sample_id: str, view: str = "vis_a") -> bytes:
        if os.path.basename(sample_id) != sample_id or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", sample_id or ""
        ):
            raise ValueError("Invalid multimodal sample identifier")
        if view not in {"vis_a", "vis_b", "xray_l"}:
            raise ValueError("Invalid multimodal view")
        return self.get_bytes(
            f"/api/libad/samples/{quote(sample_id, safe='')}/image?view={view}"
        )

    def libad_demo(self, case_id: int) -> Dict[str, Any]:
        if case_id not in {1, 2, 3, 4}:
            raise ValueError("LIBAD demo case must be 1, 2, 3, or 4")
        return self.get(f"/api/libad/demo/{case_id}")

"""
SecureCoating-Vision: Digital Twin Roll Quality Certificate & Cryptographic Audit Trail
========================================================================================
Compiles roll inspection records into a cryptographically tamper-evident
quality manifest. Regulatory/customer compliance requires external qualification.

Key Features:
- Complete 2D Roll Defect Coordinates (Linear meter X vs Cross-web mm Y)
- Standards compliance summary (Plant Electrode QA Specification & GB/T 38031 Safety Baseline)
- Micro-short circuit hazard risk distribution across all slitting lanes
- Tamper-Evident SHA-256 Defect Record with Factory-Authenticated HMAC Signature
- Exportable to Markdown, JSON, and printable inspection reports
"""

import json
import hmac
import hashlib
import logging
import os
import secrets
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("SecureCoatingVision.RollCert")

# Factory signing secret; production must provide it through a secret manager or environment.
_configured_secret = os.environ.get("SECURECOATING_FACTORY_SECRET", "").encode("utf-8")
DEFAULT_FACTORY_SECRET_KEY = _configured_secret or secrets.token_bytes(32)


@dataclass
class DigitalRollCertificate:
    """Official Battery Electrode Quality Inspection Certificate."""
    certificate_id: str
    roll_id: str
    batch_id: str
    electrode_type: str
    substrate_material: str
    total_length_m: float
    web_width_mm: float
    
    # Statistical Yield & Quality
    total_scanned_length_m: float
    pass_rate_pct: float
    total_defects_count: int
    defect_density_per_100m: float
    overall_quality_grade: str          # 'GRADE_A_PRIME', 'GRADE_B_REWORK', 'GRADE_C_SCRAP'
    standards_compliant: bool
    
    # Defect Breakdown by Class & Slitting Lane
    defects_by_class: Dict[str, int]
    defects_by_lane: Dict[int, int]
    
    # Spatial Defect Map
    defect_map_entries: List[Dict[str, Any]]
    
    # Process Diagnostic Summary
    spc_status: str
    primary_root_cause_summary: str
    quality_metrics_provisional: bool
    metric_provenance: str
    
    # Cryptographic Tamper-Evident Verification
    issued_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    payload_hash_sha256: str = ""
    hmac_digital_signature: str = ""
    signature_algorithm: str = "HMAC-SHA256"

    def _signature_payload(self) -> Dict[str, Any]:
        """Return the complete immutable certificate payload, excluding crypto fields."""
        return {
            "certificate_id": self.certificate_id,
            "roll_id": self.roll_id,
            "batch_id": self.batch_id,
            "electrode_type": self.electrode_type,
            "substrate_material": self.substrate_material,
            "total_length_m": self.total_length_m,
            "web_width_mm": self.web_width_mm,
            "total_scanned_length_m": self.total_scanned_length_m,
            "pass_rate_pct": self.pass_rate_pct,
            "total_defects_count": self.total_defects_count,
            "defect_density_per_100m": self.defect_density_per_100m,
            "overall_quality_grade": self.overall_quality_grade,
            "standards_compliant": self.standards_compliant,
            "defects_by_class": self.defects_by_class,
            "defects_by_lane": self.defects_by_lane,
            "defect_map_entries": self.defect_map_entries,
            "spc_status": self.spc_status,
            "primary_root_cause_summary": self.primary_root_cause_summary,
            "quality_metrics_provisional": self.quality_metrics_provisional,
            "metric_provenance": self.metric_provenance,
            "issued_at": self.issued_at,
            "signature_algorithm": self.signature_algorithm,
        }

    def generate_cryptographic_signature(self, secret_key: bytes = DEFAULT_FACTORY_SECRET_KEY) -> str:
        """Compute a canonical hash and HMAC over every certificate field."""
        raw_bytes = json.dumps(
            self._signature_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.payload_hash_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        
        # Cryptographic HMAC Digital Signature
        self.hmac_digital_signature = hmac.new(secret_key, raw_bytes, hashlib.sha256).hexdigest()
        return self.hmac_digital_signature

    def verify_signature(self, secret_key: bytes = DEFAULT_FACTORY_SECRET_KEY) -> bool:
        """Verify the cryptographic authenticity of the certificate."""
        raw_bytes = json.dumps(
            self._signature_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected_hash = hashlib.sha256(raw_bytes).hexdigest()
        expected_sig = hmac.new(secret_key, raw_bytes, hashlib.sha256).hexdigest()
        return (
            hmac.compare_digest(self.payload_hash_sha256, expected_hash)
            and hmac.compare_digest(self.hmac_digital_signature, expected_sig)
        )

    def to_dict(self) -> Dict[str, Any]:
        if not self.hmac_digital_signature:
            self.generate_cryptographic_signature()
        payload = copy.deepcopy(self._signature_payload())
        payload.update({
            "payload_hash_sha256": self.payload_hash_sha256,
            "hmac_digital_signature": self.hmac_digital_signature,
        })
        return payload

    def to_markdown(self) -> str:
        if not self.hmac_digital_signature:
            self.generate_cryptographic_signature()
        
        md = f"""# 📜 Battery Electrode Quality Inspection Certificate (Plant QA Specification & GB/T 38031)

**Certificate ID:** `{self.certificate_id}`  
**Issue Timestamp:** {self.issued_at}  
**Payload SHA-256 Digest:** `{self.payload_hash_sha256}`  
**Factory HMAC Authenticated Signature:** `{self.hmac_digital_signature}` (`{self.signature_algorithm}`)  

---

## 1. Jumbo Roll Specification
| Parameter | Value |
| :--- | :--- |
| **Roll Identifier** | `{self.roll_id}` |
| **Manufacturing Batch ID** | `{self.batch_id}` |
| **Electrode Chemistry** | `{self.electrode_type}` |
| **Current Collector Substrate** | `{self.substrate_material}` |
| **Total Roll Length** | {self.total_length_m:.1f} m |
| **Web Width** | {self.web_width_mm:.1f} mm |
| **Inspected Length** | {self.total_scanned_length_m:.1f} m |

---

## 2. Quality Evaluation & Yield Summary
* **Overall Quality Verdict:** **`{self.overall_quality_grade}`**
* **Configured Prototype Policy Result:** **{'PASS' if self.standards_compliant else 'NOT VERIFIED / POLICY VIOLATION'}**
* **Batch Pass Rate:** **{self.pass_rate_pct:.1f}%**
* **Total Defects Logged:** {self.total_defects_count}
* **Defect Density:** {self.defect_density_per_100m:.2f} defects / 100m
* **Process SPC Status:** `{self.spc_status}`

### Defect Distribution by Class
"""
        for cls_name, cnt in self.defects_by_class.items():
            md += f"- **{cls_name.capitalize()}:** {cnt} occurrences\n"

        md += "\n### Defect Distribution by Slitting Lane\n"
        for lane_idx, cnt in self.defects_by_lane.items():
            md += f"- **Lane {lane_idx} (Width ~162.5mm):** {cnt} defect(s)\n"

        md += f"""
---

## 3. Upstream Diagnostic Summary
* **Attributed Root Cause:** {self.primary_root_cause_summary}

*Certified by SecureCoating-Vision Automated Metrology System. Authenticated with a factory HMAC-SHA256 secret.*
"""
        return md


class RollCertificateGenerator:
    """Factory for compiling digital certificates from inspection telemetry."""

    @staticmethod
    def build_certificate(
        roll_id: str,
        batch_id: str,
        inspected_length_m: float,
        total_length_m: float,
        defect_records: List[Dict[str, Any]],
        spc_status: str = "IN CONTROL",
        root_cause_summary: str = "Nominal",
        electrode_type: str = "Cathode_LFP",
        substrate_material: str = "Aluminum_Foil_13um",
        web_width_mm: float = 650.0
        ,
        total_inspections: Optional[int] = None,
        failed_inspections: Optional[int] = None,
        metric_provenance: str = "UNSPECIFIED",
    ) -> DigitalRollCertificate:
        """Compile inspection findings into a cryptographically verified certificate."""
        total_defects = len(defect_records)
        
        metrics_provisional = total_inspections is None or failed_inspections is None
        if metrics_provisional:
            pass_rate = 0.0
        else:
            if total_inspections < 0 or failed_inspections < 0 or failed_inspections > total_inspections:
                raise ValueError("Invalid inspection counts for certificate")
            pass_rate = (
                0.0 if total_inspections == 0 else
                ((total_inspections - failed_inspections) / total_inspections) * 100.0
            )

        by_class: Dict[str, int] = {}
        by_lane = {1: 0, 2: 0, 3: 0, 4: 0}
        
        std_ok = True
        for d in defect_records:
            c_name = d.get("class_name", "unknown")
            by_class[c_name] = by_class.get(c_name, 0) + 1
            
            lane = d.get("lane_id", 1)
            by_lane[lane] = by_lane.get(lane, 0) + 1
            
            if d.get("class_name") == "delamination" or d.get("peak_height_um", 0.0) >= 12.0:
                std_ok = False

        defect_density = (total_defects / max(1.0, inspected_length_m)) * 100.0

        if metrics_provisional:
            grade = "UNVERIFIED"
            std_ok = False
        elif pass_rate >= 98.0 and std_ok and total_defects <= 5:
            grade = "GRADE_A_PRIME"
        elif pass_rate >= 90.0:
            grade = "GRADE_B_REWORK"
        else:
            grade = "GRADE_C_SCRAP"

        cert_id = f"CERT_{roll_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cert = DigitalRollCertificate(
            certificate_id=cert_id,
            roll_id=roll_id,
            batch_id=batch_id,
            electrode_type=electrode_type,
            substrate_material=substrate_material,
            total_length_m=total_length_m,
            web_width_mm=web_width_mm,
            total_scanned_length_m=inspected_length_m,
            pass_rate_pct=pass_rate,
            total_defects_count=total_defects,
            defect_density_per_100m=defect_density,
            overall_quality_grade=grade,
            standards_compliant=std_ok,
            defects_by_class=by_class,
            defects_by_lane=by_lane,
            defect_map_entries=copy.deepcopy(list(defect_records)),
            spc_status=spc_status,
            primary_root_cause_summary=root_cause_summary,
            quality_metrics_provisional=metrics_provisional,
            metric_provenance=metric_provenance,
        )
        cert.generate_cryptographic_signature()
        return cert

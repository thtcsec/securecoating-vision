"""
SecureCoating-Vision: AI Closed-Loop Root-Cause Diagnostics & Equipment Feedback
================================================================================
Translates defect spatial patterns, physical morphology, and severity into actionable
closed-loop adjustments for upstream battery manufacturing equipment.

Equipment Mappings:
1. Slot-Die Coater / Doctor Blade Metrology Unit
2. High-Shear Planetary Slurry Mixer & Vacuum De-aerator
3. Multi-Zone Hot Air Floatation Drying Oven
4. Web Handling Unwinder & Tension Control Servo
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("SecureCoatingVision.RootCause")


@dataclass
class EquipmentAdjustment:
    """Actionable tuning parameter for upstream factory equipment."""
    equipment_unit: str       # e.g., 'Slot-Die Coater Lip', 'Drying Oven Zone 1', 'Mixer Vacuum'
    parameter_name: str       # e.g., 'Zone 1 Temperature', 'Doctor Blade Index', 'Vacuum Pressure'
    current_value: str
    suggested_delta: str      # e.g., '-3.5 °C', '+5.0 kPa', '+2.0 mm'
    urgency: str              # 'IMMEDIATE', 'NEXT_BATCH', 'PREVENTATIVE'
    rationale: str


@dataclass
class RootCauseReport:
    """Comprehensive diagnostic report attributing defect root causes."""
    primary_root_cause: str
    affected_equipment: str
    confidence_score: Optional[float]
    defect_signature: str
    severity_level: str       # 'CRITICAL', 'WARNING', 'NOMINAL'
    action_items: List[EquipmentAdjustment] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_root_cause": self.primary_root_cause,
            "affected_equipment": self.affected_equipment,
            "confidence_score": (
                round(self.confidence_score, 2) if self.confidence_score is not None else None
            ),
            "evidence_level": "HEURISTIC_RULE_MATCH",
            "defect_signature": self.defect_signature,
            "severity_level": self.severity_level,
            "action_items": [
                {
                    "equipment": a.equipment_unit,
                    "parameter": a.parameter_name,
                    "delta": a.suggested_delta,
                    "urgency": a.urgency,
                    "rationale": a.rationale,
                }
                for a in self.action_items
            ],
            "timestamp": self.timestamp,
        }


class RootCauseDiagnosticEngine:
    """
    Expert rule & statistical AI engine that attributes coating defects back to manufacturing causes.
    """

    def diagnose_batch(
        self,
        defects_metrology: List[Dict[str, Any]],
        historical_stats: Optional[Dict[str, Any]] = None,
        inspection_valid: bool = True,
        gate_action: Optional[str] = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
        hold_reasons: Optional[List[str]] = None,
    ) -> RootCauseReport:
        """
        Diagnose the primary root cause based on current frame and historical batch trends.
        """
        if not inspection_valid or str(gate_action or "").upper() == "HOLD":
            raw_n = len(raw_detections or [])
            metro_n = len(defects_metrology or [])
            reasons = "; ".join(hold_reasons or ["Inspection evidence is not safety-ready"])
            return RootCauseReport(
                primary_root_cause="HOLD: evidence or communication contract not satisfied",
                affected_equipment="No automatic process change authorized",
                confidence_score=None,
                defect_signature=(
                    f"Automatic release blocked. {reasons}. "
                    f"Raw detections present: {raw_n}; metrology objects: {metro_n}."
                ),
                severity_level="HOLD",
                action_items=[],
            )

        if not defects_metrology:
            return RootCauseReport(
                primary_root_cause="No defect evidence supplied",
                affected_equipment="Not assessed",
                confidence_score=None,
                defect_signature="Empty evidence set; process control cannot be inferred",
                severity_level="UNKNOWN",
                action_items=[]
            )

        # Count defect occurrences and characteristics
        scratch_count = sum(1 for d in defects_metrology if d.get("class_name") == "scratch")
        void_count = sum(1 for d in defects_metrology if d.get("class_name") == "void")
        blister_count = sum(1 for d in defects_metrology if d.get("class_name") == "blister")
        delam_count = sum(1 for d in defects_metrology if d.get("class_name") == "delamination")
        
        max_height = max([d.get("peak_height_um", 0.0) for d in defects_metrology], default=0.0)
        max_area = max([d.get("area_mm2", 0.0) for d in defects_metrology], default=0.0)

        # Case 1: Critical Delamination -> Drying Oven / Substrate Tension
        if delam_count > 0:
            return RootCauseReport(
                primary_root_cause="Solvent Skinning & Severe Interfacial De-adhesion",
                affected_equipment="Multi-Zone Floatation Drying Oven (Zone 1 & 2)",
                confidence_score=None,
                defect_signature=f"Delamination area {max_area:.2f} mm² with substrate decoupling",
                severity_level="CRITICAL",
                action_items=[
                    EquipmentAdjustment(
                        equipment_unit="Drying Oven Zone 1",
                        parameter_name="Zone 1 Temperature",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="IMMEDIATE",
                        rationale="Prevent rapid surface evaporation that traps solvent under impermeable skin"
                    ),
                    EquipmentAdjustment(
                        equipment_unit="Exhaust Damper",
                        parameter_name="Solvent Extraction Airflow",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="IMMEDIATE",
                        rationale="Accelerate boundary layer NMP vapor evacuation"
                    ),
                ]
            )

        # Case 2: Longitudinal Scratches -> Slot-Die / Doctor Blade
        if scratch_count >= max(void_count, blister_count):
            return RootCauseReport(
                primary_root_cause="Slurry Particle Agglomeration at Coating Die Lip",
                affected_equipment="Slot-Die Coater / Doctor Blade Assembly",
                confidence_score=None,
                defect_signature=f"{scratch_count} longitudinal streak(s) aligned with web travel",
                severity_level="CRITICAL" if scratch_count > 2 else "WARNING",
                action_items=[
                    EquipmentAdjustment(
                        equipment_unit="Slot-Die Lip",
                        parameter_name="Automatic Ultrasonic Wash Cycle",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="IMMEDIATE",
                        rationale="Dislodge agglomerated active material particulates from coater gap"
                    ),
                    EquipmentAdjustment(
                        equipment_unit="Doctor Blade Servo",
                        parameter_name="Lateral Blade Indexing",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="NEXT_BATCH",
                        rationale="Present fresh un-nicked blade surface to coating foil"
                    ),
                ]
            )

        # Case 3: Blisters / Severe Protrusions -> Oven Zone 1 / Separator Hazard
        if blister_count > 0 and max_height > 15.0:
            return RootCauseReport(
                primary_root_cause="Solvent Entrapment Micro-Explosion & Thermal Blistering",
                affected_equipment="Pre-Heating IR Zone & Air Knives",
                confidence_score=None,
                defect_signature=f"Raised blister bump (peak height {max_height:.1f} um > 14um separator barrier)",
                severity_level="CRITICAL",
                action_items=[
                    EquipmentAdjustment(
                        equipment_unit="IR Heating Array",
                        parameter_name="IR Lamp Radiant Intensity",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="IMMEDIATE",
                        rationale="Mitigate localized boiling of residual NMP/water solvent"
                    ),
                    EquipmentAdjustment(
                        equipment_unit="Web Tension Controller",
                        parameter_name="Foil Web Tension",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="PREVENTATIVE",
                        rationale="Eliminate foil flutter causing uneven coating thickness"
                    ),
                ]
            )

        # Case 4: Voids / Pinholes -> Slurry Degassing
        if void_count > 0:
            return RootCauseReport(
                primary_root_cause="Incomplete Slurry Vacuum Degassing & Micro-Bubble Ingestion",
                affected_equipment="Planetary Dual-Shaft Slurry Mixer & De-aerator",
                confidence_score=None,
                defect_signature=f"{void_count} sub-surface void(s) detected via lock-in thermography",
                severity_level="WARNING",
                action_items=[
                    EquipmentAdjustment(
                        equipment_unit="Slurry De-aerator",
                        parameter_name="Vacuum Tank Pressure",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="IMMEDIATE",
                        rationale="Eliminate micro-air bubble entrapment before slurry delivery to slot-die"
                    ),
                    EquipmentAdjustment(
                        equipment_unit="Slurry Delivery Pump",
                        parameter_name="Metering Pump Flow Rate Q",
                        current_value="NOT MEASURED",
                        suggested_delta="ENGINEERING REVIEW REQUIRED",
                        urgency="NEXT_BATCH",
                        rationale="Stabilize laminar delivery velocity to prevent cavitation"
                    ),
                ]
            )

        # Fallback nominal
        return RootCauseReport(
            primary_root_cause="Minor Surface Non-Uniformity",
            affected_equipment="Coating Line In Tolerances",
            confidence_score=None,
            defect_signature="Isolated low-severity anomaly",
            severity_level="NOMINAL",
            action_items=[]
        )

"""
SecureCoating-Vision: Industrial Latency Measurement & Camera-to-Ejector Distance Budget
========================================================================================
Monte Carlo latency assumptions and physical reject-distance modeling. These
defaults are not target-hardware measurements.

Physical Latency Chain:
1. Line-scan Image Formation: N_lines * t_line_period  (dynamically linked to optical engine)
2. Sensor Readout & FPGA Frame Grabber Buffer: t_fpga
3. Zero-Copy PCIe DMA Host Transfer: t_dma
4. Multi-Modal Spatial Warp & Fusion: t_fusion
5. Assumed AI inference stage: t_ai
6. Physics Metrology & Guard-Band Standards Audit: t_metrology
7. Industrial Ethernet (Modbus TCP / EtherCAT): t_network
8. PLC Input Update & Deterministic RPI Scan Cycle: t_plc
9. Solenoid Valve Magnetic Energization: t_solenoid
10. Pneumatic Air Chamber Pressure Build-up & Jet Propagation: t_pneumatic
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from industrial.optical_budget import OpticalThroughputBudgetEngine


class LatencyVerificationStatus(str, Enum):
    MONTE_CARLO_SIMULATED = "MONTE_CARLO_SIMULATED"
    HARDWARE_BENCH_MEASURED = "HARDWARE_BENCH_MEASURED"


@dataclass
class LatencyStageDistribution:
    """Modeled samples for one stage unless verification status says hardware measured."""
    stage_name: str
    samples_ms: np.ndarray = field(default_factory=lambda: np.zeros(100, dtype=np.float32))

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples_ms, 50.0))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples_ms, 95.0))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples_ms, 99.0))

    @property
    def p99_9(self) -> float:
        return float(np.percentile(self.samples_ms, 99.9))

    @property
    def sample_max(self) -> float:
        return float(np.max(self.samples_ms))


class LatencyBudgetEngine:
    """
    Computes statistical percentiles (P50, P95, P99, P99.9) across the complete physical reject chain.
    """

    def __init__(
        self,
        num_benchmark_samples: int = 10_000,
        optical_engine: Optional[OpticalThroughputBudgetEngine] = None,
        tile_lines: int = 512
    ):
        self.num_samples = num_benchmark_samples
        self.optical = optical_engine or OpticalThroughputBudgetEngine()
        self.tile_lines = tile_lines
        self.verification_status = LatencyVerificationStatus.MONTE_CARLO_SIMULATED
        self.stages: Dict[str, LatencyStageDistribution] = self._generate_distributions()

    def _generate_distributions(self, line_speed_m_s: float = 1.8) -> Dict[str, LatencyStageDistribution]:
        """
        Generate deterministic engineering-assumption distributions.
        Stage 1 tile formation latency is derived directly from optical line rate!
        """
        rng = np.random.default_rng(seed=42)
        
        # 1. Derive Tile Formation Latency from optical line rate: N_lines * line_period
        opt_res = self.optical.compute_budget(line_speed_m_s=line_speed_m_s)
        line_period_ms = opt_res["line_period_us"] / 1000.0
        tile_formation_mean_ms = self.tile_lines * line_period_ms
        
        stage_params = {
            f"1. Line-Scan Tile Formation ({self.tile_lines} lines)": (tile_formation_mean_ms, 0.35),
            "2. FPGA Buffer & Sensor Readout": (0.45, 0.05),
            "3. PCIe DMA Host Transfer (Zero-Copy)": (1.20, 0.15),
            "4. Multi-Modal Sensor Fusion": (1.80, 0.20),
            "5. Assumed AI Inference": (8.70, 0.60),
            "6. Metrology & Guard-Band Decision": (1.50, 0.18),
            "7. Industrial Network (EtherCAT/Modbus)": (0.80, 0.12),
            "8. PLC Scan Cycle (RPI 2.0ms)": (2.00, 0.35),
            "9. Solenoid Valve Coil Energization": (2.50, 0.25),
            "10. Pneumatic Air Jet Propagation": (2.00, 0.30),
        }
        
        distributions = {}
        for name, (mean_v, std_v) in stage_params.items():
            mu = np.log(mean_v**2 / np.sqrt(std_v**2 + mean_v**2))
            sigma = np.sqrt(np.log(1 + (std_v**2 / mean_v**2)))
            samples = rng.lognormal(mu, sigma, self.num_samples).astype(np.float32)
            distributions[name] = LatencyStageDistribution(stage_name=name, samples_ms=samples)
            
        return distributions

    def compute_budget(
        self,
        line_speed_m_s: float = 1.8,
        latency_guard_band_ms: float = 3.0,
        installed_ejector_distance_mm: float = 500.0
    ) -> Dict[str, Any]:
        """
        Compute end-to-end latency percentiles and physical actuator distance requirements.
        """
        # Re-derive distributions for target line speed
        self.stages = self._generate_distributions(line_speed_m_s=line_speed_m_s)
        
        total_samples_ms = np.zeros(self.num_samples, dtype=np.float32)
        stage_breakdown = {}
        
        for name, dist in self.stages.items():
            total_samples_ms += dist.samples_ms
            stage_breakdown[name] = {
                "p50_ms": round(dist.p50, 2),
                "p95_ms": round(dist.p95, 2),
                "p99_ms": round(dist.p99, 2),
                "p99_9_ms": round(dist.p99_9, 2),
            }

        p50_total = float(np.percentile(total_samples_ms, 50.0))
        p95_total = float(np.percentile(total_samples_ms, 95.0))
        p99_total = float(np.percentile(total_samples_ms, 99.0))
        p99_9_total = float(np.percentile(total_samples_ms, 99.9))
        observed_max = float(np.max(total_samples_ms))

        # Design Latency with Engineering Guard Band
        t_design_ms = p99_9_total + latency_guard_band_ms
        t_design_sec = t_design_ms / 1000.0

        min_required_distance_mm = line_speed_m_s * t_design_sec * 1000.0
        safety_margin_ratio = installed_ejector_distance_mm / max(1.0, min_required_distance_mm)

        return {
            "line_speed_m_s": line_speed_m_s,
            "sample_count": self.num_samples,
            "verification_status": self.verification_status.value,
            "is_monte_carlo_modeled": True,
            "is_hardware_bench_verified": False,
            "stage_breakdown": stage_breakdown,
            "p50_total_ms": round(p50_total, 2),
            "p95_total_ms": round(p95_total, 2),
            "p99_total_ms": round(p99_total, 2),
            "p99_9_total_ms": round(p99_9_total, 2),
            "observed_max_total_ms": round(observed_max, 2),
            "latency_guard_band_ms": latency_guard_band_ms,
            "t_design_total_ms": round(t_design_ms, 2),
            "min_required_ejector_distance_mm": round(min_required_distance_mm, 1),
            "installed_ejector_distance_mm": installed_ejector_distance_mm,
            "safety_margin_ratio": round(safety_margin_ratio, 2),
            "is_theoretical_budget_pass": (safety_margin_ratio >= 1.0),
            "engineering_verdict": f"Modeled only: configured ejector distance ({installed_ejector_distance_mm}mm) is {safety_margin_ratio:.1f}x the distance implied by the synthetic P99.9 latency ({t_design_ms:.1f}ms); hardware verification required."
        }

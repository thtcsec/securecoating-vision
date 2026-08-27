import os
import sqlite3
import datetime
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("SecureCoatingVision.QualityMemory")

class QualityMemory:
    """
    Quality Memory Module responsible for logging inspections to SQLite,
    calculating statistical trends (SPC), and providing traceability metrics.
    """
    def __init__(self, db_path="data/quality_history.db"):
        self.db_path = db_path
        self.healthy = True
        self.last_error = ""
        self._health_lock = threading.Lock()
        
        # Ensure data folder exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        logger.info(f"Quality Memory DB initialized at: {self.db_path}")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    has_defect INTEGER NOT NULL,
                    defect_class TEXT,
                    max_length_mm REAL,
                    max_area_mm2 REAL,
                    peak_height_um REAL,
                    latency_ms REAL,
                    fallback_active INTEGER,
                    model_version TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            # Lightweight migration for DBs created before model_version existed
            cursor.execute("PRAGMA table_info(inspections)")
            cols = {row[1] for row in cursor.fetchall()}
            if "model_version" not in cols:
                cursor.execute(
                    "ALTER TABLE inspections ADD COLUMN model_version TEXT"
                )
            if "run_id" not in cols:
                cursor.execute(
                    "ALTER TABLE inspections ADD COLUMN run_id TEXT"
                )
            for column_name, column_type in (
                ("roll_id", "TEXT"),
                ("gate_action", "TEXT"),
                ("system_state", "TEXT"),
                ("inspection_valid", "INTEGER"),
                ("error_reason", "TEXT"),
            ):
                if column_name not in cols:
                    cursor.execute(
                        f"ALTER TABLE inspections ADD COLUMN {column_name} {column_type}"
                    )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_batch_timestamp "
                "ON inspections(batch_id, timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_run_id "
                "ON inspections(run_id)"
            )
            conn.commit()

    def _set_health(self, healthy: bool, error: str = "") -> None:
        with self._health_lock:
            self.healthy = healthy
            self.last_error = error

    def add_entry(
        self,
        batch_id,
        part_id,
        has_defect,
        defect_class,
        max_length=0.0,
        max_area=0.0,
        peak_height=0.0,
        latency=0.0,
        fallback=False,
        model_version=None,
        run_id=None,
        roll_id=None,
        gate_action=None,
        system_state=None,
        inspection_valid=True,
        error_reason=None,
    ) -> bool:
        """Logs inspection entry to database."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        fallback_val = 1 if fallback else 0
        has_defect_val = 1 if has_defect else 0

        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO inspections (
                        batch_id, part_id, has_defect, defect_class,
                        max_length_mm, max_area_mm2, peak_height_um,
                        latency_ms, fallback_active, model_version, run_id, timestamp,
                        roll_id, gate_action, system_state, inspection_valid, error_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        part_id,
                        has_defect_val,
                        defect_class,
                        max_length,
                        max_area,
                        peak_height,
                        latency,
                        fallback_val,
                        model_version,
                        run_id,
                        timestamp,
                        roll_id,
                        gate_action,
                        system_state,
                        1 if inspection_valid else 0,
                        error_reason,
                    ),
                )
                conn.commit()
            self._set_health(True)
            return True
        except sqlite3.Error as e:
            logger.error(f"Error logging to quality database: {e}")
            self._set_health(False, str(e))
            return False

    def update_decision(self, run_id: str, gate_action: str, system_state: str) -> bool:
        """Finalize the decision fields for a previously persisted inspection."""
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    "UPDATE inspections SET gate_action = ?, system_state = ? WHERE run_id = ?",
                    (gate_action, system_state, run_id),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.IntegrityError(f"Expected one inspection for run_id={run_id}")
                conn.commit()
            self._set_health(True)
            return True
        except sqlite3.Error as e:
            logger.error(f"Error finalizing quality decision: {e}")
            self._set_health(False, str(e))
            return False

    def get_batch_stats(self, batch_id):
        """Returns aggregated quality metrics for a given batch."""
        query_stats = """
            SELECT 
                COUNT(*), 
                SUM(CASE WHEN gate_action = 'REJECT' THEN 1 ELSE 0 END),
                SUM(CASE WHEN gate_action = 'PASS' THEN 1 ELSE 0 END),
                SUM(CASE WHEN gate_action = 'HOLD' OR inspection_valid = 0 THEN 1 ELSE 0 END),
                AVG(latency_ms)
            FROM inspections
            WHERE batch_id = ?
        """
        query_defects = """
            SELECT defect_class, COUNT(*)
            FROM inspections
            WHERE batch_id = ? AND has_defect = 1
            GROUP BY defect_class
        """
        
        stats = {"status": "OK", "total": 0, "failed": 0, "passed": 0, "pass_rate": None, "avg_latency_ms": None, "defect_distribution": {}}
        
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                
                # Overall counts
                cursor.execute(query_stats, (batch_id,))
                total, failed, passed, held, avg_latency = cursor.fetchone()
                if total and total > 0:
                    stats["total"] = total
                    stats["failed"] = failed or 0
                    stats["passed"] = passed or 0
                    stats["held"] = held or 0
                    resolved = stats["passed"] + stats["failed"]
                    stats["pass_rate"] = round((stats["passed"] / resolved) * 100.0, 2) if resolved else None
                    stats["avg_latency_ms"] = round(avg_latency or 0.0, 2)
                
                # Defect distributions
                cursor.execute(query_defects, (batch_id,))
                for row in cursor.fetchall():
                    stats["defect_distribution"][row[0]] = row[1]
                self._set_health(True)
                    
        except sqlite3.Error as e:
            logger.error(f"Error fetching batch stats: {e}")
            self._set_health(False, str(e))
            stats.update({"status": "ERROR", "error": str(e)})
            
        return stats

    def check_spc_alarms(self, batch_id, rolling_window=50):
        """
        Evaluates Statistical Process Control rules to flag process deviations:
        - If defect rate over the last N parts exceeds 10% (Critical Alert).
        - If recurrent defect coordinates suggest a blocked nozzle.
        """
        query = """
            SELECT has_defect 
            FROM inspections 
            WHERE batch_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        alarms = {
            "status": "NO DATA",
            "message": "No inspection data is available for this batch.",
            "trigger_nozzle_purge": False
        }
        
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (batch_id, rolling_window))
                results = [r[0] for r in cursor.fetchall()]
                
                if len(results) > 0:
                    alarms["status"] = "IN CONTROL"
                    alarms["message"] = "Observed defect rate is within the configured threshold."
                    defect_rate = sum(results) / len(results)
                    if defect_rate > 0.10:
                        alarms["status"] = "OUT OF CONTROL"
                        alarms["message"] = f"CRITICAL: Defect rate is {defect_rate*100:.1f}% in the last {len(results)} parts!"
                        alarms["trigger_nozzle_purge"] = True
                    elif defect_rate > 0.05:
                        alarms["status"] = "WARNING"
                        alarms["message"] = f"WARNING: Defect rate is elevated ({defect_rate*100:.1f}%) in current window."
                self._set_health(True)
                        
        except sqlite3.Error as e:
            logger.error(f"SPC evaluation failed: {e}")
            self._set_health(False, str(e))
            alarms.update({
                "status": "UNKNOWN",
                "message": "SPC state is unavailable because traceability storage failed.",
                "error": str(e),
            })
            
        return alarms

    def get_latency_stats(self, batch_id=None, limit=200):
        """Return latency percentiles for dashboard / competition metrics."""
        if batch_id:
            query = """
                SELECT latency_ms FROM inspections
                WHERE batch_id = ? AND latency_ms IS NOT NULL
                ORDER BY id DESC LIMIT ?
            """
            params = (batch_id, limit)
        else:
            query = """
                SELECT latency_ms FROM inspections
                WHERE latency_ms IS NOT NULL
                ORDER BY id DESC LIMIT ?
            """
            params = (limit,)

        values = []
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                values = [float(r[0]) for r in cursor.fetchall() if r[0] is not None]
        except sqlite3.Error as e:
            logger.error(f"Latency stats failed: {e}")
            self._set_health(False, str(e))
            return {
                "status": "ERROR",
                "error": str(e),
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
                "mean_ms": None,
                "max_ms": None,
                "target_ms": 35.0,
                "within_target_pct": None,
            }

        if not values:
            return {
                "status": "NO DATA",
                "count": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "mean_ms": 0.0,
                "max_ms": 0.0,
                "target_ms": 35.0,
                "within_target_pct": 0.0,
            }

        values.sort()
        n = len(values)

        def pct(p):
            idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            return round(values[idx], 2)

        within = sum(1 for v in values if v <= 35.0) / n * 100.0
        return {
            "status": "OK",
            "count": n,
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "mean_ms": round(sum(values) / n, 2),
            "max_ms": round(values[-1], 2),
            "target_ms": 35.0,
            "within_target_pct": round(within, 1),
        }

import os
import json
import sqlite3
import datetime
import logging
import threading
import tempfile
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
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_inspections_run_id_unique "
                "ON inspections(run_id) WHERE run_id IS NOT NULL"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspections_roll_batch_timestamp "
                "ON inspections(roll_id, batch_id, timestamp)"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inspection_identity_claims (
                    batch_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    PRIMARY KEY (batch_id, part_id)
                )
                """
            )
            # Preserve existing identities without rewriting historical rows.
            cursor.execute(
                """
                INSERT OR IGNORE INTO inspection_identity_claims(batch_id, part_id, claimed_at)
                SELECT batch_id, part_id, MIN(timestamp)
                FROM inspections
                GROUP BY batch_id, part_id
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS control_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    confirmation TEXT NOT NULL,
                    snapshot_id TEXT,
                    idempotency_key TEXT,
                    result_status TEXT NOT NULL,
                    signal_id TEXT,
                    details_json TEXT NOT NULL
                )
                """
            )
            cursor.execute("PRAGMA table_info(control_audit)")
            control_cols = {row[1] for row in cursor.fetchall()}
            if "idempotency_key" not in control_cols:
                cursor.execute("ALTER TABLE control_audit ADD COLUMN idempotency_key TEXT")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_control_audit_timestamp "
                "ON control_audit(timestamp)"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_control_audit_idempotency_unique "
                "ON control_audit(idempotency_key) WHERE idempotency_key IS NOT NULL"
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
                conn.execute("BEGIN IMMEDIATE")
                cursor.execute(
                    """
                    INSERT INTO inspection_identity_claims(batch_id, part_id, claimed_at)
                    VALUES (?, ?, ?)
                    """,
                    (batch_id, part_id, timestamp),
                )
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
        except sqlite3.IntegrityError as e:
            if "inspection_identity_claims" in str(e):
                message = f"Duplicate inspection identity rejected: {batch_id}/{part_id}"
                logger.warning(message)
                # The storage engine remains healthy; this is an input/idempotency violation.
                self._set_health(True, message)
                return False
            logger.error(f"Integrity error logging to quality database: {e}")
            self._set_health(False, str(e))
            return False
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

    def backup_to(self, destination_path: str) -> bool:
        """Create an atomic SQLite online backup without copying live WAL files."""
        destination = os.path.abspath(destination_path)
        source = os.path.abspath(self.db_path)
        if destination == source:
            raise ValueError("Backup destination must differ from the live database")
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".quality-backup-", suffix=".db", dir=destination_dir
        )
        os.close(fd)
        try:
            with self._connection() as source_conn:
                backup_conn = sqlite3.connect(temporary_path)
                try:
                    source_conn.backup(backup_conn)
                    integrity = backup_conn.execute("PRAGMA integrity_check").fetchone()
                    if not integrity or integrity[0] != "ok":
                        raise sqlite3.DatabaseError(
                            f"Backup integrity_check failed: {integrity}"
                        )
                    backup_conn.commit()
                finally:
                    backup_conn.close()
            os.replace(temporary_path, destination)
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.error(f"Quality database backup failed: {exc}")
            self._set_health(False, str(exc))
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
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

    def get_recent_inspections(self, batch_id, limit=20):
        """Return recent inspection rows for the operations snapshot. Empty is not an error."""
        payload = {"status": "OK", "batch_id": batch_id, "count": 0, "records": []}
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT timestamp, part_id, run_id, gate_action, system_state,
                           defect_class, latency_ms, inspection_valid, error_reason
                    FROM inspections
                    WHERE batch_id = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (batch_id, limit),
                )
                records = []
                for row in cursor.fetchall():
                    records.append(
                        {
                            "timestamp": row[0],
                            "part_id": row[1],
                            "run_id": row[2],
                            "gate_action": row[3],
                            "system_state": row[4],
                            "defect_class": row[5],
                            "latency_ms": row[6],
                            "inspection_valid": bool(row[7]) if row[7] is not None else None,
                            "error_reason": row[8],
                        }
                    )
                payload["count"] = len(records)
                payload["records"] = records
                self._set_health(True)
        except sqlite3.Error as exc:
            logger.error(f"Recent inspection query failed: {exc}")
            self._set_health(False, str(exc))
            return {
                "status": "ERROR",
                "batch_id": batch_id,
                "count": 0,
                "records": [],
                "error": str(exc),
            }
        return payload

    def record_control_audit(
        self,
        audit_id,
        operator_id,
        action,
        reason,
        confirmation,
        snapshot_id=None,
        idempotency_key=None,
        result_status="DISPATCHING",
        signal_id=None,
        details=None,
    ) -> bool:
        """Persist a control-audit row before a command is dispatched."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO control_audit (
                        audit_id, timestamp, operator_id, action, reason, confirmation,
                        snapshot_id, idempotency_key, result_status, signal_id, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        timestamp,
                        operator_id,
                        action,
                        reason,
                        confirmation,
                        snapshot_id,
                        idempotency_key,
                        result_status,
                        signal_id,
                        json.dumps(details or {}, sort_keys=True),
                    ),
                )
                conn.commit()
            self._set_health(True)
            return True
        except sqlite3.IntegrityError as exc:
            logger.warning(f"Duplicate control idempotency key rejected: {exc}")
            self._set_health(True, "Duplicate control idempotency key rejected")
            return False
        except sqlite3.Error as exc:
            logger.error(f"Control audit write failed: {exc}")
            self._set_health(False, str(exc))
            return False

    def get_control_audit_by_idempotency(self, idempotency_key):
        """Return a prior command result so transport retries never redispatch it."""
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT audit_id, operator_id, action, snapshot_id, result_status,
                           signal_id, details_json
                    FROM control_audit
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
            self._set_health(True)
            if row is None:
                return None
            return {
                "audit_id": row[0],
                "operator_id": row[1],
                "action": row[2],
                "snapshot_id": row[3],
                "result_status": row[4],
                "signal_id": row[5],
                "details": json.loads(row[6] or "{}"),
            }
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            logger.error(f"Control audit idempotency lookup failed: {exc}")
            self._set_health(False, str(exc))
            raise

    def finalize_control_audit(self, audit_id, result_status, signal_id=None, details=None) -> bool:
        """Update the audit row after command dispatch. Failure is reported, never hidden."""
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                if details is None:
                    cursor.execute(
                        """
                        UPDATE control_audit
                        SET result_status = ?, signal_id = COALESCE(?, signal_id)
                        WHERE audit_id = ?
                        """,
                        (result_status, signal_id, audit_id),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE control_audit
                        SET result_status = ?, signal_id = COALESCE(?, signal_id),
                            details_json = ?
                        WHERE audit_id = ?
                        """,
                        (
                            result_status,
                            signal_id,
                            json.dumps(details, sort_keys=True),
                            audit_id,
                        ),
                    )
                if cursor.rowcount != 1:
                    raise sqlite3.DatabaseError(f"Control audit {audit_id} was not updated")
                conn.commit()
            self._set_health(True)
            return True
        except sqlite3.Error as exc:
            logger.error(f"Control audit finalization failed: {exc}")
            self._set_health(False, str(exc))
            return False

    def get_recent_control_audits(self, limit=25):
        """Return recent operator control audits for the operations snapshot."""
        payload = {"status": "OK", "count": 0, "records": []}
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT audit_id, timestamp, operator_id, action, reason, confirmation,
                           snapshot_id, idempotency_key, result_status, signal_id, details_json
                    FROM control_audit
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                records = []
                for row in cursor.fetchall():
                    try:
                        details = json.loads(row[10] or "{}")
                    except json.JSONDecodeError:
                        details = {"error": "audit details_json was not valid JSON"}
                    records.append(
                        {
                            "audit_id": row[0],
                            "timestamp": row[1],
                            "operator_id": row[2],
                            "action": row[3],
                            "reason": row[4],
                            "confirmation": row[5],
                            "snapshot_id": row[6],
                            "idempotency_key": row[7],
                            "result_status": row[8],
                            "signal_id": row[9],
                            "details": details,
                        }
                    )
                payload["count"] = len(records)
                payload["records"] = records
                self._set_health(True)
        except sqlite3.Error as exc:
            logger.error(f"Control audit query failed: {exc}")
            self._set_health(False, str(exc))
            return {
                "status": "ERROR",
                "count": 0,
                "records": [],
                "error": str(exc),
            }
        return payload

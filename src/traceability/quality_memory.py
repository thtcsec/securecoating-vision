import os
import sqlite3
import datetime
import logging

logger = logging.getLogger("SecureCoatingVision.QualityMemory")

class QualityMemory:
    """
    Quality Memory Module responsible for logging inspections to SQLite,
    calculating statistical trends (SPC), and providing traceability metrics.
    """
    def __init__(self, db_path="data/quality_history.db"):
        self.db_path = db_path
        
        # Ensure data folder exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        logger.info(f"Quality Memory DB initialized at: {self.db_path}")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
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
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_entry(self, batch_id, part_id, has_defect, defect_class, max_length=0.0, max_area=0.0, peak_height=0.0, latency=0.0, fallback=False):
        """Logs inspection entry to database."""
        timestamp = datetime.datetime.now().isoformat()
        fallback_val = 1 if fallback else 0
        has_defect_val = 1 if has_defect else 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO inspections (
                        batch_id, part_id, has_defect, defect_class, 
                        max_length_mm, max_area_mm2, peak_height_um, 
                        latency_ms, fallback_active, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (batch_id, part_id, has_defect_val, defect_class, max_length, max_area, peak_height, latency, fallback_val, timestamp))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error logging to quality database: {e}")

    def get_batch_stats(self, batch_id):
        """Returns aggregated quality metrics for a given batch."""
        query_stats = """
            SELECT 
                COUNT(*), 
                SUM(has_defect),
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
        
        stats = {"total": 0, "failed": 0, "passed": 0, "pass_rate": 100.0, "avg_latency_ms": 0.0, "defect_distribution": {}}
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Overall counts
                cursor.execute(query_stats, (batch_id,))
                total, failed, avg_latency = cursor.fetchone()
                if total and total > 0:
                    stats["total"] = total
                    stats["failed"] = failed or 0
                    stats["passed"] = total - (failed or 0)
                    stats["pass_rate"] = round((stats["passed"] / total) * 100.0, 2)
                    stats["avg_latency_ms"] = round(avg_latency or 0.0, 2)
                
                # Defect distributions
                cursor.execute(query_defects, (batch_id,))
                for row in cursor.fetchall():
                    stats["defect_distribution"][row[0]] = row[1]
                    
        except sqlite3.Error as e:
            logger.error(f"Error fetching batch stats: {e}")
            
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
            "status": "IN CONTROL",
            "message": "Production line parameters are within healthy thresholds.",
            "trigger_nozzle_purge": False
        }
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (batch_id, rolling_window))
                results = [r[0] for r in cursor.fetchall()]
                
                if len(results) > 0:
                    defect_rate = sum(results) / len(results)
                    if defect_rate > 0.10:
                        alarms["status"] = "OUT OF CONTROL"
                        alarms["message"] = f"CRITICAL: Defect rate is {defect_rate*100:.1f}% in the last {len(results)} parts!"
                        alarms["trigger_nozzle_purge"] = True
                    elif defect_rate > 0.05:
                        alarms["status"] = "WARNING"
                        alarms["message"] = f"WARNING: Defect rate is elevated ({defect_rate*100:.1f}%) in current window."
                        
        except sqlite3.Error as e:
            logger.error(f"SPC evaluation failed: {e}")
            
        return alarms

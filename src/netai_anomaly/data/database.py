"""SQLite database management for network telemetry data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS network_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_host TEXT NOT NULL,
    destination_host TEXT NOT NULL,
    test_type TEXT NOT NULL,           -- throughput, latency, trace
    throughput_mbps REAL,
    latency_ms REAL,
    packet_loss_pct REAL,
    retransmits INTEGER,
    jitter_ms REAL,
    mtu INTEGER,
    tcp_window_size INTEGER,
    is_anomaly INTEGER DEFAULT 0,
    anomaly_type TEXT,                 -- slow_link, high_loss, excessive_retransmits, test_failure
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS traceroute_hops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL,
    hop_number INTEGER NOT NULL,
    hop_ip TEXT,
    hop_hostname TEXT,
    rtt_ms REAL,
    is_slow_hop INTEGER DEFAULT 0,
    FOREIGN KEY (test_id) REFERENCES network_tests(id)
);

CREATE TABLE IF NOT EXISTS network_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_host TEXT NOT NULL,
    destination_host TEXT NOT NULL,
    link_capacity_mbps REAL,
    typical_latency_ms REAL,
    hop_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tests_timestamp ON network_tests(timestamp);
CREATE INDEX IF NOT EXISTS idx_tests_hosts ON network_tests(source_host, destination_host);
CREATE INDEX IF NOT EXISTS idx_tests_anomaly ON network_tests(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_hops_test ON traceroute_hops(test_id);
"""


class TelemetryDB:
    """SQLite database interface for network telemetry."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def initialize(self) -> None:
        """Create database schema."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def insert_test(self, **kwargs) -> int:
        """Insert a network test result."""
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cursor = self.conn.execute(
            f"INSERT INTO network_tests ({columns}) VALUES ({placeholders})",
            list(kwargs.values()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_hop(self, **kwargs) -> int:
        """Insert a traceroute hop."""
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cursor = self.conn.execute(
            f"INSERT INTO traceroute_hops ({columns}) VALUES ({placeholders})",
            list(kwargs.values()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_tests_batch(self, tests: list[dict]) -> None:
        """Insert multiple test results in a single transaction."""
        if not tests:
            return
        columns = list(tests[0].keys())
        col_str = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))
        self.conn.executemany(
            f"INSERT INTO network_tests ({col_str}) VALUES ({placeholders})",
            [tuple(t[c] for c in columns) for t in tests],
        )
        self.conn.commit()

    def get_test_count(self) -> int:
        """Return the number of test records."""
        row = self.conn.execute("SELECT COUNT(*) FROM network_tests").fetchone()
        return row[0]

    def get_anomaly_count(self) -> int:
        """Return the number of anomalous records."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM network_tests WHERE is_anomaly = 1"
        ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

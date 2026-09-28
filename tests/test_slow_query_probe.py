"""The SQL probe logs call identity, never bound values or SQL text."""

import asyncio
import logging
import unittest

from sqlalchemy import create_engine, event
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import slow_query_probe


class SlowQueryProbeTests(unittest.TestCase):
    def test_slow_success_logs_fingerprint_without_sql_or_parameters(self):
        engine = create_engine("sqlite://")
        slow_query_probe.install(engine, warn_s=0)
        active_during_execute = []

        def inspect_in_flight(conn, cursor, statement, parameters, context, many):
            active_during_execute.append(slow_query_probe.active_snapshot())

        event.listen(engine, "before_cursor_execute", inspect_in_flight)
        statement = "SELECT 'private literal' AS value, ? AS bound_value"
        with self.assertLogs("app.db.slow_query_probe", logging.WARNING) as captured:
            with engine.connect() as conn:
                conn.exec_driver_sql(statement, ("private parameter",)).all()

        log = "\n".join(captured.output)
        fingerprint = slow_query_probe._fingerprint(statement)
        self.assertIn(f"sql_hmac={fingerprint}", log)
        self.assertIn("op=SELECT", log)
        self.assertIn("source=tests.test_slow_query_probe:", log)
        self.assertIn(fingerprint, active_during_execute[0])
        self.assertEqual(slow_query_probe.active_snapshot(), "-")
        self.assertNotIn("private literal", log)
        self.assertNotIn("private parameter", log)
        self.assertNotIn("private literal", active_during_execute[0])

    def test_failed_statement_is_removed_from_in_flight_snapshot(self):
        engine = create_engine("sqlite://")
        slow_query_probe.install(engine, warn_s=0)
        with self.assertLogs("app.db.slow_query_probe", logging.WARNING) as captured:
            with engine.connect() as conn:
                with self.assertRaises(DBAPIError):
                    conn.exec_driver_sql("SELECT * FROM missing_private_table")
        self.assertEqual(slow_query_probe.active_snapshot(), "-")
        self.assertIn("failed=1", "\n".join(captured.output))
        self.assertNotIn("missing_private_table", "\n".join(captured.output))

    def test_async_query_keeps_the_awaiting_call_site(self):
        async def run_query():
            engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            slow_query_probe.install(engine, warn_s=0)
            try:
                async with engine.connect() as conn:
                    await conn.exec_driver_sql("SELECT 1")
            finally:
                await engine.dispose()

        with self.assertLogs("app.db.slow_query_probe", logging.WARNING) as captured:
            asyncio.run(run_query())
        self.assertIn(
            "source=tests.test_slow_query_probe:", "\n".join(captured.output)
        )


if __name__ == "__main__":
    unittest.main()

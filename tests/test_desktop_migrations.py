"""Upgrade/downgrade the additive platform schema with Alembic operations."""
import importlib.util
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_106_107_schema_round_trip():
    modules = []
    for revision in ("0106", "0107"):
        path = next(Path("alembic/versions").glob(f"*_{revision}_*"))
        spec = importlib.util.spec_from_file_location(f"desktop_migration_{revision}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        modules.append(mod)
    assert modules[0].down_revision == "105"
    assert modules[1].down_revision == modules[0].revision
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE users (id VARCHAR(36) PRIMARY KEY)"))
        with Operations.context(MigrationContext.configure(connection)):
            for mod in modules: mod.upgrade()
            assert set(inspect(connection).get_table_names()) == {"users", "desktop_devices", "desktop_pairings", "desktop_pending_actions", "desktop_tasks"}
            columns = {c["name"] for c in inspect(connection).get_columns("desktop_pending_actions")}
            assert "remote_task_id" in columns
            for mod in reversed(modules): mod.downgrade()
        assert inspect(connection).get_table_names() == ["users"]

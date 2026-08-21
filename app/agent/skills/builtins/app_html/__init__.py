"""HTML-artifact app pipeline — an app is ONE self-contained .html file.

Replaces the React Native / Expo pipeline in ``../app_builder/``:
no scaffold, no ``node_modules`` (452 MiB and 27k files per app, measured
2026-08-20 — see ``MIGRATION_INVENTORY.md`` §3), no Metro, no dev-server
ports, no bundle-repair loop. One file on disk, one manifest row, rendered
in a sandboxed frame on a cookieless origin.
"""
